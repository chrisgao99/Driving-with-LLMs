# pylint: skip-file
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
import sys
import numpy as np
import openai
from retry import retry
from tqdm import tqdm
import json

from data_conversion.convert_data_utils import convert_to_descriptor_format
from data_conversion.convert_data import convert_data
from utils.new_prompt_utils import make_waymo_observation_prompt

client = openai.OpenAI(api_key="xx") # Replace "xx" with your actual OpenAI API key

def make_context():
    prompt = f"""I am a certified professional driving instructor and I am currently demonstrating driving in different scenarios of London to a student.
I have access to precise data for our car (the ego vehicle) and all surrounding objects. For every vehicle and pedestrian, I know their exact (x, y) coordinates, their speed, their heading, and their velocity vector (dx, dy), which tells us their direction and rate of movement.

I am also aware of the road network around us. I can see the layout of nearby individual road segments, and I know what type each one is, such as a 'Lane Center', 'Road Edge Boundary', 'Solid White Line', or 'Crosswalk'.

For each scenario, there will be a clear language instruction that describes how I should drive. 

My goal is to explain how I use this information to make safe and efficient driving decisions. I'm explaining what I see, what I'm paying attention to, and what I plan to do next based on the data.

Now, design 17 random question and answer pairs that the student might ask me about the current driving scenario. The answers should be based on the input data and my reasoning as an instructor. Ask diverse questions.

Format each QA pair in a single line as a strict JSON dictionary of {{'question': 'xxx', 'answer': 'xxx'}} with ',' as delimiter and complete open and close brakets. Only output 17 lines of single-line JSON. Do not include any other explanation.

You must include these 6 questions, but please rephrase them in a natural way:
- What are you observing in this scene?
- What are you paying attention to right now, and why?
- Are there any traffic lights? If so, what color are they?
- What is ego car's current state?
- What is ego car's driving plan for the next few seconds?
- Summarize the current driving scenario in high level / describe the current situation

When asked about ego car's driving plan, only return the answer by rephrasing the language instruction.

When asked about ego car's next waypoint, always answer with the ground truth next_waypoint.
"""
    return prompt

def make_prompt(language_instruction, next_waypoint, lang_gen):
    input_prompt = f"""The language instruction is: {language_instruction}. 

    The next waypoint of ego car is: {next_waypoint}.

    All the information I have is based on the current observation data: {lang_gen}.
    """
    return input_prompt

@retry(tries=1, delay=2, backoff=2)
def make_description_from_prompt(language_instruction, next_waypoint, lang_gen):
    global total_input_tokens, total_output_tokens
    
    context = make_context()
    input_prompt = make_prompt(language_instruction, next_waypoint, lang_gen)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": input_prompt},
        ],
        temperature=1.0,
    )
    first_response = response.choices[0].message.content
    # print("Response: ", first_response)
    return first_response

def process_single_timestep(args):
    """Process a single timestep - designed for parallel execution"""
    t, ego_traj, language_instruction, converted_data_t = args
    
    next_waypoint = ego_traj[t + 1] if t + 1 < ego_traj.shape[0] else ego_traj[t]
    lang_gen = make_waymo_observation_prompt(converted_data_t)
    
    data_with_qa = {
        "frame_num": t,
        "observation": converted_data_t, 
        "input_prompt": lang_gen,
        "response_content": None,
    }
    
    # Generate the description for this time step
    response = make_description_from_prompt(language_instruction, next_waypoint, lang_gen)
    response = response.split('\n')
    response = [json.loads(line) for line in response if line.strip() and line.startswith('{') and line.endswith('}')]
    response.append({"question": language_instruction + " Predict the next ego vehicle waypoint.", "answer": str(next_waypoint)})
    data_with_qa["response content"] = response
    
    return t, data_with_qa

def prepare_batch_requests(list_of_converted_data, sample_data):
    """Prepare batch requests for OpenAI API"""
    ego_traj = sample_data['Ego Trajectory']['trajectory']
    language_instruction = sample_data['Language Condition']
    
    batch_requests = []
    for t in range(ego_traj.shape[0]):
        next_waypoint = ego_traj[t + 1] if t + 1 < ego_traj.shape[0] else ego_traj[t]
        lang_gen = make_waymo_observation_prompt(list_of_converted_data[t])
        
        context = make_context()
        input_prompt = make_prompt(language_instruction, next_waypoint, lang_gen)
        
        request = {
            "custom_id": f"timestep_{t}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": input_prompt},
                ],
                "temperature": 1.0,
            }
        }
        batch_requests.append(request)
    
    return batch_requests

def create_and_process_batch(batch_requests):
    """Create and process batch requests with OpenAI"""
    import tempfile
    import time
    
    batch_file_path = None
    try:
        # Write batch requests to a JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            batch_file_path = f.name
                
        # Upload the batch file
        with open(batch_file_path, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
                
        # Create the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "VQA batch processing"}
        )
                
        # Poll for completion with timeout
        max_wait_time = 14400  # 4 hour timeout
        start_time = time.time()
        check_interval = 30
        
        while batch.status in ["validating", "in_progress", "finalizing"]:
            elapsed_time = time.time() - start_time
            
            if elapsed_time > max_wait_time:
                print(f"Batch timeout after {max_wait_time}s. Current status: {batch.status}")
                raise TimeoutError(f"Batch processing timeout after {max_wait_time} seconds")
            
            time.sleep(check_interval)
            
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:
                print(f"Error retrieving batch status: {e}")
                time.sleep(check_interval)
                continue
                
        if batch.status == "completed":
            # Download results
            result_file_id = batch.output_file_id
            print(f"Downloading results from file ID: {result_file_id}")
            result = client.files.content(result_file_id)
            return result.content.decode('utf-8')
        elif batch.status == "failed":
            print(f"Batch failed. Error details: {getattr(batch, 'errors', 'No error details available')}")
            raise Exception(f"Batch failed with status: {batch.status}")
        else:
            raise Exception(f"Unexpected batch status: {batch.status}")
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise
    finally:
        # Clean up temporary file
        if batch_file_path and os.path.exists(batch_file_path):
            try:
                os.unlink(batch_file_path)
                print(f"Cleaned up temporary file: {batch_file_path}")
            except Exception as e:
                print(f"Failed to clean up temporary file {batch_file_path}: {e}")

def parse_batch_results(batch_results_content):
    """Parse batch results from JSONL format"""
    results = {}
    for line in batch_results_content.strip().split('\n'):
        if line:
            result = json.loads(line)
            custom_id = result['custom_id']
            timestep = int(custom_id.split('_')[1])
            response_content = result['response']['body']['choices'][0]['message']['content']
            results[timestep] = response_content
    return results

def prepare_batch_requests_multi_scene(total_scene_data, timesteps_per_scene=None):
    """Prepare batch requests for multiple scenes with optional timestep sampling"""
    batch_requests = []
    scene_timestep_mapping = {}  # To track which request belongs to which scene/timestep
    
    for scene_id, scene_info in total_scene_data.items():
        sample_data = scene_info["scene_data"]
        list_of_converted_data = scene_info["converted_data"]
        
        ego_traj = sample_data['Ego Trajectory']['trajectory']
        language_instruction = sample_data['Language Condition']
        
        # Sample timesteps if specified
        if timesteps_per_scene is not None:
            max_timesteps = min(timesteps_per_scene, ego_traj.shape[0])
            sampled_indices = np.random.choice(ego_traj.shape[0], size=max_timesteps, replace=False)
            sampled_indices = sorted(sampled_indices)  # Keep temporal order
        else:
            sampled_indices = list(range(ego_traj.shape[0]))
        
        for t in sampled_indices:
            next_waypoint = ego_traj[t + 1] if t + 1 < ego_traj.shape[0] else ego_traj[t]
            lang_gen = make_waymo_observation_prompt(list_of_converted_data[t])
            
            context = make_context()
            input_prompt = make_prompt(language_instruction, next_waypoint, lang_gen)
            
            custom_id = f"scene_{scene_id}_timestep_{t}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": input_prompt},
                    ],
                    "temperature": 1.0,
                }
            }
            batch_requests.append(request)
            scene_timestep_mapping[custom_id] = (scene_id, t)
    
    return batch_requests, scene_timestep_mapping

def parse_batch_results_multi_scene(batch_results_content, scene_timestep_mapping):
    """Parse batch results for multiple scenes"""
    results = {}
    for line in batch_results_content.strip().split('\n'):
        if line:
            result = json.loads(line)
            custom_id = result['custom_id']
            scene_id, timestep = scene_timestep_mapping[custom_id]
            response_content = result['response']['body']['choices'][0]['message']['content']
            
            if scene_id not in results:
                results[scene_id] = {}
            results[scene_id][timestep] = response_content
    return results

def get_qa_descriptor(total_scene_data, output_dir=None, timesteps_per_scene=None, use_batch=True, use_parallel=False, num_workers=None, batch_size_limit=1000):
    """
    Process all scenes in total_scene_data with optional timestep sampling
    
    Args:
        total_scene_data: Dict with scene_id as key, containing "scene_data" and "converted_data"
        output_dir: Directory to save results after each batch (required for batch processing)
        timesteps_per_scene: Number of timesteps to sample per scene (None for all timesteps)
        use_batch: Whether to use OpenAI batch API
        use_parallel: Whether to use parallel processing (ignored if use_batch=True)
        num_workers: Number of workers for parallel processing
        batch_size_limit: Maximum number of requests per batch (default: 1000)
    """
    
    if use_batch:
        if output_dir is None:
            raise ValueError("output_dir must be provided when using batch processing")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Use OpenAI batch API for all scenes with chunking
        batch_requests, scene_timestep_mapping = prepare_batch_requests_multi_scene(
            total_scene_data, timesteps_per_scene
        )
        
        all_batch_results = {}
        
        # Process requests in chunks
        for i in tqdm(range(0, len(batch_requests), batch_size_limit), desc="Processing batches"):
            chunk_requests = batch_requests[i:i + batch_size_limit]
            chunk_mapping = {k: v for k, v in scene_timestep_mapping.items() 
                           if k in [req['custom_id'] for req in chunk_requests]}
            
            chunk_results_content = create_and_process_batch(chunk_requests)
            chunk_results = parse_batch_results_multi_scene(chunk_results_content, chunk_mapping)
            
            # Merge chunk results
            for scene_id, timestep_results in chunk_results.items():
                if scene_id not in all_batch_results:
                    all_batch_results[scene_id] = {}
                all_batch_results[scene_id].update(timestep_results)
            
            # Process and save results after each chunk
            current_scenes_qa_data = {}
            for scene_id, scene_info in total_scene_data.items():
                if scene_id not in all_batch_results:
                    continue
                    
                sample_data = scene_info["scene_data"]
                list_of_converted_data = scene_info["converted_data"]
                ego_traj = sample_data['Ego Trajectory']['trajectory']
                language_instruction = sample_data['Language Condition']
                
                scene_qa_data = []
                scene_results = all_batch_results.get(scene_id, {})
                
                for timestep in sorted(scene_results.keys()):
                    next_waypoint = ego_traj[timestep + 1] if timestep + 1 < ego_traj.shape[0] else ego_traj[timestep]
                    lang_gen = make_waymo_observation_prompt(list_of_converted_data[timestep])
                    
                    data_with_qa = {
                        "frame_num": timestep,
                        "observation": list_of_converted_data[timestep], 
                        "input_prompt": lang_gen,
                        "response_content": None,
                    }
                    
                    # Parse the batch response
                    response = scene_results[timestep]
                    response = response.split('\n')
                    try:
                        response = [json.loads(line) for line in response if line.strip() and line.startswith('{') and line.endswith('}')]
                    except:
                        continue
                    response.append({"question": language_instruction + " Predict the next ego vehicle waypoint.", "answer": str(next_waypoint)})
                    data_with_qa["response content"] = response
                    scene_qa_data.append(data_with_qa)
                
                current_scenes_qa_data[scene_id] = scene_qa_data
            
            # Save current results after each chunk
            chunk_output_path = os.path.join(output_dir, f"all_scenes_qa_data_chunk_{i//batch_size_limit + 1}.pkl")
            with open(chunk_output_path, "wb") as f:
                pickle.dump(current_scenes_qa_data, f)
        
        # Organize final results by scene
        all_scenes_qa_data = {}
        for scene_id, scene_info in total_scene_data.items():
            sample_data = scene_info["scene_data"]
            list_of_converted_data = scene_info["converted_data"]
            ego_traj = sample_data['Ego Trajectory']['trajectory']
            language_instruction = sample_data['Language Condition']
            
            scene_qa_data = []
            scene_results = all_batch_results.get(scene_id, {})
            
            for timestep in sorted(scene_results.keys()):
                next_waypoint = ego_traj[timestep + 1] if timestep + 1 < ego_traj.shape[0] else ego_traj[timestep]
                lang_gen = make_waymo_observation_prompt(list_of_converted_data[timestep])
                
                data_with_qa = {
                    "frame_num": timestep,
                    "observation": list_of_converted_data[timestep], 
                    "input_prompt": lang_gen,
                    "response_content": None,
                }
                
                # Parse the batch response
                response = scene_results[timestep]
                response = response.split('\n')
                response = [json.loads(line) for line in response if line.strip() and line.startswith('{') and line.endswith('}')]
                response.append({"question": language_instruction + " Predict the next ego vehicle waypoint.", "answer": str(next_waypoint)})
                data_with_qa["response content"] = response
                scene_qa_data.append(data_with_qa)
            
            all_scenes_qa_data[scene_id] = scene_qa_data
            
        return all_scenes_qa_data
    else:
        # Fallback to processing scenes individually (existing logic)
        all_scenes_qa_data = {}
        for scene_id, scene_info in total_scene_data.items():
            sample_data = scene_info["scene_data"]
            list_of_converted_data = scene_info["converted_data"]
            
            # Apply timestep sampling if specified
            if timesteps_per_scene is not None:
                ego_traj = sample_data['Ego Trajectory']['trajectory']
                max_timesteps = min(timesteps_per_scene, ego_traj.shape[0])
                sampled_indices = np.random.choice(ego_traj.shape[0], size=max_timesteps, replace=False)
                sampled_indices = sorted(sampled_indices)
                list_of_converted_data = [list_of_converted_data[i] for i in sampled_indices]
                
                # Create a modified sample_data with sampled trajectory
                modified_sample_data = sample_data.copy()
                modified_sample_data['Ego Trajectory'] = {'trajectory': ego_traj[sampled_indices]}
            else:
                modified_sample_data = sample_data
            
            # Process single scene (reuse existing logic)
            if use_parallel:
                args_list = [
                    (i, modified_sample_data['Ego Trajectory']['trajectory'], modified_sample_data['Language Condition'], list_of_converted_data[i])
                    for i in range(len(list_of_converted_data))
                ]
                
                with Pool(processes=num_workers) as pool:
                    results = list(tqdm(pool.imap(process_single_timestep, args_list), 
                                      total=len(args_list), 
                                      desc=f"Processing scene {scene_id}"))
                
                results.sort(key=lambda x: x[0])
                scene_qa_data = [result[1] for result in results]
            else:
                scene_qa_data = []
                ego_traj = modified_sample_data['Ego Trajectory']['trajectory']
                language_instruction = modified_sample_data['Language Condition']
                for t in range(len(list_of_converted_data)):
                    _, data_with_qa = process_single_timestep((t, ego_traj, language_instruction, list_of_converted_data[t]))
                    scene_qa_data.append(data_with_qa)
            
            all_scenes_qa_data[scene_id] = scene_qa_data
        
        return all_scenes_qa_data

def process_tfrecord_for_qa(tfrecord_path, scene_sampling=10):
    try:
        vector_data = convert_data(tfrecord_path)
    except Exception as e:
        print(f"Failed to convert {tfrecord_path}: {e}")
        return None
    
    # sample scenes from vector_data
    scene_keys = list(vector_data.keys())
    sampled_scene_keys = np.random.choice(scene_keys, size=min(scene_sampling, len(scene_keys)), replace=False)
    sampled_vector_data = {key: vector_data[key] for key in sampled_scene_keys}

    output_data = {}

    for sid_egoid, scene_data in sampled_vector_data.items():
        output_data[sid_egoid] = {}
        output_data[sid_egoid]["scene_data"] = scene_data
        converted_data = convert_to_descriptor_format(scene_data)
        output_data[sid_egoid]["converted_data"] = converted_data

    return output_data  
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_stage_1_data.py <batch_dir>")
        sys.exit(1)

    batch_dir = sys.argv[1]
    output_dir = "/p/ruishen/dllm_waymo_data/stage_2_data"

    np.random.seed(42)

    # total_scene_data = {}

    # total_batch_dir = os.listdir(batch_dir)
    # total_batch_dir = np.random.choice(total_batch_dir, size=20, replace=False).tolist()

    # for batch_file in tqdm(total_batch_dir, desc="Processing batches"):
    #     if not batch_file.endswith(".txt"):
    #         continue
    #     batch_path = os.path.join(batch_dir, batch_file)
    #     with open(batch_path, "r") as f:
    #         for line in f:
    #             tfrecord_path = line.strip()
    #             if not tfrecord_path:
    #                 continue
    #             get_converted_data = process_tfrecord_for_qa(tfrecord_path, scene_sampling=20)
    #             if get_converted_data is None:
    #                 continue
    #             total_scene_data.update(get_converted_data)

    # os.makedirs(output_dir, exist_ok=True)
    # with open(os.path.join(output_dir, "total_scene_data.pkl"), "wb") as f:
    #     pickle.dump(total_scene_data, f)

    with open("/p/ruishen/dllm_waymo_data/stage_2_data/total_scene_data.pkl", "rb") as f:
        total_scene_data = pickle.load(f)

    # Process all scenes with timestep sampling and batch size limit
    all_scenes_qa_data = get_qa_descriptor(total_scene_data, output_dir, timesteps_per_scene=3, use_batch=True, batch_size_limit=100)
    
    output_path = os.path.join(output_dir, "all_scenes_qa_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_scenes_qa_data, f)
