# pylint: skip-file
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import openai
from retry import retry
from tqdm import tqdm
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.new_prompt_utils import make_waymo_observation_prompt

client = openai.OpenAI(api_key="xx") # Replace "xx" with your actual OpenAI API key

def make_context():
    prompt = f"""I am a certified professional driving instructor and I am currently demonstrating driving in different scenarios of London to a student.
I have access to precise data for our car (the ego vehicle) and all surrounding objects. For every vehicle and pedestrian, I know their exact (x, y) coordinates, their speed, their heading, and their velocity vector (dx, dy), which tells us their direction and rate of movement.

I am also aware of the road network around us. I can see the layout of nearby individual road segments, and I know what type each one is, such as a 'Lane Center', 'Road Edge Boundary', 'Solid White Line', or 'Crosswalk'.

For each scenario, there will be a clear language instruction that describes how I should drive. 

My goal is to explain how I use this information to make safe and efficient driving decisions. I'm explaining what I see, what I'm paying attention to, and what I plan to do next based on the data.

Now, design 17 random question and answer pairs that the student might ask me about the current driving scenario. The answers should be based on the input data and my reasoning as an instructor. Ask diverse questions.

Format each QA pair in a single line as a JSON dictionary like {{"question": "xxx", "answer": "xxx"}}. Only output 17 lines of single-line JSON. Do not include any other explanation.

You must include these 7 questions, but please rephrase them in a natural way:
- What are you observing in this scene?
- What are you paying attention to right now, and why?
- Are there any traffic lights? If so, what color are they?
- What is ego car's current state?
- What is ego car's driving plan for the next few seconds?
- Summarize the current driving scenario in high level / describe the current situation
- Based on the language instruction, predict the next waypoint of the ego car. (include the real language instruction in the question)

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
    context = make_context()
    input_prompt = make_prompt(language_instruction, next_waypoint, lang_gen)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": input_prompt},
        ],
        temperature=1.0,
    )
    first_response = response.choices[0].message.content
    # print("Response: ", first_response)
    return first_response

def get_qa_descriptor(list_of_converted_data, sample_data):
    ego_traj = sample_data['Ego Trajectory']['trajectory']
    list_data_with_qa = []
    for t in range(ego_traj.shape[0]):
        next_waypoint = ego_traj[t + 1] if t + 1 < ego_traj.shape[0] else ego_traj[t]
        language_instruction = sample_data['Language Condition']
        lang_gen = make_waymo_observation_prompt(list_of_converted_data[t])
        data_with_qa = {"observation": list_of_converted_data[t], "input_prompt": lang_gen,"response_content": None,}
        
        # Generate the description for this time step
        response = make_description_from_prompt(language_instruction, next_waypoint, lang_gen)
        data_with_qa["response content"] = response
        list_data_with_qa.append(data_with_qa)

    return list_data_with_qa




if __name__ == "__main__":
    sample_data = {
        'Map Data': [{'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.56982422, 7806.85546875]), np.array([1963.52709961, 7843.18310547])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.96679688, 7806.84423828]), np.array([1966.98547363, 7843.69970703])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.93896484, 7785.59375]), np.array([1966.95495605, 7793.99853516]), np.array([1969.23913574, 7797.90966797]), np.array([1970.70812988, 7798.00439453])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.94824219, 7793.01367188]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.64428711, 7792.94677734]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1969.01513672, 7801.32470703]), np.array([1967.06616211, 7801.60839844]), np.array([1963.73632812, 7804.89355469]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1962.15856934, 7801.75683594])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1967.05712891, 7804.48193359]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1967.02539062, 7801.43017578]), np.array([1962.11303711, 7802.08984375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1968.0501709, 7801.18896484]), np.array([1962.22180176, 7800.45507812])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.92077637, 7759.53564453]), np.array([1966.51672363, 7776.18798828]), np.array([1966.83496094, 7780.14111328]), np.array([1966.93896484, 7785.59375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.57446289, 7760.57958984]), np.array([1963.64428711, 7792.94677734])]}, {'type': 'RoadLine-BrokenSingleWhite', 'pos_xy': [np.array([1961.87744141, 7761.74658203]), np.array([1961.93127441, 7802.21337891])]}, {'type': 'RoadLine-SolidSingleWhite', 'pos_xy': [np.array([1965.40332031, 7785.50488281]), np.array([1965.16125488, 7843.51123047])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.43725586, 7822.7265625]), np.array([1968.78320312, 7845.12158203])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.4642334, 7806.51367188]), np.array([1968.43725586, 7822.7265625])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1967.46484375, 7760.70117188]), np.array([1968.63195801, 7782.35400391]), np.array([1968.51171875, 7794.00048828]), np.array([1971.43725586, 7794.03027344])]}, {'type': 'Crosswalk', 'pos_xy': [np.array([1965.60144043, 7845.03515625]), np.array([1968.67199707, 7845.16992188])]}],
        'Ego Trajectory': {'type': 'Vehicle', 'trajectory': np.array([[-1.00000000e+00, -1.00000000e+00], [1.96526404e+03, 7.76571191e+03], [1.96557971e+03, 7.77292139e+03], [1.96608276e+03, 7.78193604e+03], [1.96643823e+03, 7.78937842e+03], [1.96669189e+03, 7.79597900e+03], [1.96683398e+03, 7.80195312e+03], [1.96690027e+03, 7.80714648e+03], [1.96693176e+03, 7.81174316e+03], [1.96695154e+03, 7.81586426e+03], [1.96697522e+03, 7.81946924e+03], [1.96698767e+03, 7.82249658e+03], [1.96699011e+03, 7.82487500e+03], [1.96699292e+03, 7.82656885e+03], [1.96699207e+03, 7.82779785e+03], [1.96698657e+03, 7.82872119e+03], [1.96698218e+03, 7.82952588e+03], [1.96697607e+03, 7.83041113e+03], [1.96697656e+03, 7.83131445e+03]])},
        'Nearby Agent Trajectories': {650: {'type': 'Vehicle', 'trajectory': np.array([[1966.87072754, 7828.43798828], [1966.87475586, 7828.45458984], [1966.86901855, 7828.484375], [1966.86706543, 7828.66162109], [1966.86401367, 7829.0703125], [1966.85656738, 7829.80566406], [1966.84875488, 7830.76269531], [1966.83544922, 7831.95849609], [1966.8236084, 7833.11181641], [1966.81616211, 7834.24609375], [1966.81835938, 7835.26904297], [1966.81335449, 7836.15917969], [1966.80432129, 7836.77832031], [1966.7989502, 7837.21044922], [1966.81237793, 7837.69873047], [1966.82092285, 7838.26416016], [1966.83056641, 7838.88818359], [1966.83239746, 7839.64648438], [1966.84216309, 7840.64794922]])}, 672: {'type': 'Vehicle', 'trajectory': np.array([[1964.07141113, 7790.80615234], [1964.06347656, 7795.94921875], [1964.04870605, 7800.78564453], [1964.0435791, 7805.06054688], [1964.01977539, 7809.04199219], [1964.00061035, 7812.54638672], [1963.9921875, 7815.69042969], [1963.96240234, 7818.52832031], [1963.95715332, 7821.07714844], [1963.95263672, 7823.296875], [1963.94311523, 7825.2109375], [1963.94128418, 7826.83349609], [1963.94030762, 7828.14355469], [1963.94372559, 7829.20605469], [1963.94592285, 7830.00097656], [1963.94299316, 7830.61865234], [1963.94763184, 7831.14111328], [1963.94812012, 7831.54931641], [1963.9543457, 7831.82324219]])}},
        'Language Condition': "The vehicle comes to a complete stop.",
    }

    from data_conversion.convert_data_utils import convert_to_descriptor_format

    list_of_converted_data = convert_to_descriptor_format(sample_data)
    print("list_of_converted_data: ", list_of_converted_data)


    list_data_with_qa = get_qa_descriptor(list_of_converted_data, sample_data)

    print("QA lists: ", list_data_with_qa)
