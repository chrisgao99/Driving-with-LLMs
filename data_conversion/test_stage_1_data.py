from utils.waymo_training_utils import get_train_val_data
from utils.model_utils import load_llama_tokenizer


if __name__ == "__main__":
    base_model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = load_llama_tokenizer(base_model)
    train_data, val_data = get_train_val_data(
        data_path="/p/liverobotics/Rui/Driving-with-LLMs/stage_1_data/tfrecord-00100-of-00150.pkl",
        tokenizer=tokenizer,
        val_data_path=None,
        val_set_size=1,
        augment_times=1,
        load_pre_prompt_dataset=False,
        vqa=False,
        add_input_prompt=False,
        eval_only=False,
        eval_items=None,
    )

    print(train_data[0])