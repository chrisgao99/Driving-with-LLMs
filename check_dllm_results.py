import json

generation_path = "/p/liverobotics/Rui/Driving-with-LLMs/eval_output.json"

with open(generation_path, "r") as f:
    data = f.readlines()
data = [json.loads(line) for line in data]

for item in data:
    pred = item['pred']
    pred = pred.split("Response:")[-1].strip()
    label = item['label']
    print(f"Prediction: {pred}")
    print(f"Label: {label}")
    breakpoint()
