import json

from tqdm import tqdm

if __name__ == "__main__":
    output_file = open("dev.txt", "w")

    filename = "data/ie_dataset_v3/val_dataset.jsonl"

    all_text = []
    for item in tqdm(open(filename)):
        raw_example = json.loads(item)
        all_text.append(raw_example["text"])
        output_file.write(raw_example["text"] + "\n")

    output_file = open("test.txt", "w")

    filename = "data/ie_dataset_v3/test_dataset.jsonl"

    all_text = []
    for item in tqdm(open(filename)):
        raw_example = json.loads(item)
        all_text.append(raw_example["text"])
        output_file.write(raw_example["text"] + "\n")

