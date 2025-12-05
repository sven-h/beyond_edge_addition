import argparse

import jsonlines

def generate_text_file(items: list, output_file: str):
    output_file = open(output_file, "w")

    for item in items:
        output_file.write(item["text"] + "\n")
def main():
    argparser = argparse.ArgumentParser(description="Create test.txt from test_dataset.jsonl.")
    argparser.add_argument(
        "--input_path",
        type=str,
        default="test_dataset.jsonl",
        help="Path to the input JSONL file.")
    args = argparser.parse_args()


    generate_text_file(list(jsonlines.open(args.input_path, "r")), "test.txt")


if __name__ == "__main__":
    main()