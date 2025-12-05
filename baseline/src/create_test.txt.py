import jsonlines


def main():
    argparser = argparse.ArgumentParser(description="Create test.txt from test_dataset.jsonl.")
    argparser.add_argument(
        "--input_path",
        type=str,
        default="test_dataset.jsonl",
        help="Path to the input JSONL file.")
    args = argparser.parse_args()

    output_file = open("test.txt", "w")

    for item in jsonlines.open(args.input_path, "r"):
        output_file.write(item["text"] + "\n")


if __name__ == "__main__":
    main()