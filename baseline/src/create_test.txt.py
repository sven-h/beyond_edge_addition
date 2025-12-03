import jsonlines


def main():

    path = "test_dataset.jsonl"

    output_file = open("test.txt", "w")

    for item in jsonlines.open(path, "r"):
        output_file.write(item["text"] + "\n")


if __name__ == "__main__":
    main()