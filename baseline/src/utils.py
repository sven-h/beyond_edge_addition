import json
import random


def load_json_data(path: str) -> object:
    """Load and return the contents of a JSON file."""
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_data(path: str) -> list:
    """Load and return all records from a JSONL file as a list of dicts."""
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            obj = json.loads(line.strip())
            data.append(obj)
    return data


def dump_json_data(data: object, path: str) -> None:
    """Write data to a JSON file with standard formatting."""
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=True,
                  indent=2, separators=(", ", ": "))


def shorten_entity_description(entity_description: str, max_len: int) -> str:
    """Truncate an entity description to at most max_len whitespace-separated tokens."""
    entity_description_tokens = entity_description.split(" ")
    entity_description = ' '.join(entity_description_tokens[: max_len])
    return entity_description


def formulate_candidates(candidate_list: list, max_len: int) -> str:
    """Format a shuffled list of candidate entities into a numbered prompt string."""
    candidates = ""
    candidate_template = '\n\nID: {}\nEntity: {}\nEntity Description: {}\nEntity Types: {}'
    random.shuffle(candidate_list)
    for i, candidate_obj in enumerate(candidate_list):
        entity_description = shorten_entity_description(
            candidate_obj["entity_description"], max_len)
        candidate = candidate_template.format(
            i, candidate_obj["title"], entity_description, ", ".join(candidate_obj["entity_types"]))
        candidates += candidate

    return candidates


def is_length_valid(model_path: str, human_value: str, gpt_value: str, tokenizer, max_input_length: int = 4000) -> bool:
    """Return True if the tokenized chat prompt is within max_input_length tokens."""
    messages = [
        {"role": "user", "content": human_value},
        {"role": "assistant", "content": gpt_value}
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer([prompt])
    input_length = len(inputs["input_ids"][0])

    if random.randint(1, 100) == 1:
        print(f"vicuna input length + output length = {input_length}")

    if input_length > max_input_length:
        return False

    return True