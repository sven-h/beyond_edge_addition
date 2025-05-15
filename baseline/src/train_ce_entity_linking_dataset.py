import csv
import json
import argparse
import logging
import os
import pickle
import random
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, CrossEncoderTrainingArguments, CrossEncoderTrainer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from tqdm import tqdm
from typing import List
from datasets import Dataset,DatasetDict
from src.data_utils import load_data, KGContainer, Example, create_full_entity_description

INSTRUCT = '''Entity Mention: {}\nEntity Mention Definition: {}\nEntity Mention Types: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers:{}'''

INSTRUCT_WITH_NONE_CASE = '''Entity Mention: {}\nEntity Mention Definition: {}\nEntity Mention Types: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers (if none of them, assign the ID as "None"):{}'''


CANDIDATE_NUM = 10

MAX_INPUT_LENGTH = 4000


def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            obj = json.loads(line.strip())
            data.append(obj)
    return data


def dump_json_data(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=True,
                  indent=2, separators=(", ", ": "))


def shorten_entity_description(entity_description, max_len):
    entity_description_tokens = entity_description.split(" ")
    entity_description = ' '.join(entity_description_tokens[: max_len])
    return entity_description


def formulate_candidates(candidate_list, max_len):
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


def is_length_valid(model_path, human_value, gpt_value, tokenizer):
    messages = [
        {"role": "user", "content": human_value},
        {"role": "assistant", "content": gpt_value}
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # vicuna_convo = get_conversation_template(model_path)
    # vicuna_convo.append_message(vicuna_convo.roles[0], human_value)
    # vicuna_convo.append_message(vicuna_convo.roles[1], gpt_value)
    # prompt = vicuna_convo.get_prompt()

    inputs = tokenizer([prompt])
    input_length = len(inputs["input_ids"][0])

    if random.randint(1, 100) == 1:
        print(f"vicuna input length + output length = {input_length}")

    if input_length > MAX_INPUT_LENGTH:
        return False

    return True

def create_mention_mention(sft_data):
    positive_to_index = defaultdict(list)
    potential_negatives = defaultdict(set)
    for idx, item in enumerate(sft_data):
        mention = item["query"]
        positive = item["positive"]
        positive_to_index[positive].append(idx)
        for i in range(1, len(item) - 2):
            negative = item[f"negative_{i}"]
            potential_negatives[positive].add(negative)
    potential_negatives = {k: [x for x in v  if x in positive_to_index] for k, v in potential_negatives.items()}
    for positive in tqdm(positive_to_index):
        random.shuffle(positive_to_index[positive])
        positive_candidates = positive_to_index[positive][:10]
        negatives = potential_negatives[positive]
        random.shuffle(negatives)
        final_negatives = []
        for neg in negatives:
            neg_indices = positive_to_index[neg]
            neg_indices = neg_indices[:2]
            final_negatives.extend(neg_indices)

        if len(final_negatives) ==0:
            continue

        if len(positive_candidates) > 1:
            # Take each succeeding pair of positive candidates as new examples
            for i in range(len(positive_candidates) - 1):
                mention = sft_data[positive_candidates[i]]["query"]
                positive = sft_data[positive_candidates[i + 1]]["query"]
                negative_indices = random.sample(final_negatives, min(10, len(final_negatives)))
                negative = [sft_data[idx]["query"] for idx in negative_indices]
                example = {
                    "query": mention,
                    "positive": positive,
                    **{f"negative_{i + 1}": negative[i] for i in range(len(negative))},}
                sft_data.append(example)
    return sft_data



def create_sft_data(context_candidates_list, add_mentions):
    sft_data = []
    for each in tqdm(context_candidates_list):
        exp = {"mention_id": each["mention_id"],
               "mention": each["mention"]}
        mention_types = [x for x in each["mention_types"] if isinstance(x, str)]

        mention_types_str = ", ".join(mention_types)

        anchor = f"{each['mention']} ({mention_types_str}): {each['mention_definition']}"
        positive_candidate = None
        negative_candidates = []
        for idx, candidate in enumerate(each["candidates"]):
            candidate_types = [x for x in candidate["entity_types"] if isinstance(x, str)]
            candidate_types_str = ", ".join(candidate_types)
            candidate_description = f"{candidate['title']} ({candidate_types_str}): {candidate['entity_description']}"
            if each["label_id"] == idx:
                positive_candidate = candidate_description
            else:
                negative_candidates.append(candidate_description)
        if positive_candidate is None:
            continue


        example = {
            "query": anchor,
            "positive": positive_candidate,
            **{f"negative_{i + 1}": negative_candidates[i] for i in range(len(negative_candidates))},
        }
        sft_data.append(example)
    if add_mentions:
        sft_data = create_mention_mention(sft_data)
    return sft_data




def get_retrieval_elements(entity_index, entity_mapping, candidate_retrieval_model) -> tuple:

    model = SentenceTransformer(candidate_retrieval_model, trust_remote_code=True)
    faiss_index = faiss.read_index(entity_index)  # Replace with actual index
    index_mapping = json.load(open(entity_mapping))  # Replace with actual index mapping
    index_mapping = {int(k): v for k, v in index_mapping.items()}

    return model, faiss_index, index_mapping




def retrieve_candidates(entities,
                        model, candidate_index, candidate_mapping) -> set:
    prompts = []
    prompt_to_add = ""
    for entity in entities:
        prompt = prompt_to_add + create_full_entity_description(entity.candidate_mention, entity.candidate_definition
                                                                , entity.candidate_types)
        prompts.append(prompt)

    embeddings = model.encode(prompts, show_progress_bar=True, normalize_embeddings=True)
    nearest_indices = candidate_index.search(embeddings, 2 * CANDIDATE_NUM)[1]
    all_candidates = []
    for nearest_indices_ in nearest_indices:
        candidates = set()
        for idx in nearest_indices_:
            if idx in candidate_mapping:
                candidates.add(candidate_mapping[idx]["identifier"])
        all_candidates.append(candidates)
    return all_candidates



def prepare_context_candidates(data: List[Example], entity_index, entity_mapping, candidate_retrieval_model):
    model, candidate_index, candidate_mapping = get_retrieval_elements(entity_index, entity_mapping, candidate_retrieval_model)

    all_entities = []
    for item in data:
        for entity in item.entities:
            if entity.candidate_mention is None:
                continue
            all_entities.append(entity)
    all_candidates = retrieve_candidates(
                all_entities,
                model, candidate_index, candidate_mapping)
    none_case_num = 0
    context_candidates_list = []
    counter = 0
    recall = 0
    for entity, candidates in zip(all_entities, all_candidates):
        if entity.candidate_mention is None:
            continue
        candidates = list(candidates)
        if entity.qid in candidates:
            recall += 1
        else:
            candidates.append(entity.qid)
        candidates_list = [candidates[:CANDIDATE_NUM]]
        if random.random() < 0.25:
            if entity.qid in candidates:
                candidates_copy = candidates.copy()
                candidates_copy.remove(entity.qid)
                candidates_list.append(candidates_copy[:CANDIDATE_NUM])
        for candidates in candidates_list:
            random.shuffle(candidates)
            if entity.qid not in candidates:
                continue
            else:
                label_id = candidates.index(entity.qid)
            candidate_reps = []
            for candidate_qid in candidates:
                label = kg_container.label(candidate_qid)
                description = kg_container.definition(candidate_qid)
                entity_types = kg_container.types(candidate_qid)
                candidate = {
                    "title": label,
                    "entity_description": description.strip(),
                    "entity_types": entity_types[:3],
                }
                candidate_reps.append(candidate)

            context_candidates = {
                "mention_id": counter,
                "mention": entity.candidate_mention,
                "mention_definition": entity.candidate_definition,
                "mention_types": entity.candidate_types,
                "candidates": candidate_reps,
                "label_id": label_id
            }
            counter += 1
            context_candidates_list.append(context_candidates)

    print(f"num of none case: {none_case_num} out of {counter}")
    print(f"recall: {recall/len(all_entities)} out of {len(all_entities)}")
    return context_candidates_list

logger = logging.getLogger(__name__)
class CustomCrossEncoderRerankingEvaluator(CrossEncoderRerankingEvaluator):
    def __call__(
            self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"CrossEncoderRerankingEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        base_mrr_scores = []
        base_ndcg_scores = []
        base_ap_scores = []
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        detection_tp = 0
        detection_fp = 0
        detection_fn = 0
        for instance in tqdm(self.samples, desc="Evaluating samples", disable=not self.show_progress_bar, leave=False):
            if "query" not in instance:
                raise ValueError("CrossEncoderRerankingEvaluator requires a 'query' key in each sample.")
            if "positive" not in instance:
                raise ValueError("CrossEncoderRerankingEvaluator requires a 'positive' key in each sample.")
            if ("negative" in instance and "documents" in instance) or (
                    "negative" not in instance and "documents" not in instance
            ):
                raise ValueError(
                    "CrossEncoderRerankingEvaluator requires exactly one of 'negative' and 'documents' in each sample."
                )

            query = instance["query"]
            positive = instance["positive"]
            if isinstance(positive, str):
                positive = [positive]

            negative = instance.get("negative", None)
            documents = instance.get("documents", None)

            if documents:
                base_is_relevant = [int(sample in positive) for sample in documents]
                if sum(base_is_relevant) == 0:
                    base_mrr, base_ndcg, base_ap = 0, 0, 0
                else:
                    # If not all positives are in documents, we need to add them at the end
                    base_is_relevant += [1] * (len(positive) - sum(base_is_relevant))
                    base_pred_scores = np.array(range(len(base_is_relevant), 0, -1))
                    base_mrr, base_ndcg, base_ap = self.compute_metrics(base_is_relevant, base_pred_scores)
                base_mrr_scores.append(base_mrr)
                base_ndcg_scores.append(base_ndcg)
                base_ap_scores.append(base_ap)

                if self.always_rerank_positives:
                    docs = positive + [doc for doc in documents if doc not in positive]
                    is_relevant = [1] * len(positive) + [0] * (len(docs) - len(positive))
                else:
                    docs = documents
                    is_relevant = [int(sample in positive) for sample in documents]
            else:
                docs = positive + negative
                is_relevant = [1] * len(positive) + [0] * len(negative)

            num_queries += 1

            num_positives.append(len(positive))
            num_negatives.append(len(is_relevant) - sum(is_relevant))

            if sum(is_relevant) == 0:
                all_mrr_scores.append(0)
                all_ndcg_scores.append(0)
                all_ap_scores.append(0)
                continue

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)

            detection_tp += pred_scores[0] > 0.5
            detection_fn += pred_scores[0] <= 0.5
            for i in range(1, len(pred_scores)):
                if pred_scores[i] > 0.5:
                    detection_fp += 1

            # Add the ignored positives at the end
            if num_ignored_positives := len(is_relevant) - len(pred_scores):
                pred_scores = np.concatenate([pred_scores, np.zeros(num_ignored_positives)])

            mrr, ndcg, ap = self.compute_metrics(is_relevant, pred_scores)

            all_mrr_scores.append(mrr)
            all_ndcg_scores.append(ndcg)
            all_ap_scores.append(ap)

        recall = detection_tp / (detection_tp + detection_fn) if (detection_tp + detection_fn) > 0 else 0
        precision = detection_tp / (detection_tp + detection_fp) if (detection_tp + detection_fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)
        mean_ap = np.mean(all_ap_scores)
        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

        logger.info(
            f"Queries: {num_queries}\t"
            f"Positives: Min {np.min(num_positives):.1f}, Mean {np.mean(num_positives):.1f}, Max {np.max(num_positives):.1f}\t"
            f"Negatives: Min {np.min(num_negatives):.1f}, Mean {np.mean(num_negatives):.1f}, Max {np.max(num_negatives):.1f}"
        )
        if documents:
            mean_base_mrr = np.mean(base_mrr_scores)
            mean_base_ndcg = np.mean(base_ndcg_scores)
            mean_base_ap = np.mean(base_ap_scores)
            base_metrics = {
                "base_map": mean_base_ap,
                f"base_mrr@{self.at_k}": mean_base_mrr,
                f"base_ndcg@{self.at_k}": mean_base_ndcg,
            }
            logger.info(f"{' ' * len(str(self.at_k))}       Base  -> Reranked")
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {mean_base_ap * 100:.2f} -> {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {mean_base_mrr * 100:.2f} -> {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {mean_base_ndcg * 100:.2f} -> {mean_ndcg * 100:.2f}")

            model_card_metrics = {
                "map": f"{mean_ap:.4f} ({mean_ap - mean_base_ap:+.4f})",
                f"mrr@{self.at_k}": f"{mean_mrr:.4f} ({mean_mrr - mean_base_mrr:+.4f})",
                f"ndcg@{self.at_k}": f"{mean_ndcg:.4f} ({mean_ndcg - mean_base_ndcg:+.4f})",
            }
            model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)
            self.store_metrics_in_model_card_data(model, model_card_metrics, epoch, steps)

            metrics.update(base_metrics)
            metrics = self.prefix_name_to_metrics(metrics, self.name)
        else:
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {mean_ndcg * 100:.2f}")

            metrics = self.prefix_name_to_metrics(metrics, self.name)
            self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_ndcg])

        return metrics


def train(train_dataset, dev_dataset, output_path):
    model = CrossEncoder(
        model_name_or_path="roberta-base")

    loss = BinaryCrossEntropyLoss(
        model
    )
    samples = []
    for item in dev_dataset:
        samples.append({
            "query": item["query"],
            "positive": item["positive"],
            "negative": [item[f"negative_{i + 1}"] for i in range(len(item) - 2)]
        })


    evaluator = CustomCrossEncoderRerankingEvaluator(
            samples,
        show_progress_bar=True,
    )

    def transform_data(data):
        final_examples = []
        for elem in tqdm(data):
            final_examples.append({
                "query": elem["query"],
                "text": elem["positive"],
                "label": 1.0,
            })
            for i in range(len(elem) - 2):
                final_examples.append({
                    "query": elem["query"],
                    "text": elem[f"negative_{i + 1}"],
                    "label": 0.0,
                })
        return final_examples

    train_dataset = transform_data(train_dataset)
    dev_dataset = transform_data(dev_dataset)

    total_steps = len(train_dataset) // 128 * 1

    train_dataset = Dataset.from_list(train_dataset)
    dev_dataset = Dataset.from_list(dev_dataset)



    args = CrossEncoderTrainingArguments(
        output_dir=f"models/{output_path}",
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=total_steps // 10,
        save_steps=total_steps // 10,
        load_best_model_at_end=True,
        eval_on_start=True,
        save_total_limit=2,
        logging_steps=100,
        logging_first_step=True,
        run_name="el",  # Will be used in W&B if `wandb` is installed
        disable_tqdm=False,
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    trainer.save_model(f"models/{output_path}/final")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("training_data_path", type=str)
    argument_parser.add_argument("development_data_path", type=str)
    argument_parser.add_argument("output_path", type=str)
    argument_parser.add_argument("--add_mentions", action="store_true",)
    argument_parser.add_argument("--candidate_retrieval_model", type=str, default="candidate_retriever/final")
    argument_parser.add_argument("--entity_index", type=str, default="entity_index.index")
    argument_parser.add_argument("--entity_mapping", type=str, default="entity_index.json")


    args = argument_parser.parse_args()

    tmp_train_file = args.training_data_path + ".pkl"
    tmp_dev_file = args.development_data_path + ".pkl"

    entity_index = args.entity_index
    entity_mapping = args.entity_mapping
    candidate_retrieval_model = args.candidate_retrieval_model


    kg_container = KGContainer()
    train_data = load_data(args.training_data_path, kg_container)
    dev_data = load_data(args.development_data_path, kg_container)

    context_candidates_list = prepare_context_candidates(train_data, entity_index, entity_mapping, candidate_retrieval_model)
    dev_context_candidates_list = prepare_context_candidates(dev_data, entity_index, entity_mapping, candidate_retrieval_model)

    train_dataset = create_sft_data(context_candidates_list, args.add_mentions)

    dev_dataset = create_sft_data(dev_context_candidates_list, args.add_mentions)

    train(train_dataset, dev_dataset, args.output_path)


