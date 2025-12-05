import argparse
from collections import defaultdict
from typing import Iterable, Any

import torch
from datasets import DatasetDict, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class CustomCollator(SentenceTransformerDataCollator):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break
        if "all_positive_relations" in column_names:
            positive_relations = pad_sequence([torch.tensor(row["all_positive_relations"]) for row in features], padding_value=-1, batch_first=True)
            negative_relation = torch.tensor([row["negative_relation"] for row in features])
            positive_relation = torch.tensor([row["positive_relation"] for row in features])
            labels = torch.cat((positive_relation, negative_relation), dim=0)

            neutralize_mask = []
            for i in range(positive_relations.size(0)):
                found = torch.isin(labels, positive_relations[i] )
                neutralize_mask.append(found)
            neutralize_mask = torch.stack(neutralize_mask, dim=0)

            column_names.remove("all_positive_relations")
            column_names.remove("negative_relation")
            column_names.remove("positive_relation")

            batch["label"] = neutralize_mask

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            tokenized = self.tokenize_fn([row[column_name] for row in features])
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch
class CustomMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives and negatives),
        # also from other anchors. This gives us a lot of in-batch negatives.
        scores = self.similarity_fct(anchors, candidates) * self.scale
        # (batch_size, batch_size * (1 + num_negatives))

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(0, scores.size(0), device=scores.device)

        if labels is not None:
            labels[torch.arange(labels.size(0)), torch.arange(labels.size(0))] = False
            scores[labels] = -1e6
        return self.cross_entropy_loss(scores, range_labels)




def train(dataset, model_name: str, batch_size: int, epochs: int, output_dir: str, lr: float = 2e-5):
    # 1. Load a model to finetune with 2. (Optional) model card data
    torch.set_float32_matmul_precision("high")
    model = SentenceTransformer(
        model_name,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["valid"]

    # 4. Define a loss function
    loss = CustomMultipleNegativesRankingLoss(model)

    total_steps = len(train_dataset) // batch_size

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=total_steps // 10,
        save_strategy="steps",
        save_steps=total_steps // 10,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=100,
        run_name=output_dir,  # Will be used in W&B if `wandb` is installed
        dataloader_num_workers=10,
    )

    query_to_id = {}
    document_to_id = {}
    query_id_to_document = defaultdict(set)
    for sentence, positive, negative in zip(eval_dataset["sentence"], eval_dataset["positive"], eval_dataset["negative"]):
        if sentence not in query_to_id:
            query_id = str(len(query_to_id))
            query_to_id[sentence] = query_id
        if positive not in document_to_id:
            document_to_id[positive] = str(len(document_to_id))
        if negative not in document_to_id:
            document_to_id[negative] = str(len(document_to_id))
        query_id_to_document[query_to_id[sentence]].add(document_to_id[positive])
    for sentence, positive, negative in zip(train_dataset["sentence"], train_dataset["positive"], train_dataset["negative"]):
        if positive not in document_to_id:
            document_to_id[positive] = str(len(document_to_id))
        if negative not in document_to_id:
            document_to_id[negative] = str(len(document_to_id))
    id_to_query = {v: k for k, v in query_to_id.items()}
    id_to_document = {v: k for k, v in document_to_id.items()}

    # 6. (Optional) Create an evaluator & evaluate the base model
    # dev_evaluator = TripletEvaluator(
    #     anchors=eval_dataset["sentence"],
    #     positives=eval_dataset["positive"],
    #     negatives=eval_dataset["negative"],
    #     name="all-nli-dev",
    # )
    dev_evaluator = InformationRetrievalEvaluator(
            queries=id_to_query,
            corpus=id_to_document,
            relevant_docs=query_id_to_document,
            name="all-nli-dev",
        )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        data_collator=CustomCollator(
            tokenize_fn=model.tokenize,
            valid_label_columns=["label", "score"],
        ),
    )
    trainer.train()

    # 8. Save the trained model
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sentence Transformers model.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("output_dir", type=str, help="Where to store the final model.")

    parser.add_argument("--model_name",
                        type=str,
        default="all-mpnet-base-v2",
        help="The name of the pre-trained model to use.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs to perform.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )



    args = parser.parse_args()


    dataset = DatasetDict.load_from_disk(args.dataset_dir)


    # Train the model
    train(
        dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        lr=args.lr,
    )
