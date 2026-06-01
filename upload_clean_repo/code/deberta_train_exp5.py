import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def softmax_np(logits):
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3-small for LLM detection")

    parser.add_argument("--model-checkpoint", default="microsoft/deberta-v3-small")
    parser.add_argument("--pile-dir", default=None, help="dir containing pile2.parquet, plies3.parquet, plies4.parquet, Ultra.parquet, lmsys.parquet")
    parser.add_argument("--human-llm-parquet", default=None, help="parquet with columns text/source where source in [Human, ...]")
    parser.add_argument("--valid-csv", default=str(root / "code/nonTargetText_llm_slightly_modified_gen.csv"))
    parser.add_argument("--output-dir", default=str(root / "models/deberta/deberta-v3-small-finetuned_v5"))
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_train_from_parquets(args):
    dfs = []

    def _read_parquet_with_fallback(path):
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            return pd.read_parquet(path, engine="pyarrow")

    if args.pile_dir:
        pile_dir = Path(args.pile_dir)
        for name in ["pile2.parquet", "plies3.parquet", "plies4.parquet", "Ultra.parquet", "lmsys.parquet"]:
            p = pile_dir / name
            if p.exists():
                dfs.append(_read_parquet_with_fallback(p))

    if args.human_llm_parquet and Path(args.human_llm_parquet).exists():
        human_llm = _read_parquet_with_fallback(args.human_llm_parquet)
        if {"text", "source"}.issubset(human_llm.columns):
            human_llm["label"] = np.where(human_llm["source"] == "Human", 0, 1)
            dfs.append(human_llm[["text", "label"]])

    if not dfs:
        raise FileNotFoundError(
            "No training parquet loaded. Provide --pile-dir and/or --human-llm-parquet."
        )

    train = pd.concat(dfs, axis=0, ignore_index=True)
    if not {"text", "label"}.issubset(train.columns):
        raise ValueError("Training data must contain text,label columns")

    train["text"] = train["text"].fillna("").astype(str).str.strip("\n")
    train = train.dropna(subset=["text", "label"]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    return train


def main():
    args = parse_args()

    train = load_train_from_parquets(args)
    valid = pd.read_csv(args.valid_csv)
    valid["text"] = valid["text"].fillna("").astype(str).str.strip("\n")

    ds_train = Dataset.from_pandas(train[["text", "label"]])
    ds_valid = Dataset.from_pandas(valid[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=False)

    def preprocess(examples):
        return tokenizer(examples["text"], max_length=args.max_length, padding=True, truncation=True)

    ds_train_enc = ds_train.map(preprocess, batched=True)
    ds_valid_enc = ds_valid.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax_np(logits)
        auc = roc_auc_score(labels, probs[:, 1], multi_class="ovr")
        return {"roc_auc": auc}

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        report_to="none",
        save_total_limit=3,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train_enc,
        eval_dataset=ds_valid_enc,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
