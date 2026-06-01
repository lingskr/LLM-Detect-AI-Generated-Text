import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


def normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def char_noise(text: str, rate: float, seed: int, mode: str = "replace") -> str:
    if rate <= 0:
        return text
    rng = random.Random(seed)
    chars = list(text)
    n = len(chars)
    k = max(1, int(n * rate))
    idx = rng.sample(range(n), min(k, n)) if n > 0 else []

    if mode == "delete":
        mark = set(idx)
        return "".join(ch for i, ch in enumerate(chars) if i not in mark)

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i in idx:
        chars[i] = rng.choice(alphabet)
    return "".join(chars)


def truncate_text(text: str, keep_ratio: float) -> str:
    if keep_ratio >= 1.0:
        return text
    n = len(text)
    k = max(1, int(n * keep_ratio))
    return text[:k]


def whitespace_normalize(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def predict_scores(model, tokenizer, df: pd.DataFrame, positive_class_index: int, max_length: int, batch_size: int):
    ds = Dataset.from_pandas(df[["text"]])

    def preprocess(examples):
        return tokenizer(examples["text"], max_length=max_length, padding=True, truncation=True)

    ds_enc = ds.map(preprocess, batched=True)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    logits = trainer.predict(ds_enc).predictions
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return probs[:, positive_class_index]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Chapter 5 robustness evaluation for HF binary classifier")

    parser.add_argument("--eval-csv", default=str(root / "code/nonTargetText_llm_slightly_modified_gen.csv"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positive-class-index", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=str(root / "outputs/robustness_results.csv"))
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.eval_csv)
    if "label" not in df.columns:
        raise ValueError("eval csv must contain label column")
    df["text"] = normalize_text(df["text"])

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)

    y = df["label"].values
    rows = []

    # baseline
    base_scores = predict_scores(
        model=model,
        tokenizer=tokenizer,
        df=df,
        positive_class_index=args.positive_class_index,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    base_auc = roc_auc_score(y, base_scores)
    rows.append({"perturbation": "baseline", "strength": "-", "auc": float(base_auc), "drop": 0.0})

    # char noise replace
    for r in [0.01, 0.03, 0.05]:
        d = df.copy()
        d["text"] = [char_noise(t, r, seed=args.seed + i, mode="replace") for i, t in enumerate(d["text"].tolist())]
        s = predict_scores(model, tokenizer, d, args.positive_class_index, args.max_length, args.batch_size)
        a = roc_auc_score(y, s)
        rows.append({"perturbation": "char_replace", "strength": f"{int(r*100)}%", "auc": float(a), "drop": float(base_auc - a)})

    # truncation
    for keep in [0.5, 0.7]:
        d = df.copy()
        d["text"] = d["text"].map(lambda t: truncate_text(t, keep_ratio=keep))
        s = predict_scores(model, tokenizer, d, args.positive_class_index, args.max_length, args.batch_size)
        a = roc_auc_score(y, s)
        rows.append({"perturbation": "truncate", "strength": f"keep_{int(keep*100)}%", "auc": float(a), "drop": float(base_auc - a)})

    # format normalization
    d = df.copy()
    d["text"] = d["text"].map(whitespace_normalize)
    s = predict_scores(model, tokenizer, d, args.positive_class_index, args.max_length, args.batch_size)
    a = roc_auc_score(y, s)
    rows.append({"perturbation": "whitespace_norm", "strength": "binary", "auc": float(a), "drop": float(base_auc - a)})

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[done] robustness table -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
