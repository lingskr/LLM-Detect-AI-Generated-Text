import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
try:
    import seaborn as sns
except ImportError:
    sns = None


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize train embeddings with t-SNE")
    parser.add_argument("--input-csv", default=str(root / "code/train1.csv"))
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None, help="cuda/cpu, default auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--output", default=str(root / "outputs/embedding_tsne.png"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("input csv must contain text,label columns")

    llm_df = df[df["label"] == 1].head(args.max_per_class).copy()
    human_df = df[df["label"] == 0].head(args.max_per_class).copy()

    model = SentenceTransformer(args.model_name, device=args.device)
    emb_llm = model.encode(
        llm_df["text"].fillna("").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb_human = model.encode(
        human_df["text"].fillna("").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    show_df = pd.concat([llm_df, human_df], ignore_index=True)
    show_df["src"] = ["LLM"] * len(llm_df) + ["student"] * len(human_df)
    show_emb = np.concatenate([emb_llm, emb_human], axis=0)

    tsne = TSNE(random_state=args.seed)
    x_2d = tsne.fit_transform(show_emb)

    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.scatterplot(x=x_2d[:, 0], y=x_2d[:, 1], hue=show_df["src"], legend="full", alpha=0.8, s=22)
    else:
        llm_mask = show_df["src"] == "LLM"
        human_mask = ~llm_mask
        plt.scatter(x_2d[llm_mask, 0], x_2d[llm_mask, 1], s=22, alpha=0.8, label="LLM")
        plt.scatter(x_2d[human_mask, 0], x_2d[human_mask, 1], s=22, alpha=0.8, label="student")
        plt.legend()
    plt.title("t-SNE of Essay Embeddings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"[done] figure saved to {out_path}")


if __name__ == "__main__":
    main()
