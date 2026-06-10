#!/usr/bin/env python3
"""Contrastive fine-tuning of sentence-transformer embeddings for category separation.

Usage:
    python scripts/contrastive_finetune.py \
        --input data/corpus.parquet \
        --output data/corpus_finetuned.parquet \
        --model all-MiniLM-L6-v2 \
        --epochs 5 \
        --batch-size 64 \
        --temperature 0.07 \
        --target-dim 128

Reads a Parquet corpus with columns (label, category, embedding), fine-tunes
the embedding model so same-category concepts cluster tighter, and writes a
new Parquet with updated embeddings. The original label/category/metadata
columns are preserved unchanged.

The contrastive loss is supervised NT-Xent (normalized temperature-scaled
cross-entropy): for each anchor, positives are same-category concepts in the
batch, negatives are all other concepts. This is the same loss used by
SimCLR/SupCon but applied to text embeddings with category labels as the
supervision signal.

Requirements:
    pip install sentence-transformers torch pyarrow pandas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ── Dataset ─────────────────────────────────────────────────────────────

class ConceptDataset(Dataset):
    """Yields (label_text, category_index) pairs for the sentence transformer."""

    def __init__(self, labels: list[str], categories: list[str]):
        self.labels = labels
        unique_cats = sorted(set(categories))
        self.cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        self.cat_indices = [self.cat_to_idx[c] for c in categories]
        self.n_categories = len(unique_cats)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.labels[idx], self.cat_indices[idx]


# ── Supervised NT-Xent loss ─────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    For each anchor i in the batch, positives are all other items with the
    same category label. The loss pulls positives together and pushes
    negatives apart in the normalized embedding space.

    Temperature controls the sharpness of the similarity distribution:
    lower = harder negatives, tighter clusters. 0.07 is the SupCon default.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, dim) L2-normalized embeddings
            labels: (batch_size,) integer category labels
        Returns:
            scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float().to(device)
        mask.fill_diagonal_(0)

        positives_per_anchor = mask.sum(dim=1)

        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        self_mask = torch.eye(batch_size, device=device)
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        log_prob = logits - log_sum_exp

        mean_log_prob = (mask * log_prob).sum(dim=1) / (positives_per_anchor + 1e-12)

        valid = positives_per_anchor > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob[valid].mean()
        return loss


# ── Projection head ─────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Optional MLP projection head (SimCLR-style).

    Projects the sentence-transformer output to `target_dim` dimensions
    with L2 normalization. The contrastive loss operates in this projected
    space; at inference time, we use the projected embeddings directly
    (not the raw transformer output) because the projection is trained
    to optimize category separation.
    """

    def __init__(self, input_dim: int, target_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        return F.normalize(projected, dim=-1)


# ── Training loop ───────────────────────────────────────────────────────

def train(
    model,
    projection: ProjectionHead,
    dataloader: DataLoader,
    criterion: SupConLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> list[float]:
    """Fine-tune the model + projection head with SupCon loss."""
    model.train()
    projection.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for texts, cat_labels in dataloader:
            cat_labels = cat_labels.to(device)

            features = model.tokenize(texts)
            features = {k: v.to(device) for k, v in features.items()}
            model_output = model(features)
            raw_embeddings = model_output["sentence_embedding"]

            projected = projection(raw_embeddings)

            loss = criterion(projected, cat_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        print(f"  Epoch {epoch + 1}/{epochs}: loss = {mean_loss:.4f}")

    return epoch_losses


# ── Inference ───────────────────────────────────────────────────────────

@torch.no_grad()
def embed_all(
    model,
    projection: ProjectionHead,
    labels: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Embed all concept labels through the fine-tuned model + projection."""
    model.eval()
    projection.eval()
    all_embeddings = []

    for start in range(0, len(labels), batch_size):
        batch_texts = labels[start : start + batch_size]
        features = model.tokenize(batch_texts)
        features = {k: v.to(device) for k, v in features.items()}
        model_output = model(features)
        raw = model_output["sentence_embedding"]
        projected = projection(raw)
        all_embeddings.append(projected.cpu().numpy())

    return np.vstack(all_embeddings)


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_separation_metrics(
    embeddings: np.ndarray,
    categories: list[str],
) -> dict:
    """Compute intra/inter category similarity and separation ratio."""
    from collections import defaultdict

    cat_indices = defaultdict(list)
    for i, c in enumerate(categories):
        cat_indices[c].append(i)

    intra_sims = []
    for cat, indices in cat_indices.items():
        if len(indices) < 2:
            continue
        cat_embs = embeddings[indices]
        sim_matrix = cat_embs @ cat_embs.T
        n = len(indices)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        intra_sims.extend(sim_matrix[mask].tolist())

    rng = np.random.RandomState(42)
    inter_sims = []
    cat_names = list(cat_indices.keys())
    n_samples = min(50000, len(embeddings) * 10)
    for _ in range(n_samples):
        c1, c2 = rng.choice(len(cat_names), 2, replace=False)
        i = rng.choice(cat_indices[cat_names[c1]])
        j = rng.choice(cat_indices[cat_names[c2]])
        inter_sims.append(float(embeddings[i] @ embeddings[j]))

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    separation = mean_intra / max(mean_inter, 1e-8)

    return {
        "intra_category_similarity": mean_intra,
        "inter_category_similarity": mean_inter,
        "category_separation_ratio": separation,
        "n_intra_pairs": len(intra_sims),
        "n_inter_pairs": len(inter_sims),
    }


# ── Collate function ────────────────────────────────────────────────────

def collate_fn(batch: list[tuple[str, int]]) -> tuple[list[str], torch.Tensor]:
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for category separation")
    parser.add_argument("--input", required=True, help="Input Parquet corpus")
    parser.add_argument("--output", required=True, help="Output Parquet with fine-tuned embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformer model name")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="SupCon temperature")
    parser.add_argument("--target-dim", type=int, default=128, help="Output embedding dimension")
    parser.add_argument("--min-category-size", type=int, default=5,
                        help="Skip categories with fewer items during training (still embedded)")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading corpus from {args.input}")
    df = pd.read_parquet(args.input)
    assert "label" in df.columns and "category" in df.columns, \
        f"Parquet must have 'label' and 'category' columns, got {list(df.columns)}"

    labels = df["label"].tolist()
    categories = df["category"].tolist()
    print(f"  {len(labels)} concepts across {len(set(categories))} categories")

    cat_counts = pd.Series(categories).value_counts()
    train_mask = [cat_counts[c] >= args.min_category_size for c in categories]
    train_labels = [l for l, m in zip(labels, train_mask) if m]
    train_categories = [c for c, m in zip(categories, train_mask) if m]
    n_skipped = len(labels) - len(train_labels)
    print(f"  Training on {len(train_labels)} concepts ({n_skipped} skipped from small categories)")

    print(f"Loading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(args.model, device=str(device))
    raw_dim = st_model.get_sentence_embedding_dimension()
    print(f"  Raw embedding dimension: {raw_dim}")

    projection = ProjectionHead(raw_dim, args.target_dim).to(device)

    dataset = ConceptDataset(train_labels, train_categories)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW([
        {"params": st_model.parameters(), "lr": args.lr},
        {"params": projection.parameters(), "lr": args.lr * 10},
    ], weight_decay=1e-4)

    criterion = SupConLoss(temperature=args.temperature)

    print("\nBaseline metrics (before fine-tuning):")
    with torch.no_grad():
        baseline_embs = embed_all(st_model, projection, labels, args.batch_size, device)
    baseline_metrics = compute_separation_metrics(baseline_embs, categories)
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nFine-tuning for {args.epochs} epochs...")
    losses = train(st_model, projection, dataloader, criterion, optimizer, device, args.epochs)

    print("\nPost-training metrics:")
    final_embs = embed_all(st_model, projection, labels, args.batch_size, device)
    final_metrics = compute_separation_metrics(final_embs, categories)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nImprovement:")
    for k in ["intra_category_similarity", "inter_category_similarity", "category_separation_ratio"]:
        before = baseline_metrics[k]
        after = final_metrics[k]
        delta = after - before
        pct = (delta / max(abs(before), 1e-8)) * 100
        print(f"  {k}: {before:.4f} → {after:.4f} ({delta:+.4f}, {pct:+.1f}%)")

    print(f"\nWriting fine-tuned corpus to {args.output}")
    df["embedding"] = [row.tolist() for row in final_embs]
    df.to_parquet(args.output, index=False)
    print(f"  Done. {len(df)} rows written.")

    head_path = Path(args.output).with_suffix(".projection_head.pt")
    torch.save({
        "projection_state_dict": projection.state_dict(),
        "raw_dim": raw_dim,
        "target_dim": args.target_dim,
        "model_name": args.model,
        "training_losses": losses,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
    }, head_path)
    print(f"  Projection head saved to {head_path}")


if __name__ == "__main__":
    main()
