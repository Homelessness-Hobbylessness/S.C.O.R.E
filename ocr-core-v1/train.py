"""
Training script for S.C.O.R.E. CRNN OCR model.
Target: CER < 0.15 on IAM validation set before moving to real student data.

Usage:
    python train.py --data_dir data/raw --labels data/labels.json \
                    --epochs 50 --batch_size 32 --lr 1e-3
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import CRNN
from src.dataset import HandwritingDataset, build_vocab, collate_fn
from src.utils import ctc_greedy_decode, ctc_beam_search_decode, character_error_rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/raw")
    p.add_argument("--labels",      default="data/labels.json")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--img_height",  type=int,   default=32)
    p.add_argument("--lstm_hidden", type=int,   default=256)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_file",    default="training_log.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Build vocabulary from all labels
    with open(args.labels) as f:
        all_labels = json.load(f)
    char2idx, idx2char = build_vocab(list(all_labels.values()))
    num_classes = len(char2idx) + 1  # +1 for blank at index 0
    print(f"Vocabulary size: {num_classes} (including blank)")

    # Writer-based splits — CRITICAL: prevents data leakage
    # IAM writer IDs are the first part of the filename (e.g. "a01", "a02")
    # Adjust these based on your actual IAM subset
    writer_splits = {
        "train": ["a0", "a1", "b0", "b1", "c0", "c1", "d0", "d1"],
        "val":   ["e0", "e1"],
        "test":  ["f0", "f1"],
    }

    # Augmentation for training only
    import torchvision.transforms as T
    train_aug = T.Compose([
        T.RandomRotation(degrees=3),
        T.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    train_ds = HandwritingDataset(
        args.data_dir, args.labels, char2idx,
        img_height=args.img_height, split="train",
        writer_splits=writer_splits, transform=train_aug,
    )
    val_ds = HandwritingDataset(
        args.data_dir, args.labels, char2idx,
        img_height=args.img_height, split="val",
        writer_splits=writer_splits, transform=None,
    )

    if len(train_ds) == 0:
        raise ValueError("train_ds is empty. Check writer_splits and available data.")
    if len(val_ds) == 0:
        raise ValueError("val_ds is empty. Check writer_splits and available data.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    model = CRNN(
        num_classes=num_classes,
        img_height=args.img_height,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5
    )

    training_log = []
    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, targets, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)                         # (T, N, C)
            log_probs = F.log_softmax(logits, dim=2)       # CTCLoss needs log-softmax

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        all_preds, all_targets_str = [], []
beam_search_decode(logits, idx2char, blank_idx=0, beam_width=1
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = ctc_greedy_decode(logits, idx2char, blank_idx=0)
                all_preds.extend(preds)

                # Reconstruct target strings from concatenated label tensor
                offset = 0
                for length in target_lengths:
                    indices = targets[offset:offset + length].tolist()
                    all_targets_str.append("".join(idx2char.get(i, "") for i in indices))
                    offset += length

        val_cer = character_error_rate(all_preds, all_targets_str)
        scheduler.step(val_cer)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val CER: {val_cer:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save checkpoint if best so far
        if val_cer < best_cer:
            best_cer = val_cer
            ckpt_path = Path(args.checkpoint_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_cer": val_cer,
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ New best CER {best_cer:.4f} — checkpoint saved")

        # Append to local JSON training log (no external services)
        training_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_cer": val_cer,
            "best_cer": best_cer,
            "lr": optimiser.param_groups[0]["lr"],
        })
        with open(args.log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best Val CER: {best_cer:.4f}")
    print(f"Checkpoint: {args.checkpoint_dir}/best_model.pt")
    print(f"Training log: {args.log_file}")


if __name__ == "__main__":
    main()
