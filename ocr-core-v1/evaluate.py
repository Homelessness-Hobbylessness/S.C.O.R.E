"""
Evaluation script — runs the trained model on the test split and reports
CER and WER using beam search decoding (not greedy).

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --data_dir data/raw --labels data/labels.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import CRNN
from src.dataset import HandwritingDataset, collate_fn
from src.utils import ctc_beam_search_decode, character_error_rate, word_error_rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_dir",    default="data/raw")
    p.add_argument("--labels",      default="data/labels.json")
    p.add_argument("--beam_width",  type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    char2idx: dict = ckpt["char2idx"]
    idx2char: dict = ckpt["idx2char"]
    saved_args: dict = ckpt.get("args", {})

    model = CRNN(
        num_classes=len(char2idx) + 1,
        img_height=saved_args.get("img_height", 32),
        lstm_hidden=saved_args.get("lstm_hidden", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    writer_splits = {
        "test": ["f0", "f1"],
    }
    test_ds = HandwritingDataset(
        args.data_dir, args.labels, char2idx,
        img_height=saved_args.get("img_height", 32),
        split="test", writer_splits=writer_splits,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets, input_lengths, target_lengths in test_loader:
            images = images.to(device)
            logits = model(images)
            # Beam search — more accurate than greedy, recovers punctuation
            preds = ctc_beam_search_decode(logits, idx2char, blank_idx=0, beam_width=args.beam_width)
            all_preds.extend(preds)

            offset = 0
            for length in target_lengths:
                indices = targets[offset:offset + length].tolist()
                all_targets.append("".join(idx2char.get(i, "") for i in indices))
                offset += length

    cer = character_error_rate(all_preds, all_targets)
    wer = word_error_rate(all_preds, all_targets)

    print(f"\n{'='*40}")
    print("S.C.O.R.E. OCR Evaluation Results")
    print(f"{'='*40}")
    print(f"Test samples:  {len(all_preds)}")
    print(f"Beam width:    {args.beam_width}")
    print(f"CER:           {cer:.4f}  (target: < 0.15)")
    print(f"WER:           {wer:.4f}")
    print(f"{'='*40}")

    if cer < 0.15:
        print("✓ CER target met — ready to proceed to real student handwriting data")
    else:
        print("✗ CER target not yet met — continue training or review architecture")

    # Show a few example predictions
    print("\nSample predictions (beam search):")
    for pred, target in zip(all_preds[:5], all_targets[:5]):
        print(f"  Target: {target}")
        print(f"  Pred:   {pred}")
        print()


if __name__ == "__main__":
    main()
