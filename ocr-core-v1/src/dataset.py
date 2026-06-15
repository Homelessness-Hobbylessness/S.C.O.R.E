"""
Dataset loader for S.C.O.R.E. OCR training.
Targets the IAM Handwriting Database for Stage 1 proof-of-concept.
Split must be by writer ID (not randomly by image) to prevent data leakage.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def build_vocab(transcriptions: List[str]) -> Tuple[dict, dict]:
    """
    Build character vocabulary from a list of transcription strings.
    Index 0 is reserved for the CTC blank token.

    Returns:
        char2idx: character → index
        idx2char: index → character
    """
    chars = sorted(set("".join(transcriptions)))
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = blank
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    idx2char[0] = "<blank>"
    return char2idx, idx2char


class HandwritingDataset(Dataset):
    """
    Loads handwriting images and their transcriptions from a labels.json file.

    labels.json format:
    {
        "a01-000u-00.png": "He rose from",
        "a01-000u-01.png": "a stooped",
        ...
    }

    Args:
        image_dir:    directory containing the image files
        labels_path:  path to labels.json
        char2idx:     character to index mapping (from build_vocab)
        img_height:   fixed height to resize images to (width stays proportional)
        split:        'train', 'val', or 'test'
        writer_splits: dict mapping split name → list of writer ID prefixes.
                       IMPORTANT: splits must be by writer, not by image,
                       to prevent data leakage between train and eval sets.
        transform:    optional additional transforms (augmentation for train only)
    """

    def __init__(
        self,
        image_dir: str | Path,
        labels_path: str | Path,
        char2idx: Dict[str, int],
        img_height: int = 32,
        split: str = "train",
        writer_splits: Optional[Dict[str, List[str]]] = None,
        transform: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.char2idx = char2idx
        self.img_height = img_height
        self.transform = transform

        with open(labels_path) as f:
            all_labels: Dict[str, str] = json.load(f)

        # Filter by writer ID if splits provided
        if writer_splits:
            if split not in writer_splits:
                raise ValueError(f"Split '{split}' not found in explicitly provided writer_splits configuration.")
            allowed_prefixes = tuple(writer_splits[split])
            self.samples = [
                (fname, text)
                for fname, text in all_labels.items()
                if fname.startswith(allowed_prefixes)
            ]
        else:
            self.samples = list(all_labels.items())

        # Base transform: grayscale, resize height, normalise
        self.base_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height,)),   # keeps aspect ratio, fixes height
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        fname, text = self.samples[idx]
        img_path = self.image_dir / fname

        img = Image.open(img_path).convert("RGB")
        img = self.base_transform(img)

        if self.transform:
            img = self.transform(img)

        img_tensor = T.ToTensor()(img)  # (1, H, W), values in [0, 1]
        img_tensor = T.Normalize(mean=[0.5], std=[0.5])(img_tensor)  # → [-1, 1]

        label = torch.tensor(
            [self.char2idx[c] for c in text if c in self.char2idx],
            dtype=torch.long,
        )
        return img_tensor, label, len(text)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads images to the same width and stacks labels for CTCLoss.
    CTCLoss requires:
        inputs:         (T, N, C)
        input_lengths:  (N,)  — sequence length after CNN, not image width
        targets:        1D concatenated labels
        target_lengths: (N,)
    """
    images, labels, _ = zip(*batch)

    max_w = max(img.shape[2] for img in images)
    padded = torch.stack([
        torch.nn.functional.pad(img, (0, max_w - img.shape[2])) for img in images
    ])  # (N, 1, H, W)

    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    # CNN reduces width by factor of 4 (two MaxPool2d with stride 2 on W axis,
    # minus 1 from the final Conv2d kernel=2). Compute actual sequence lengths.
    # Adjust this formula if you change the CNN architecture.
    input_lengths = torch.tensor(
        [max(1, (img.shape[2] // 4) - 1) for img in images], dtype=torch.long
    )

    return padded, targets, input_lengths, target_lengths
