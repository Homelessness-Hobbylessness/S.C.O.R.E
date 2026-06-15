"""
Export trained CRNN to ONNX format for on-prem deployment via ONNX Runtime.
Dynamic axes are set for variable-length inputs (different image widths).

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pt \
                          --output model.onnx
"""

from __future__ import annotations
import argparse
import json

import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np

from src.model import CRNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default="model.onnx")
    p.add_argument("--img_height", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_args = ckpt.get("args", {})
    char2idx = ckpt["char2idx"]

    img_height = saved_args.get("img_height", args.img_height)

    model = CRNN(
        num_classes=len(char2idx) + 1,
        img_height=img_height,
        lstm_hidden=saved_args.get("lstm_hidden", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dummy input — batch of 1, height 32, width 128 (arbitrary)
    dummy = torch.randn(1, 1, img_height, 128)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch_size", 3: "width"},   # variable batch + width
            "logits": {0: "seq_len",    1: "batch_size"},
        },
    )
    print(f"Exported to {args.output}")

    # Validate ONNX model matches PyTorch output
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(args.output)
    ort_out = session.run(None, {"image": dummy.numpy()})[0]
    pt_out  = model(dummy).detach().numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    print(f"Max output difference PyTorch vs ONNX Runtime: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("✓ ONNX export validated — outputs match")
    else:
        print("✗ WARNING: ONNX output differs from PyTorch — check export")

    # Save vocab alongside the model
    vocab_path = args.output.replace(".onnx", "_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": ckpt["idx2char"]}, f)
    print(f"Vocabulary saved to {vocab_path}")


if __name__ == "__main__":
    main()
