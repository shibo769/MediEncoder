# ============================================================
# nn_utils.py
# Shared utilities for all models
# ============================================================

import os
import torch
import torch.nn as nn

_requested_device = os.environ.get("MEDIENC_DEVICE", "auto")
if _requested_device == "auto":
    _requested_device = "cuda" if torch.cuda.is_available() else "cpu"
if _requested_device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
device = torch.device(_requested_device)


def build_mlp(
    input_dim,
    output_dim,
    hidden_dims,
    activation="relu",
    dropout=0.0
):
    layers = []
    prev = input_dim

    if activation.lower() == "relu":
        act = nn.ReLU
    elif activation.lower() == "gelu":
        act = nn.GELU
    elif activation.lower() == "tanh":
        act = nn.Tanh
    else:
        raise ValueError("Unsupported activation")

    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
