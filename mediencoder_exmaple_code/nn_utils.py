# ============================================================
# nn_utils.py
# Shared utilities for all models
# ============================================================

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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