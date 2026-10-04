"""Local-only observed data loading. This module performs no learned scaling."""
from pathlib import Path

import numpy as np


def validate_observed_data(X, M, A, Y):
    arrays = {"X": np.asarray(X), "M": np.asarray(M),
              "A": np.asarray(A), "Y": np.asarray(Y)}
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{key} must contain numeric values")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains missing or nonfinite values; no rows are silently removed")
    if arrays["X"].ndim != 2 or arrays["M"].ndim != 2:
        raise ValueError("X and M must be subject-by-feature matrices")
    if arrays["A"].ndim != 1 or arrays["Y"].ndim != 1:
        raise ValueError("A and Y must be one-dimensional subject vectors")
    n = len(arrays["Y"])
    if n < 4 or any(len(v) != n for v in arrays.values()):
        raise ValueError("All observed arrays must have the same subject count, at least four")
    if not arrays["X"].shape[1] or not arrays["M"].shape[1]:
        raise ValueError("X and M must each have at least one feature")
    if set(np.unique(arrays["A"])) != {0, 1}:
        raise ValueError("A must contain both treatment groups encoded as 0 and 1")
    return arrays


def load_npz(path):
    """Read X, M, A, Y in a common subject order; never allow pickle payloads."""
    with np.load(Path(path), allow_pickle=False) as archive:
        expected = {"X", "M", "A", "Y"}
        if set(archive.files) != expected:
            raise ValueError("Observed-data NPZ must contain exactly X, M, A, Y")
        return validate_observed_data(**{key: archive[key].copy() for key in expected})


def load_rds(*, covariates, mediators, outcome, treatment_table,
             treatment_column="GDTOTAL", threshold=6., assume_row_aligned=False):
    """Load four explicitly supplied RDS objects without copying private files.

    Non-default row identifiers must agree. When row identifiers are absent, an
    explicit row-alignment declaration is required. No filtering or imputation
    is inferred from the files.
    """
    try:
        import pyreadr
        import pandas as pd
    except ImportError as exc:
        raise ImportError("RDS input needs the optional 'pyreadr' dependency") from exc

    def one(path):
        contents = list(pyreadr.read_r(str(path)).values())
        if len(contents) != 1:
            raise ValueError("Each RDS path must contain one tabular object")
        return contents[0]

    tables = [one(p) for p in (covariates, mediators, outcome, treatment_table)]
    X_table, M_table, Y_table, exposure = tables
    if not isinstance(exposure, pd.DataFrame) or treatment_column not in exposure:
        raise ValueError(f"Treatment table lacks column {treatment_column!r}")
    if not np.isfinite(threshold):
        raise ValueError("Treatment threshold must be finite")
    raw_treatment = np.asarray(exposure[treatment_column], dtype=float)
    if not np.all(np.isfinite(raw_treatment)):
        raise ValueError("Treatment source contains missing/nonfinite values")
    # A treatment-defining variable cannot also be a pretreatment confounder.
    if hasattr(X_table, "columns") and treatment_column in X_table.columns:
        raise ValueError("Remove the treatment-defining column and its derived features from X before analysis")

    identifiers = []
    for table in tables:
        index = getattr(table, "index", None)
        # R's default row names commonly arrive as strings "1", ..., "n",
        # rather than a pandas RangeIndex. They do not establish subject identity.
        sequential = False
        if index is not None:
            strings = np.asarray(index).astype(str)
            sequential = any(np.array_equal(strings, np.arange(start, start + len(index)).astype(str))
                             for start in (0, 1))
        if index is None or isinstance(index, pd.RangeIndex) or sequential:
            identifiers.append(None)
        else:
            if not index.is_unique:
                raise ValueError("RDS subject identifiers must be unique")
            identifiers.append(np.asarray(index))
    if any(index is None for index in identifiers):
        if not assume_row_aligned:
            raise ValueError("RDS row identifiers are missing; explicitly confirm common row order")
    available = [index for index in identifiers if index is not None]
    if available and any(not np.array_equal(index, available[0]) for index in available[1:]):
        raise ValueError("RDS subject identifiers/order do not agree")
    Y = np.asarray(Y_table, dtype=float)
    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = Y[:, 0]
    return validate_observed_data(np.asarray(X_table, dtype=float),
                                  np.asarray(M_table, dtype=float),
                                  (raw_treatment >= threshold).astype(float), Y)
