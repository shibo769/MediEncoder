"""R's default row names are positions, not independently verified subject IDs."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mediencoder.real_data.io import load_rds


def test_r_default_string_row_names_require_alignment_declaration(monkeypatch):
    idx = [str(i) for i in range(1, 9)]
    tables = {
        "x": pd.DataFrame({"age": np.arange(8)}, index=idx),
        "m": pd.DataFrame({"marker": np.arange(8)}, index=idx),
        "y": pd.DataFrame({"outcome": np.arange(8)}, index=idx),
        "a": pd.DataFrame({"GDTOTAL": [3, 8] * 4}, index=idx),
    }
    monkeypatch.setitem(sys.modules, "pyreadr", SimpleNamespace(read_r=lambda name: {None: tables[name]}))
    kwargs = dict(covariates="x", mediators="m", outcome="y", treatment_table="a")
    with pytest.raises(ValueError, match="row identifiers are missing"):
        load_rds(**kwargs)
    data = load_rds(**kwargs, assume_row_aligned=True)
    np.testing.assert_array_equal(data["A"], [0, 1] * 4)
