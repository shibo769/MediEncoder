"""Freeze actual observed bytes once, even when numerical generation varies."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mediencoder.simulation import dgp, runner


@pytest.fixture
def cache_setup(monkeypatch):
    mechanism = SimpleNamespace(mechanism_hash="fixed-mechanism")
    config = dict(dgp=dict(p=3, q=2))
    task = dict(n=8, rep=0, data_seed=1234)
    calls = []
    guard = threading.Lock()
    def sampler(mechanism, n, seed):
        with guard:
            calls.append(seed)
            value = len(calls)
        time.sleep(.02)
        return dict(X=np.full((n, 3), float(value)), M=np.full((n, 2), float(value)),
                    A=np.arange(n) % 2, Y=np.arange(n, dtype=float),
                    oracle={"latent": object()}, theta_population=17.)
    monkeypatch.setattr(dgp, "sample_data", sampler)
    return mechanism, config, task, calls


def test_parallel_producer_called_once_and_resume_ignores_generator_perturbation(cache_setup, tmp_path, monkeypatch):
    mechanism, config, task, calls = cache_setup
    def read(_):
        return runner.cached_observed_data(mechanism, task, config, tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read, range(8)))
    assert calls == [task["data_seed"]]
    assert {runner.observed_data_hash(data) for data in results} == {runner.observed_data_hash(results[0])}
    assert all(set(data) == {"X", "M", "A", "Y"} for data in results)
    monkeypatch.setattr(dgp, "sample_data", lambda *args: pytest.fail("Dataset was regenerated"))
    resumed = read(None)
    assert all(np.array_equal(resumed[key], results[0][key]) for key in resumed)
    with np.load(next((tmp_path / "datasets").glob("*.npz")), allow_pickle=False) as saved:
        assert set(saved.files) == {"X", "M", "A", "Y"}


def test_external_cache_is_read_only_and_backend_training_independent(cache_setup, tmp_path, monkeypatch):
    mechanism, config, task, calls = cache_setup
    first = runner.cached_observed_data(mechanism, task, config, tmp_path / "source")
    shared = tmp_path / "source" / "datasets"
    before = {path.name: path.read_bytes() for path in shared.iterdir()}
    other = dict(config, device="cuda", training={"epochs": 999})
    monkeypatch.setattr(dgp, "sample_data", lambda *args: pytest.fail("External data cache was regenerated"))
    second = runner.cached_observed_data(mechanism, task, other, tmp_path / "gpu", shared)
    assert runner.observed_data_hash(first) == runner.observed_data_hash(second)
    assert before == {path.name: path.read_bytes() for path in shared.iterdir()}
    assert not (tmp_path / "gpu" / "datasets").exists()
    assert list((tmp_path / "gpu" / "dataset_bindings").glob("*.json"))
    with pytest.raises(ValueError, match="missing or incomplete"):
        runner.cached_observed_data(mechanism, dict(task, rep=1), config, tmp_path, shared)


@pytest.mark.parametrize("corruption", ["npz", "metadata", "shape", "nonfinite", "oracle", "seed", "mechanism"])
def test_corruption_or_identity_mismatch_fails_closed(cache_setup, tmp_path, corruption):
    mechanism, config, task, calls = cache_setup
    data = runner.cached_observed_data(mechanism, task, config, tmp_path)
    identity = runner.observed_dataset_identity(mechanism, task, config)
    npz, receipt, intent = runner.observed_dataset_paths(tmp_path / "datasets", identity)
    if corruption == "npz":
        npz.write_bytes(b"damaged")
    elif corruption == "metadata":
        metadata = json.loads(receipt.read_text())
        metadata["identity"]["data_seed"] += 1
        runner.atomic_json(receipt, metadata)
    elif corruption in ("shape", "nonfinite", "oracle"):
        if corruption == "shape":
            data["X"] = data["X"][:-1]
        elif corruption == "nonfinite":
            data["Y"][0] = np.nan
        else:
            data["oracle"] = np.ones(8)
        np.savez(npz, **data)
        metadata = json.loads(receipt.read_text())
        metadata["npz_sha256"] = runner.file_hash(npz)
        runner.atomic_json(receipt, metadata)
    elif corruption == "seed":
        task = dict(task, data_seed=999)
    else:
        mechanism = SimpleNamespace(mechanism_hash="different-mechanism")
    before = {path.name: path.read_bytes() for path in (tmp_path / "datasets").iterdir()}
    with pytest.raises((ValueError, KeyError)):
        runner.cached_observed_data(mechanism, task, config, tmp_path, tmp_path / "datasets")
    assert before == {path.name: path.read_bytes() for path in (tmp_path / "datasets").iterdir()}
    assert len(calls) == 1


def test_interrupted_atomic_npz_is_adopted_without_redraw_only_before_returned_fit(cache_setup, tmp_path, monkeypatch):
    mechanism, config, task, calls = cache_setup
    data = runner.cached_observed_data(mechanism, task, config, tmp_path)
    identity = runner.observed_dataset_identity(mechanism, task, config)
    npz, receipt, intent = runner.observed_dataset_paths(tmp_path / "datasets", identity)
    before = npz.read_bytes()
    receipt.unlink()
    runner.atomic_json(intent, identity)
    monkeypatch.setattr(dgp, "sample_data", lambda *args: pytest.fail("Orphan NPZ was overwritten"))
    restored = runner.cached_observed_data(mechanism, task, config, tmp_path)
    assert npz.read_bytes() == before
    assert runner.observed_data_hash(restored) == runner.observed_data_hash(data)
    receipt.unlink()
    runner.atomic_json(intent, identity)
    runner.atomic_json(tmp_path / "tasks" / "n00008_r0000_projection.json", dict(task, status="complete"))
    with pytest.raises(ValueError, match="after a fit"):
        runner.cached_observed_data(mechanism, task, config, tmp_path)
    assert not receipt.exists() and npz.read_bytes() == before


def test_unidentified_orphan_is_not_overwritten(cache_setup, tmp_path):
    mechanism, config, task, calls = cache_setup
    identity = runner.observed_dataset_identity(mechanism, task, config)
    npz, receipt, intent = runner.observed_dataset_paths(tmp_path / "datasets", identity)
    npz.parent.mkdir()
    npz.write_bytes(b"unknown bytes")
    with pytest.raises(ValueError, match="Unidentified orphan"):
        runner.cached_observed_data(mechanism, task, config, tmp_path)
    assert npz.read_bytes() == b"unknown bytes" and not calls


@pytest.mark.parametrize("keep_binding", [False, True])
def test_partial_resume_cannot_switch_to_another_internally_valid_cache(cache_setup, tmp_path, keep_binding):
    mechanism, config, task, calls = cache_setup
    first = runner.cached_observed_data(mechanism, task, config, tmp_path / "source_a")
    second = runner.cached_observed_data(mechanism, task, config, tmp_path / "source_b")
    assert runner.observed_data_hash(first) != runner.observed_data_hash(second)
    output = tmp_path / "partial"
    runner.cached_observed_data(mechanism, task, config, output, tmp_path / "source_a" / "datasets")
    runner.atomic_json(output / "tasks" / "n00008_r0000_projection.json",
                       dict(task, status="complete", observed_data_sha256=runner.observed_data_hash(first)))
    if not keep_binding:
        for path in (output / "dataset_bindings").glob("*.json"):
            path.unlink()
    with pytest.raises(ValueError, match="binding|previously returned"):
        runner.cached_observed_data(mechanism, task, config, output, tmp_path / "source_b" / "datasets")
    reused = runner.cached_observed_data(mechanism, task, config, output, tmp_path / "source_a" / "datasets")
    assert runner.observed_data_hash(reused) == runner.observed_data_hash(first)


def test_external_cache_cli_is_execution_only():
    ordinary = runner.parse_args(["--output-dir", "unused"])
    imported = runner.parse_args(["--output-dir", "unused", "--observed-data-from", "shared/datasets"])
    assert imported.observed_data_from == Path("shared/datasets")
    assert runner.make_config(ordinary) == runner.make_config(imported)
