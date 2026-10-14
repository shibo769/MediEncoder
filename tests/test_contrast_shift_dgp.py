"""Explicit additive mediator shift after a centered contrast; no training."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mediencoder.simulation import dgp


@pytest.fixture(scope="module")
def centered():
    config = dgp.DGPConfig(p=7, q=6, bar_p=2, bar_q=2, pilot_size=128,
                           delta1_spread=.6, delta1_contrast=.05)
    return dgp.draw_parameters(config, 1234)


def test_nonzero_shift_is_applied_after_contrast_without_other_structural_redraws(centered):
    shifted = dgp.draw_parameters(replace(centered.config, delta1_shift=.1), centered.parameter_seed)
    np.testing.assert_allclose(shifted.arrays["delta_1"], centered.arrays["delta_1"] + .1,
                               rtol=0, atol=1e-15)
    for name, expected in centered.arrays.items():
        if name != "delta_1":
            np.testing.assert_array_equal(shifted.arrays[name], expected)
    assert shifted.truth == centered.truth  # theta10 depends on delta0, never delta1.
    assert "contrast_shift_rule" not in centered.metadata["delta_distribution"]
    assert shifted.metadata["delta_distribution"]["contrast_shift_rule"] == "add-after-centered-contrast"


def test_zero_shift_preserves_the_original_centered_contrast_equation(centered):
    c = centered.config
    structural_seed = dgp._stream_seeds(centered.parameter_seed, 0x50415241, 3)[0]
    rng = np.random.default_rng(structural_seed)
    rng.uniform(0, c.alpha_upper, size=c.bar_p)
    delta0 = 1 + c.delta1_spread * (rng.uniform(.5, 1.5, size=(c.bar_p, c.bar_p)) - 1)
    rng.uniform(.5, 1.5, size=(c.bar_p, c.bar_p))
    contrast = rng.normal(size=(c.bar_p, c.bar_p))
    contrast -= contrast.mean(axis=1, keepdims=True)
    np.testing.assert_array_equal(centered.arrays["delta_0"], delta0)
    np.testing.assert_array_equal(centered.arrays["delta_1"], delta0 + c.delta1_contrast * contrast)


def test_legacy_saved_nonzero_shift_artifact_remains_authoritative(centered, tmp_path):
    # Old positive-contrast code ignored the config shift. Loading must retain
    # those saved arrays/metadata/hash, without redrawing under the new rule.
    legacy = replace(centered, config=replace(centered.config, delta1_shift=.1),
                     metadata={**centered.metadata, "delta_distribution": {
                         **centered.metadata["delta_distribution"], "delta1_shift": .1}})
    prefix = tmp_path / "legacy"
    dgp.save_mechanism(legacy, prefix)
    restored = dgp.load_mechanism(prefix)
    assert restored.mechanism_hash == legacy.mechanism_hash
    np.testing.assert_array_equal(restored.arrays["delta_1"], centered.arrays["delta_1"])
    assert "contrast_shift_rule" not in restored.metadata["delta_distribution"]
    fresh = dgp.draw_parameters(legacy.config, legacy.parameter_seed)
    assert fresh.mechanism_hash != legacy.mechanism_hash
    np.testing.assert_allclose(fresh.arrays["delta_1"], restored.arrays["delta_1"] + .1,
                               rtol=0, atol=1e-15)


def test_new_shift_rule_roundtrip_records_distinct_mechanism(centered, tmp_path):
    shifted = dgp.draw_parameters(replace(centered.config, delta1_shift=.1), centered.parameter_seed)
    prefix = tmp_path / "shifted"
    dgp.save_mechanism(shifted, prefix)
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["metadata"]["delta_distribution"]["contrast_shift_rule"] == "add-after-centered-contrast"
    restored = dgp.load_mechanism(prefix)
    assert restored.mechanism_hash == shifted.mechanism_hash != centered.mechanism_hash
    for name in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(restored, 9, 11)[name],
                                      dgp.sample_data(shifted, 9, 11)[name])
