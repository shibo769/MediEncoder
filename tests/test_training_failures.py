"""Invalid nuisance subsets must never return an untrained random model."""
import numpy as np
import pytest

from mediencoder.models import train_nuisance_nn, train_autoencoder


@pytest.mark.parametrize("which", ["train", "validation"])
def test_empty_treatment_stratum_fails_before_training(which):
    x = np.ones((4, 2))
    y = np.ones(4)
    args = (x[:0], y[:0], x, y) if which == "train" else (x, y, x[:0], y[:0])
    with pytest.raises(ValueError, match="nonempty"):
        train_nuisance_nn(*args, epochs=1)


def test_nonfinite_nuisance_values_rejected():
    x, y = np.ones((4, 2)), np.ones(4)
    y[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train_nuisance_nn(x, y, x, np.ones(4), epochs=1)


def test_zero_epochs_is_not_a_successful_fit():
    x, y = np.ones((4, 2)), np.ones(4)
    with pytest.raises(ValueError, match="epochs"):
        train_nuisance_nn(x, y, x, y, epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        train_autoencoder(x, latent_dim=1, epochs=0)


def test_autoencoder_rejects_invalid_validation():
    with pytest.raises(ValueError, match="nonempty"):
        train_autoencoder(np.ones((4, 2)), X_val=np.empty((0, 2)), latent_dim=1)
