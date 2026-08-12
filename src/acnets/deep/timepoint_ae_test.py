"""Tests for the per-timepoint autoencoder and its latent-connectivity head."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from .timepoint_ae import LatentConnectivityClassifier, TimepointAE, latent_summary
from .training import Trainer


def correlated_channels(n=512, channels=6, seed=0):
    """Synthetic data with real cross-channel covariance to learn."""
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(n, 2))
    mixing = rng.normal(size=(2, channels))
    x = sources @ mixing + 0.3 * rng.normal(size=(n, channels))
    x = (x - x.mean(0)) / x.std(0)
    return torch.tensor(x, dtype=torch.float32)


class TestTimepointAE:

    def test_bottleneck_is_enforced(self):
        with pytest.raises(ValueError, match='bottleneck'):
            TimepointAE(n_channels=6, latent=6)

    def test_learns_covariance_structure(self):
        """On genuinely correlated channels the AE must beat the mean predictor
        decisively — this is the property the Seq2SeqAE failed on LEMON."""
        torch.manual_seed(0)
        x = correlated_channels()
        model = TimepointAE(n_channels=6, latent=2)
        trainer = Trainer(max_epochs=60, logger=None, lr=1e-2)
        trainer.fit(model, DataLoader(TensorDataset(x), batch_size=64))
        assert trainer.history[-1]['relative_error/train'] < 0.5

    def test_encoder_is_shared_across_timepoints(self):
        model = TimepointAE(n_channels=6, latent=3)
        run = torch.randn(2, 10, 6)
        flat = model.encoder(run.reshape(-1, 6)).reshape(2, 10, 3)
        torch.testing.assert_close(model.latent_series(run), flat)


class TestLatentSummary:

    def test_shape(self):
        z = torch.randn(5, 100, 4)
        assert latent_summary(z).shape == (5, 4 + 10)

    def test_covariance_block_matches_numpy(self):
        z = torch.randn(1, 200, 3, dtype=torch.float64)
        summary = latent_summary(z)[0]
        expected = np.cov(z[0].numpy().T)
        rows, cols = np.triu_indices(3)
        np.testing.assert_allclose(summary[3:].numpy(), expected[rows, cols],
                                   atol=1e-10)

    def test_rejects_flat_input(self):
        with pytest.raises(ValueError, match='batch, time, latent'):
            latent_summary(torch.randn(10, 4))


class TestLatentConnectivityClassifier:

    def test_freezing_controls_exactly_the_encoder(self):
        model = LatentConnectivityClassifier(TimepointAE(6, latent=3))
        model.set_encoder_frozen(True)
        assert not any(p.requires_grad for p in model.autoencoder.parameters())
        assert all(p.requires_grad for p in model.head.parameters())
        model.set_encoder_frozen(False)
        assert all(p.requires_grad for p in model.autoencoder.parameters())

    def test_separates_groups_differing_in_covariance(self):
        """The features are latent second moments, so a covariance difference
        between groups must be linearly separable after fine-tuning."""
        torch.manual_seed(1)
        rng = np.random.default_rng(1)

        def group(coupling, n=24):
            shared = rng.normal(size=(n, 124, 1))
            noise = rng.normal(size=(n, 124, 6))
            x = noise + coupling * shared  # coupling raises all covariances
            x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
            return torch.tensor(x, dtype=torch.float32)

        x = torch.cat([group(1.2), group(0.0)])
        y = torch.tensor([0] * 24 + [1] * 24)

        model = LatentConnectivityClassifier(TimepointAE(6, latent=3))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(150):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        accuracy = float((model(x).argmax(1) == y).float().mean())
        assert accuracy > 0.9
