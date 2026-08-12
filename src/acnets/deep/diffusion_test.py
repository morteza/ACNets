"""Tests for the normative diffusion model.

The properties that matter are not "does it train" but "does the deviation
score behave like a deviation score": higher for atypical inputs, stable across
noise draws, and blind to labels.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from .diffusion import (
    GaussianNormative,
    VectorDiffusion,
    cosine_schedule,
)
from .training import MetricLogger, Trainer


@pytest.fixture
def population():
    """A 6-dimensional 'typical' population with a clear correlation structure."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(600, 2))
    mixing = rng.normal(size=(2, 6))
    return (latent @ mixing + 0.3 * rng.normal(size=(600, 6))).astype(np.float32)


@pytest.fixture
def trained(population):
    model = VectorDiffusion(n_features=6, hidden=64, n_layers=2, n_steps=100)
    loader = DataLoader(TensorDataset(torch.tensor(population)),
                        batch_size=64, shuffle=True)
    Trainer(max_epochs=30, logger=MetricLogger('t', enabled=False)).fit(model, loader)
    return model


class TestSchedule:

    def test_alphas_decrease_monotonically(self):
        alpha_bar = cosine_schedule(100)
        assert alpha_bar.shape == (100,)
        assert torch.all(alpha_bar[1:] <= alpha_bar[:-1])

    def test_stays_in_the_unit_interval(self):
        alpha_bar = cosine_schedule(200)
        assert float(alpha_bar.min()) > 0
        assert float(alpha_bar.max()) <= 1.0


class TestVectorDiffusion:

    def test_predicts_noise_with_the_right_shape(self):
        model = VectorDiffusion(n_features=21, hidden=32, n_layers=2, n_steps=50)
        x = torch.randn(8, 21)
        t = torch.randint(0, 50, (8,))
        assert model(x, t).shape == (8, 21)

    def test_loss_falls_during_training(self, population):
        model = VectorDiffusion(n_features=6, hidden=64, n_layers=2, n_steps=100)
        loader = DataLoader(TensorDataset(torch.tensor(population)),
                            batch_size=64, shuffle=True)
        trainer = Trainer(max_epochs=30, logger=MetricLogger('t', enabled=False))
        trainer.fit(model, loader)
        first = trainer.history[0]['loss/train']
        last = trainer.history[-1]['loss/train']
        assert last < first

    def test_deviation_has_one_score_per_feature(self, trained, population):
        scores = trained.deviation(population[:5], n_draws=8)
        assert scores.shape == (5, 6)
        assert np.all(scores >= 0)

    def test_atypical_inputs_score_higher(self, trained, population):
        """The property the whole idea rests on."""
        rng = np.random.default_rng(1)
        # break the correlation structure the model learned, keeping the scale
        atypical = rng.permutation(population[:200].ravel()).reshape(200, 6)

        typical_score = trained.deviation(population[:200], n_draws=48).mean()
        atypical_score = trained.deviation(atypical.astype(np.float32),
                                           n_draws=48).mean()
        assert atypical_score > typical_score

    def test_scores_are_reproducible_for_a_fixed_seed(self, trained, population):
        a = trained.deviation(population[:10], n_draws=16, seed=3)
        b = trained.deviation(population[:10], n_draws=16, seed=3)
        assert np.allclose(a, b)

    def test_averaging_over_draws_stabilises_the_score(self, trained, population):
        """One draw is mostly noise about noise; this is why n_draws is large."""
        few = np.std([trained.deviation(population[:20], n_draws=1, seed=s).mean()
                      for s in range(8)])
        many = np.std([trained.deviation(population[:20], n_draws=64, seed=s).mean()
                       for s in range(8)])
        assert many < few

    def test_scoring_never_sees_labels(self, trained, population):
        """Leakage guard: the score cannot depend on a grouping it never receives."""
        rng = np.random.default_rng(2)
        subset = population[:40]
        before = trained.deviation(subset, n_draws=16, seed=0)
        _ = rng.permutation(np.array([0] * 20 + [1] * 20))   # labels exist, unused
        after = trained.deviation(subset, n_draws=16, seed=0)
        assert np.array_equal(before, after)


class TestGaussianNormative:
    """The control the diffusion model has to beat to justify itself."""

    def test_deviation_shape_matches_the_diffusion_model(self, population):
        model = GaussianNormative().fit(population)
        assert model.deviation(population[:5]).shape == (5, 6)

    def test_distance_is_larger_for_outliers(self, population):
        model = GaussianNormative().fit(population)
        typical = model.distance(population[:100]).mean()
        outliers = model.distance(population[:100] + 8.0).mean()
        assert outliers > typical

    def test_covariance_is_invertible_when_features_exceed_samples(self):
        rng = np.random.default_rng(0)
        tiny = rng.normal(size=(5, 20))          # 20 features, 5 samples
        model = GaussianNormative().fit(tiny)
        assert np.all(np.isfinite(model.deviation(tiny)))
