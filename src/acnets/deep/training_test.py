"""Tests for the plain-PyTorch training harness.

These are the tests the Lightning version never had, because Lightning was
assumed to be correct. The harness is small enough to check directly, and the
things worth checking are the ones the framework used to hide: that the loop
actually reduces a loss, that the device default is CPU, that dropout is not
left switched on, and that a two-optimizer model updates both halves.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .conditional_gan import ConditionalGAN
from .gan import GAN
from .training import (
    MetricLogger,
    Trainer,
    accuracy,
    checkpoint_id,
    load_checkpoint,
    resolve_device,
)


@pytest.fixture
def linear_data():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=16)


class Tiny(nn.Module):
    def __init__(self, n_features=8):
        super().__init__()
        self.net = nn.Linear(n_features, 2)

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, batch, device):
        x, y = batch[0].to(device), batch[1].to(device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss, {'accuracy': accuracy(logits, y)}


class TestResolveDevice:

    def test_defaults_to_cpu(self, monkeypatch):
        """Never silently picks an accelerator; MPS disagrees with CPU on LSTM."""
        monkeypatch.delenv('ACNETS_DEVICE', raising=False)
        assert resolve_device().type == 'cpu'

    def test_env_var_opts_in(self, monkeypatch):
        monkeypatch.setenv('ACNETS_DEVICE', 'cpu')
        assert resolve_device().type == 'cpu'

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv('ACNETS_DEVICE', 'meta')
        assert resolve_device('cpu').type == 'cpu'

    def test_auto_is_available_but_not_the_default(self):
        assert resolve_device('auto').type in ('cpu', 'mps', 'cuda')


class TestAccuracy:

    def test_counts_argmax_matches(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        assert accuracy(logits, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)

    def test_accepts_one_hot_targets(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert accuracy(logits, one_hot) == pytest.approx(1.0)

    def test_is_stateless(self):
        """torchmetrics needed .reset() between reads and the old code forgot."""
        logits = torch.tensor([[2.0, 0.0]])
        target = torch.tensor([0])
        assert accuracy(logits, target) == accuracy(logits, target) == 1.0


class TestTrainer:

    def test_reduces_the_loss(self, linear_data):
        trainer = Trainer(max_epochs=30, logger=None)
        trainer.fit(Tiny(), linear_data)

        losses = [record['loss/train'] for record in trainer.history]
        assert losses[-1] < losses[0]

    def test_records_one_history_entry_per_epoch(self, linear_data):
        trainer = Trainer(max_epochs=5, logger=None)
        trainer.fit(Tiny(), linear_data)
        assert len(trainer.history) == 5
        assert [r['epoch'] for r in trainer.history] == [0, 1, 2, 3, 4]

    def test_validation_metrics_are_labelled_separately(self, linear_data):
        trainer = Trainer(max_epochs=2, logger=None)
        trainer.fit(Tiny(), linear_data, linear_data)
        record = trainer.history[-1]
        assert 'loss/train' in record
        assert 'loss/val' in record

    def test_leaves_the_model_in_training_mode(self, linear_data):
        """evaluate() switches to eval; forgetting to switch back silently
        disables dropout for every epoch after the first validation pass."""
        model = Tiny()
        Trainer(max_epochs=2, logger=None).fit(model, linear_data, linear_data)
        assert model.training

    def test_writes_a_checkpoint_that_round_trips(self, linear_data, tmp_path):
        model = Tiny()
        trainer = Trainer(max_epochs=2, logger=None, checkpoint_dir=tmp_path)
        trainer.fit(model, linear_data)

        assert (tmp_path / 'last.pt').exists()
        restored = load_checkpoint(Tiny(), tmp_path / 'last.pt')
        for a, b in zip(model.state_dict().values(), restored.state_dict().values(), strict=False):
            torch.testing.assert_close(a, b)

    def test_disabled_logger_keeps_history_without_writing(self, linear_data, tmp_path):
        logger = MetricLogger('run', root=tmp_path, enabled=False)
        Trainer(max_epochs=3, logger=logger).fit(Tiny(), linear_data)

        assert len(logger.history['loss/train']) == 3
        assert not list(tmp_path.iterdir())


class TestCheckpointId:

    def test_is_stable_across_key_order(self):
        assert (checkpoint_id({'a': 1, 'b': 2})
                == checkpoint_id({'b': 2, 'a': 1}))

    def test_differs_for_different_hyperparameters(self):
        assert checkpoint_id({'a': 1}) != checkpoint_id({'a': 2})


class TestGAN:

    def test_both_halves_update(self):
        """The generator and discriminator each get their own optimizer step."""
        torch.manual_seed(0)
        model = GAN(input_size=6, latent_size=4, n_classes=2)
        before_g = [p.clone() for p in model.generator.parameters()]
        before_d = [p.clone() for p in model.discriminator.parameters()]

        x = torch.randn(8, 6)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        Trainer(max_epochs=2, logger=None).fit(model, loader)

        assert any(not torch.allclose(a, b)
                   for a, b in zip(before_g, model.generator.parameters(), strict=False))
        assert any(not torch.allclose(a, b)
                   for a, b in zip(before_d, model.discriminator.parameters(), strict=False))

    def test_skips_a_single_sample_batch(self):
        """BatchNorm1d cannot compute statistics from one sample."""
        model = GAN(input_size=6, latent_size=4, n_classes=2)
        optimizers = model.configure_optimizers()
        batch = (torch.randn(1, 6), torch.tensor([0]))
        assert model.training_step(batch, optimizers, torch.device('cpu')) == {}

    def test_unlabelled_batches_spread_the_real_target(self):
        """Without labels, 'real' means 'some real class, no opinion which'."""
        model = GAN(input_size=6, latent_size=4, n_classes=2)
        targets = model._real_targets(torch.zeros(3, 6), None)
        assert targets.shape == (3, 3)
        np.testing.assert_allclose(targets[0].numpy(), [0.0, 0.5, 0.5])

    def test_labelled_batches_shift_past_the_fake_class(self):
        model = GAN(input_size=6, latent_size=4, n_classes=2)
        y = torch.tensor([1, 2])  # already shifted by the caller
        targets = model._real_targets(torch.zeros(2, 6), y)
        np.testing.assert_allclose(targets.numpy(),
                                   [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_discriminator_returns_logits_not_probabilities(self):
        """A Sigmoid before cross_entropy double-normalises and flattens
        gradients; the Lightning version had one."""
        model = GAN(input_size=6, latent_size=4, n_classes=2)
        out = model.discriminator(torch.randn(16, 6) * 10)
        assert out.min() < 0.0 or out.max() > 1.0


class TestConditionalGAN:

    def test_generator_is_conditioned_on_the_label(self):
        torch.manual_seed(0)
        model = ConditionalGAN(input_dim=6, latent_dim=4, n_classes=2)
        model.eval()
        y = torch.zeros(256, dtype=torch.long)
        class_0 = model.sample(y).mean(dim=0)
        class_1 = model.sample(torch.ones_like(y)).mean(dim=0)
        assert not torch.allclose(class_0, class_1, atol=1e-3)

    def test_training_updates_both_halves(self):
        torch.manual_seed(0)
        model = ConditionalGAN(input_dim=6, latent_dim=4, n_classes=2)
        before_g = [p.clone() for p in model.generator.parameters()]
        before_d = [p.clone() for p in model.discriminator.parameters()]

        loader = DataLoader(
            TensorDataset(torch.randn(16, 6), torch.randint(0, 2, (16,))),
            batch_size=8)
        Trainer(max_epochs=2, logger=None).fit(model, loader)

        assert any(not torch.allclose(a, b)
                   for a, b in zip(before_g, model.generator.parameters(), strict=False))
        assert any(not torch.allclose(a, b)
                   for a, b in zip(before_d, model.discriminator.parameters(), strict=False))

    def test_generator_target_accounts_for_the_fake_class(self):
        """Class 0 means 'generated'. Targeting y rather than y+1 asked the
        generator to make class 1 look fake, which is what the Keras version did.
        """
        torch.manual_seed(0)
        model = ConditionalGAN(input_dim=6, latent_dim=4, n_classes=2)
        optimizers = model.configure_optimizers()
        batch = (torch.randn(8, 6), torch.zeros(8, dtype=torch.long))

        metrics = model.training_step(batch, optimizers, torch.device('cpu'))
        assert metrics['g_loss'] > 0
        assert metrics['combined_loss'] == pytest.approx(
            metrics['d_loss'] + metrics['g_loss'], rel=1e-6)
