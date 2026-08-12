"""Tests for the two-phase pretrain/finetune protocol.

The phase machinery was the part of the Lightning code with the most implicit
behaviour: `training_step` called `self.unfreeze()` then conditionally
`self.cls_head.freeze()`, and whether the head was actually frozen depended on
a `phase` attribute set several frames away. It is worth asserting.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._fine_tunable import FineTunable


class Toy(FineTunable):
    """Minimal FineTunable: reconstruct, and optionally classify."""

    def __init__(self, n_features=10, n_embeddings=4):
        super().__init__()
        self.save_hyperparameters(n_features=n_features, n_embeddings=n_embeddings)
        self.encoder = nn.Linear(n_features, n_embeddings)
        self.decoder = nn.Linear(n_embeddings, n_features)
        self.cls_head = nn.Linear(n_embeddings, 2)

    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        loss_recon = nn.functional.mse_loss(x_recon, x)
        y = self.cls_head(h) if self.phase == 'finetune' else None
        return y, h, loss_recon


class Loaders:
    """Stands in for a DataModule."""

    def __init__(self, loader):
        self.loader = loader

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.loader

    def val_dataloader(self):
        return self.loader


@pytest.fixture
def datamodule():
    torch.manual_seed(0)
    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    return Loaders(DataLoader(TensorDataset(x, y), batch_size=8))


class TestPhase:

    def test_rejects_an_unknown_phase(self):
        with pytest.raises(ValueError, match='Invalid phase'):
            Toy().phase = 'whatever'

    def test_fit_rejects_a_missing_phase(self, datamodule):
        with pytest.raises(ValueError, match='Invalid phase'):
            Toy().fit(datamodule, max_epochs=1, phase=None)


class TestPretrain:

    def test_freezes_the_classification_head(self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = Toy()
        model.fit(datamodule, max_epochs=2, phase='pretrain', log_root=tmp_path)
        assert not any(p.requires_grad for p in model.cls_head.parameters())

    def test_reports_no_classification_metrics(self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        trainer = Toy().fit(datamodule, max_epochs=2, phase='pretrain',
                            log_root=tmp_path)
        last = trainer.history[-1]
        assert 'loss_recon/train' in last
        assert 'loss_cls/train' not in last
        assert 'accuracy/train' not in last

    def test_writes_a_checkpoint_named_after_the_hyperparameters(
            self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Toy().fit(datamodule, max_epochs=1, phase='pretrain', log_root=tmp_path)
        checkpoints = list((tmp_path / 'checkpoints').rglob('last.pt'))
        assert len(checkpoints) == 1
        assert checkpoints[0].parent.name.startswith('Toy_')


class TestFinetune:

    def test_unfreezes_the_head(self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = Toy()
        model.fit(datamodule, max_epochs=1, phase='pretrain', log_root=tmp_path)
        model.fit(datamodule, max_epochs=1, phase='finetune', log_root=tmp_path)
        assert all(p.requires_grad for p in model.cls_head.parameters())

    def test_adds_classification_metrics(self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        trainer = Toy().fit(datamodule, max_epochs=2, phase='finetune',
                            log_root=tmp_path)
        last = trainer.history[-1]
        for key in ('loss_recon/train', 'loss_cls/train', 'accuracy/train'):
            assert key in last

    def test_head_actually_trains(self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = Toy()
        before = [p.clone() for p in model.cls_head.parameters()]
        model.fit(datamodule, max_epochs=3, phase='finetune', log_root=tmp_path)
        assert any(not torch.allclose(a, b)
                   for a, b in zip(before, model.cls_head.parameters(), strict=False))

    def test_pretrained_weights_round_trip(self, datamodule, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = Toy()
        model.fit(datamodule, max_epochs=2, phase='pretrain', log_root=tmp_path)
        checkpoint = next((tmp_path / 'checkpoints').rglob('last.pt'))

        restored = Toy().load_pretrained(checkpoint)
        torch.testing.assert_close(model.encoder.weight, restored.encoder.weight)


class TestHyperparameters:

    def test_are_recorded_explicitly(self):
        assert Toy(n_features=7).hparams == {'n_features': 7, 'n_embeddings': 4}

    def test_default_to_empty_rather_than_raising(self):
        class Bare(FineTunable):
            pass

        assert Bare().hparams == {}
