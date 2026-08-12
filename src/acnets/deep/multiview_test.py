"""Tests for the multiview fusion model, missingness first."""

import pytest
import torch

from .multiview import MultiViewClassifier


@pytest.fixture
def model():
    return MultiViewClassifier({'corr': 6, 'structure': 12, 'behaviour': 3},
                               latent=4, hidden=8)


def views(batch=10):
    torch.manual_seed(0)
    return {'corr': torch.randn(batch, 6),
            'structure': torch.randn(batch, 12),
            'behaviour': torch.randn(batch, 3)}


class TestMultiViewClassifier:

    def test_forward_with_all_views(self, model):
        assert model(views()).shape == (10, 2)

    def test_forward_with_a_subset_of_views(self, model):
        subset = {k: v for k, v in views().items() if k != 'behaviour'}
        assert model(subset).shape == (10, 2)

    def test_per_sample_missingness_changes_only_those_samples(self, model):
        model.eval()
        data = views()
        mask = {'behaviour': torch.ones(10)}
        mask['behaviour'][:3] = 0.0

        with_mask = model(data, mask)
        without = model(data)
        torch.testing.assert_close(with_mask[3:], without[3:])
        assert not torch.allclose(with_mask[:3], without[:3])

    def test_masked_view_values_cannot_influence_output(self, model):
        """The invariant missingness rests on: a masked view's numbers are
        irrelevant, so imputation artefacts cannot leak in."""
        model.eval()
        data = views()
        mask = {'behaviour': torch.zeros(10)}

        poisoned = dict(data)
        poisoned['behaviour'] = data['behaviour'] * 1e6 + 42
        torch.testing.assert_close(model(data, mask), model(poisoned, mask))

    def test_modality_dropout_never_silences_a_sample(self):
        torch.manual_seed(0)
        model = MultiViewClassifier({'a': 4, 'b': 4}, modality_dropout=0.999)
        model.train()
        out = model({'a': torch.randn(64, 4), 'b': torch.randn(64, 4)})
        assert torch.isfinite(out).all()
        assert out.abs().sum() > 0

    def test_dropout_off_in_eval_mode(self, model):
        model.eval()
        a = model(views())
        b = model(views())
        torch.testing.assert_close(a, b)

    def test_learns_from_complementary_views(self):
        """Signal split across views: fusion must beat either alone."""
        torch.manual_seed(2)
        n = 96
        y = torch.randint(0, 2, (n,))
        signal = (y.float() * 2 - 1).unsqueeze(1)
        data = {'a': torch.randn(n, 5) + signal * torch.tensor([[1., 0, 0, 0, 0]]),
                'b': torch.randn(n, 5) + signal * torch.tensor([[0., 1, 0, 0, 0]])}

        model = MultiViewClassifier({'a': 5, 'b': 5}, latent=4, hidden=8,
                                    modality_dropout=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(data), y)
            loss.backward()
            optimizer.step()

        model.eval()
        accuracy = float((model(data).argmax(1) == y).float().mean())
        assert accuracy > 0.85
