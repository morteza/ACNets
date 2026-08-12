"""A small MLP with the sklearn surface the multiverse runner expects.

This is what puts a deep model *inside* the specification grid, as a value of
the classifier axis, rather than beside it: `fit`/`predict` over the same
connectivity feature vectors the SVM and XGBoost see, deterministic under its
seed, CPU-only by default. The `pretraining='lemon'` axis composes with it for
free, because that axis acts on the features (tangent reference, scaler)
upstream of any classifier.

torch is imported lazily so environments without the `dl` extra fail at
construction time with a clear message instead of at import of the runner.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Two-layer MLP with an sklearn-compatible surface.

    Inheriting from `BaseEstimator`/`ClassifierMixin` rather than duck-typing is
    what lets this drop into a `Pipeline`, a `GridSearchCV` or a `cross_val_*`
    helper. Duck-typing was enough for `runner._make_classifier`, which only
    calls fit/predict directly, but sklearn >= 1.6 asks every estimator in a
    composed object for `__sklearn_tags__` and raises `AttributeError` when it
    is missing -- so `Pipeline([('scale', StandardScaler()), ('clf', this)])`
    failed while the same model used standalone worked.

    `ClassifierMixin` must come first: it has to be left of `BaseEstimator` in
    the MRO for the tag machinery to resolve.
    """

    def __init__(self, hidden: int = 16, epochs: int = 100, lr: float = 1e-2,
                 batch_size: int = 8, random_state: int = 0):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _torch(self):
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "the 'mlp' classifier needs torch; install the dl extra "
                '(uv sync --all-extras)') from error
        return torch

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch = self._torch()
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)

        x = torch.tensor(np.asarray(X), dtype=torch.float32)
        targets = torch.tensor(np.asarray(y), dtype=torch.long)
        self.classes_ = np.unique(y)

        self._model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], self.hidden),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden, len(self.classes_)),
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        n = x.shape[0]
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    self._model(x[batch]), targets[batch])
                loss.backward()
                optimizer.step()
        self._model.eval()
        return self

    def _logits(self, X: np.ndarray):
        torch = self._torch()
        with torch.no_grad():
            return self._model(torch.tensor(np.asarray(X), dtype=torch.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._logits(X).argmax(1).numpy()]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch()
        return torch.softmax(self._logits(X), dim=1).numpy()

    def attribution(self, X: np.ndarray) -> np.ndarray:
        """Mean |gradient x input| of the class contrast, per feature.

        The gradient of (logit_1 - logit_0) with respect to the input, times
        the input, averaged in magnitude over samples. Comparable in kind to
        the mean |SHAP| the tree models report: both measure how strongly a
        feature carries the discriminative signal, and both are consumed by
        rank only. shap.GradientExplainer agrees with this up to sampling
        noise but costs a background set; the closed form is exact for what
        the agreement analysis needs.
        """
        torch = self._torch()
        x = torch.tensor(np.asarray(X), dtype=torch.float32,
                         requires_grad=True)
        logits = self._model(x)
        contrast = (logits[:, 1] - logits[:, 0]).sum()
        contrast.backward()
        return (x.grad * x.detach()).abs().mean(dim=0).numpy()
