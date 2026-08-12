"""A plain replacement for pytorch_lightning.LightningDataModule.

The Lightning class contributed three things the data modules here actually
used: a `hparams` dict populated by `save_hyperparameters()`, a `setup(stage)`
hook, and `{train,val,test}_dataloader()` methods a trainer would call. None of
that needs a framework, and keeping the same method names means notebooks that
call `dm.setup('fit')` and `dm.train_dataloader()` do not change.

`save_hyperparameters` here is explicit rather than frame-inspecting: pass the
values you want recorded. Lightning read them out of the caller's stack frame,
which is impressive until a subclass takes `**kwargs`.
"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader


class DataModule:
    """Base class: hyperparameters, a setup hook, and three dataloaders."""

    def __init__(self):
        self.hparams: dict = {}
        self.train = None
        self.val = None
        self.test = None
        self.full_data = None

    def save_hyperparameters(self, **kwargs) -> None:
        self.hparams.update(kwargs)

    def prepare_data(self) -> None:
        """Anything that must happen once, before setup. Downloads, caching."""

    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError('call setup() before requesting a dataloader')
        num_workers = self.hparams.get('num_workers', 0)
        return DataLoader(
            dataset,
            batch_size=self.hparams.get('batch_size', 8),
            shuffle=shuffle,
            num_workers=num_workers,
            # persistent_workers is invalid with num_workers=0, which is the
            # sensible default for datasets small enough to sit in memory.
            persistent_workers=num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train, shuffle=self.hparams.get('shuffle', True))

    def val_dataloader(self) -> DataLoader:
        # NOTE the same split as test; there is no separate validation set.
        return self._loader(self.test)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test)


# Zero, unless the caller asks for more. The datasets here are at most a few
# hundred megabytes and load fully into memory, so worker processes cost more in
# startup than they save. The previous default was `os.cpu_count() - 1`, which
# spawned eleven processes to iterate 32 subjects.
#
# A constant rather than a function so it can sit in an argument default without
# being re-evaluated per call.
DEFAULT_WORKERS = 0


def max_workers() -> int:
    """For the one job that is actually IO-bound: parcellating raw NIfTIs."""
    return max(1, (os.cpu_count() or 1) - 1)
