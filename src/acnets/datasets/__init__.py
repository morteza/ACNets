"""Dataset loaders.

The two data modules pull in torch, which lives in the `dl` extra rather than
the base dependencies. Importing them eagerly here made `acnets.datasets`
unimportable without it, and since `tests/conftest.py` reaches into this package
for the connectivity fixtures, that took unrelated tests down with it. The
lazy `__getattr__` below keeps `from acnets.datasets import LEMONDataModule`
working where torch is installed without imposing it on the classic pipeline.
"""

__all__ = ['Julia2018DataModule', 'LEMONDataModule']


def __getattr__(name):
    if name == 'Julia2018DataModule':
        from .julia2018 import Julia2018DataModule
        return Julia2018DataModule
    if name == 'LEMONDataModule':
        from .lemon import LEMONDataModule
        return LEMONDataModule
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(__all__)
