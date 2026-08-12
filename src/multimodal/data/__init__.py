"""Data modules, re-exported from `acnets.datasets`.

Both data modules existed twice, once here and once under `acnets/datasets/`,
differing only in where they got the timeseries from: the copies here read the
cached netCDF, the `acnets` ones parcellated the raw BIDS tree. Neither was
wrong, and neither worked in the other's situation. The surviving
implementations do both -- cache first, parcellate if it is absent -- so there
is one class per dataset again.

`Julia2018Dataset`, a `keras.utils.PyDataset` wrapper that batched two lists,
is gone with Keras. `torch.utils.data.TensorDataset` does the same job and is
what the data modules already build.
"""

from acnets.datasets import Julia2018DataModule, LEMONDataModule

__all__ = ['Julia2018DataModule', 'LEMONDataModule']
