""" Defines a design matrix"""

from dataclasses import dataclass


@dataclass
class DesignMatrix():

  # .data
  # .masker
  # .X
    # regressors/predictors

  def __post_init__(self):
    raise NotImplementedError()

  def convolve(self):
    # convolve() -> with HRF
    raise NotImplementedError()

  def plot(self):
    raise NotImplementedError()

  def corr(self):
    raise NotImplementedError()

  # Variance Inflation Factors (VIFs) exceeding 10 indicate significant multicollinearity and will likely require intervention.

  def filter(self):
    # fitering (dct, etc)
    raise NotImplementedError()

  def add_confounds(self):
    raise NotImplementedError()

  def fit(self, X):
    raise NotImplementedError()

  def regress(self):
    raise NotImplementedError()

