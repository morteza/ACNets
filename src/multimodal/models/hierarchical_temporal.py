import keras

class HierarchicalTemporalModel(keras.Model):
  """

  Notes:
    - Step1: The model receives region-level timeseries
      and fits network-level timeseries, one per network.
      Each region only belongs to one network.
    - Step2: The model receives network-level timeseries
      and calculates similarity matrix between them, e.g. cosine similarity.
    - Step3: The model receives similarity matrix and flattens it and only
      keep the upper triangular part as it's symmetric.
    - Step4: The flatten similarity vector is then fed to a multi-layer dense
      later to predict the target variable, i.e., AVGP or NVGP.

    - Normalization: region-level timeseries are normalized by min-max scaling.

  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, x, training=False):
    raise NotImplementedError("HierarchicalTemporalModel.call()")

  def train_step(self, data):
    x, y = data
    raise NotImplementedError("HierarchicalTemporalModel.train_step()")
