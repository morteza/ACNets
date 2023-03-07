from sklearn.base import ClassifierMixin, BaseEstimator


class ConnectivityClassifier(ClassifierMixin, BaseEstimator):
  def __init__(self, estimator='svc', kernel='linear', probability=True) -> None:
    if estimator == 'linear_svc':
      from sklearn.svm import SVC
      self.estimator = SVC(kernel='linear', probability=probability)

    super().__init__()

  def fit(self, X, y=None, **fit_params):  # noqa: N803
    return self.estimator.fit(X, y, **fit_params)

  def transform(self, X):  # noqa: N803
    return self.estimator.transform(X)
