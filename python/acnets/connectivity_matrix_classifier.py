# custom model
from sklearn.base import BaseEstimator, ClassifierMixin

class MixedReducerModel(ClassifierMixin):
    def __init__(self, select_k=10, n_neighbors=15, n_components=3):
        self.selector = feature_selection.SelectKBest(k=select_k)
        self.reducer = UMAP(n_neighbors=n_neighbors, n_components=n_components) 
        self.model = ensemble.GradientBoostingClassifier()  
    def fit(self, X, y):
        _y_beh = y[:,0]  #  behavioral
        _y = y[:,1]      # classification output
        X_selected = self.selector.fit_transform(X, _y)
        X_reduced = self.reducer.fit_transform(X_selected, _y_beh)
        self.model.fit(X_reduced, _y)
        return self
    def transform(self, X):
        X_selected = self.selector.transform(X)
        X_reduced = self.reducer.transform(X_selected)
        y = self.model.predict(X_reduced)
        return y
    def predict(self, X):
        X_selected = self.selector.transform(X)
        X_reduced = self.reducer.transform(X_selected)
        y = self.model.predict(X_reduced)
        return y

# DEBUG
# pipeline.fit(X_train, np.vstack([y_beh_train, y_train]).T)
# score = pipeline.score(X_test, y_test)
