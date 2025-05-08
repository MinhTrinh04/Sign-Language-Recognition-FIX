import numpy as np
from collections import Counter
from newDecisionTree import DecisionTree
from sklearn.base import BaseEstimator, ClassifierMixin

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2,
                 n_features=None, random_state=None,
                 min_samples_leaf=1, bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self.n_features_in_ = None
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.trees = []

        for i in range(self.n_trees):
            tree_random_state = None
            if self.random_state is not None:
                tree_random_state = self.random_state + i
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )
            
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_samples(X, y)
            else:
                X_sample, y_sample = X, y
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        if n_samples == 0:
            return X, y
        idxs = self.rng.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y_preds_for_sample):
        if len(y_preds_for_sample) == 0:
            return self.rng.choice(self.classes_) if self.classes_ is not None and len(self.classes_) > 0 else None
        counter = Counter(y_preds_for_sample)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        if not self.trees:
            raise ValueError("RandomForest model has not been trained. Call 'fit' first.")
        tree_predictions = np.array([tree.predict(X) for tree in self.trees]).T
        if tree_predictions.shape[0] == 0 :
             return np.array([])
        y_pred = np.array([self._most_common_label(preds) for preds in tree_predictions])
        return y_pred

    def predict_proba(self, X):
        if not self.trees:
            raise ValueError("RandomForest model has not been trained. Call 'fit' first.")
        if X.shape[0] == 0:
            return np.empty((0, len(self.classes_) if self.classes_ is not None else 0))

        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        avg_probas = np.mean(tree_probas, axis=0)
        return avg_probas

    def get_params(self, deep=True):
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "min_samples_leaf": self.min_samples_leaf,
            "bootstrap": self.bootstrap
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        if "random_state" in params:
            self.rng = np.random.RandomState(params["random_state"])
        return self

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

