import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, probability=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probability = probability

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree(BaseEstimator):
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=None, min_samples_leaf=1):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.classes_ = None
        self.n_features_in_ = None
        self._effective_n_features = None

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if self.n_features is None:
            self._effective_n_features = self.n_features_in_
        elif isinstance(self.n_features, str):
            if self.n_features == 'sqrt':
                self._effective_n_features = int(np.sqrt(self.n_features_in_)) if self.n_features_in_ > 0 else 1
            elif self.n_features == 'log2':
                self._effective_n_features = int(np.log2(self.n_features_in_)) if self.n_features_in_ > 1 else 1
            else:
                raise ValueError(f"Unsupported string value for n_features: {self.n_features}. Expected 'sqrt', 'log2'.")
        elif isinstance(self.n_features, (int, float)):
            if 0 < self.n_features <= 1.0 :
                 self._effective_n_features = int(self.n_features * self.n_features_in_)
            else:
                 self._effective_n_features = int(self.n_features)
        else:
            raise ValueError(f"Unsupported type for n_features: {type(self.n_features)}. Expected 'sqrt', 'log2', int, float, or None.")

        self._effective_n_features = max(1, min(self._effective_n_features if self._effective_n_features is not None else self.n_features_in_, self.n_features_in_))
        if self._effective_n_features == 0 and self.n_features_in_ > 0 :
             self._effective_n_features = 1
        
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        if n_samples == 0:
            value = self.rng.choice(self.classes_) if self.classes_ is not None and len(self.classes_) > 0 else None
            prob = np.ones(len(self.classes_)) / len(self.classes_) if self.classes_ is not None and len(self.classes_) > 0 else None
            return Node(value=value, probability=prob)

        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split or
            n_samples < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            class_counts = Counter(y)
            leaf_probabilities = np.array([class_counts.get(cls, 0) / n_samples if n_samples > 0 else 0.0 for cls in self.classes_])
            return Node(value=leaf_value, probability=leaf_probabilities)

        feat_idxs = self.rng.choice(n_feats, self._effective_n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            class_counts = Counter(y)
            leaf_probabilities = np.array([class_counts.get(cls, 0) / n_samples if n_samples > 0 else 0.0 for cls in self.classes_])
            return Node(value=leaf_value, probability=leaf_probabilities)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            class_counts = Counter(y)
            leaf_probabilities = np.array([class_counts.get(cls, 0) / n_samples if n_samples > 0 else 0.0 for cls in self.classes_])
            return Node(value=leaf_value, probability=leaf_probabilities)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            if len(thresholds) <= 1:
                continue

            for thr in thresholds:
                temp_left_idxs, temp_right_idxs = self._split(X_column, thr)
                if len(temp_left_idxs) < self.min_samples_leaf or \
                   len(temp_right_idxs) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        
        if best_gain <= 0:
            return None, None
            
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        
        y_int = y.astype(int) 
        
        min_len_bincount = 0
        if self.classes_ is not None and len(self.classes_) > 0:
            max_class_val = np.max(self.classes_.astype(int)) if len(self.classes_) > 0 else -1
            min_len_bincount = max_class_val + 1
        
        if len(y_int) > 0:
            max_y_val = np.max(y_int)
            min_len_bincount = max(min_len_bincount, max_y_val + 1)

        counts = np.bincount(y_int, minlength=min_len_bincount if min_len_bincount > 0 else None)
        probabilities = counts[counts > 0] / len(y_int)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _most_common_label(self, y):
        if len(y) == 0:
            return self.rng.choice(self.classes_) if self.classes_ is not None and len(self.classes_) > 0 else None
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        if self.root is None:
            raise ValueError("DecisionTree model has not been trained. Call 'fit' first.")
        return np.array([self._traverse_tree(x, self.root)[0] for x in X])

    def predict_proba(self, X):
        if self.root is None:
            raise ValueError("DecisionTree model has not been trained. Call 'fit' first.")
        return np.array([self._traverse_tree(x, self.root)[1] for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value, node.probability

        if node.feature is None:
             return node.value, node.probability

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def get_params(self, deep=True):
        return {
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "min_samples_leaf": self.min_samples_leaf
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        if "random_state" in params:
            self.rng = np.random.RandomState(params["random_state"])
        return self
