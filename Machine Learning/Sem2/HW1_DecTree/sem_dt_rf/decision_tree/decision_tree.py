from typing import Optional

import numpy as np

from sem_dt_rf.decision_tree.criterio import Criterion, GiniCriterion, EntropyCriterion, MSECriterion
from sem_dt_rf.decision_tree.tree_node import TreeNode


class DecisionTree:
    def __init__(self, max_depth: int = 10, min_samples_leaf: int = 5, min_improvement: Optional[float] = None):
        self.criterion = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement

    def _build_nodes(self, x: np.ndarray, y: np.ndarray, criterion: Criterion, indices: np.ndarray, node: TreeNode):
        """
        Builds tree recursively

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        indices : samples' indices in node,
            np.ndarray.shape = (n_samples, )
            nd.ndarray.dtype = int
        node : current node to split, TreeNode
        """
        if self.max_depth is not None and self.max_depth <= node.depth:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return
        if len(np.unique(y[indices])) <= 1:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return
        if self.min_samples_leaf is not None and 2*self.min_samples_leaf >= len(indices):
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return

        node.find_best_split(x[indices], y[indices], criterion)
        
        if node.feature_id is None or node.threshold is None or node.q_value_max <= 0:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            node.right_child = None
            node.left_child = None
            return

        if self.min_improvement is not None and self.min_improvement >= node.q_value_max:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return

        node.create_children()
        mask = node.get_best_split_mask(x[indices])

        if mask.sum() == 0 or mask.sum() == len(mask):
            node.set_predictions(criterion.get_predict_val(y[indices]))
            node.left_child = None
            node.right_child = None
            return

        self._build_nodes(x, y, criterion, indices[mask], node.left_child)
        self._build_nodes(x, y, criterion, indices[~mask], node.right_child)

    def _get_nodes_predictions(self, x: np.ndarray, predictions: np.ndarray, indices: np.ndarray, node: TreeNode):
        if node.is_terminal():
            predictions[indices, :] = node.predictions
            return
        mask = node.get_best_split_mask(x[indices])
        self._get_nodes_predictions(x, predictions, indices[mask], node.left_child)
        self._get_nodes_predictions(x, predictions, indices[~mask], node.right_child)


class ClassificationDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "gini", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["gini", "entropy"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.n_classes = 0
        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        self.n_classes = np.max(y) + 1#np.unique(y).shape[0]
        self.n_features = x.shape[1]
        criterion = None
        if self.criterion == "gini":
            criterion = GiniCriterion(self.n_classes)
        elif self.criterion == "entropy":
            criterion = EntropyCriterion(self.n_classes)
        self.root = TreeNode(depth=0)
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)
        #self.smth(self.root)


    def smth(self, node: TreeNode):

        if node.is_terminal():
            print('terminal', node.predictions)  # returns nothing because dictionary is mutable
            return

        print(node.predictions)
        print(node.left_child, node.right_child)
        self.smth(node.left_child)
        self.smth(node.right_child)
        return  # returns nothing because dictionary is mutable

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)  # predict class for every object in x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros((x.shape[0], self.n_classes))
        self._get_nodes_predictions(x, predictions, np.arange(x.shape[0]), self.root)
        return predictions
    
    def InOrderTraversal(self, node: TreeNode, feature_importances: dict):

        if node is None or node.is_terminal():
            return  # returns nothing because dictionary is mutable

        self.InOrderTraversal(node.left_child, feature_importances)
        self.InOrderTraversal(node.right_child, feature_importances)

        feature = node.feature_id
        feature_importances[feature] += node.q_value_max


    def feature_importances_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )
        """
        feature_importances = {}
        for f in range(self.n_features):
            feature_importances[f] = 0

        self.InOrderTraversal(self.root, feature_importances)
        output = np.array([feature_importances[f] for f in range(self.n_features)])
        if output.sum() == 0:
            return False
        return output / output.sum()


class RegressionDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "mse", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["gini", "entropy", "mse"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        self.n_features = x.shape[1]
        if self.criterion == "gini":
            criterion = GiniCriterion(self.n_classes)
        elif self.criterion == "entropy":
            criterion = EntropyCriterion(self.n_classes)
        elif self.criterion == 'mse':
            criterion = MSECriterion()
        self.root = TreeNode(depth=0)
        print('building nodes...')
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x)  # predict class for every object in x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros(x.shape[0])
        self._get_nodes_predictions(x, predictions, np.arange(x.shape[0]), self.root)
        return predictions
    
    def InOrderTraversal(self, node: TreeNode, feature_importances: dict):

        print('got node')
        print(node)

        if node is None:
            return  # returns nothing because dictionary is mutable
        
        if node.is_terminal():
            return

        self.InOrderTraversal(node.left_child, feature_importances)
        self.InOrderTraversal(node.right_child, feature_importances)

        feature = node.feature_id
        feature_importances[feature] += node.q_value_max


    def feature_importances_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )
        """
        feature_importances = {}
        for f in range(self.n_features):
            feature_importances[f] = 0

        self.InOrderTraversal(self.root, feature_importances)
        output = np.array([feature_importances[f] for f in range(self.n_features)])
        if output.sum() == 0:
            return False
        return output / output.sum()
