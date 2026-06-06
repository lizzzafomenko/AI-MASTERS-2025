from typing import Union

import numpy as np
#from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree

from sem_dt_rf.random_forest.sampler import FeatureSampler, ObjectSampler, BaseSampler
from sem_dt_rf.decision_tree.decision_tree import ClassificationDecisionTree, DecisionTree


class RandomForest:
    def __init__(self, base_estimator,
                 object_sampler: BaseSampler, feature_sampler: BaseSampler,
                 n_estimators=10, bootstrap = True, verbose = 1, **params):
        """
        n_estimators : int
            number of base estimators
        base_estimator : class for base_estimator with fit(), predict() and predict_proba() methods
        feature_sampler : instance of FeatureSampler
        object_sampler : instance of ObjectSampler
        n_estimators : int
            number of base_estimators
        params : kwargs
            params for base_estimator initialization
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.feature_sampler = feature_sampler
        self.object_sampler = object_sampler
        self.estimators = []
        self.indices = []
        self.params = params
        self.verbose = verbose
        self.n_features = None

    def fit(self, x, y):
        """
        for i in range(self.n_estimators):
            1) select random objects and answers for train
            2) select random indices of features for current estimator
            3) fit base_estimator (don't forget to remain only selected features)
            4) save base_estimator (self.estimators) and feature indices (self.indices)

        NOTE that self.base_estimator is class and you should init it with
        self.base_estimator(**self.params) before fitting
        """
        self.n_features = x.shape[1]
        for i in range(self.n_estimators):
            if self.verbose:
                print(f'fitting tree num {i}')
            x_sample, y_sample = self.object_sampler.sample(x, y)
            ind = self.feature_sampler.sample_indices(x.shape[1])
            x_sample = x_sample[:, ind]
            be = self.base_estimator(**self.params)
            be.fit(x_sample, y_sample)
            self.estimators.append(be)
            self.indices.append(ind)
        return self

    def predict_proba(self, x):
        """
        Returns
        -------
        probas : numpy ndarrays of shape (n_objects, n_classes)

        Calculate mean value of all probas from base_estimators
        Don't forget, that each estimator has its own feature indices for prediction
        """
        if not (0 < len(self.estimators) == len(self.indices)):
            raise RuntimeError('Bagger is not fitted', (len(self.estimators), len(self.indices)))

        total_proba = []
        for i in range(self.n_estimators):
            ind = self.indices[i]
            p = self.estimators[i].predict_proba(x[:, ind])
            total_proba.append(p)
        
        return np.array(total_proba).mean(axis = 0)



    def predict(self, x):
        """
        Returns
        -------
        predictions : numpy ndarrays of shape (n_objects, )
        """
        probas = self.predict_proba(x)
        class_predictions = probas.argmax(axis = 1)
        return class_predictions


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=30, max_objects_samples=0.9, max_features_samples=0.8,
                 max_depth=None, min_samples_leaf=1, random_state=None, bootstrap = True, verbose = 1, **params):
        base_estimator = ClassificationDecisionTree
        object_sampler = ObjectSampler(max_samples=max_objects_samples, random_state=random_state)
        feature_sampler = FeatureSampler(max_samples=max_features_samples, random_state=random_state)

        super().__init__(
            base_estimator=base_estimator,
            object_sampler=object_sampler,
            feature_sampler=feature_sampler,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            verbose = verbose,
            **params,
        )

    def feature_importances_(self):
        all_importances = np.zeros(self.n_features)
        for tree, ind in zip(self.estimators, self.indices):
            tree_imp = tree.feature_importances_()
            if tree_imp is not False:
                for local_i, global_i in enumerate(ind):
                    all_importances[global_i] += tree_imp[local_i]
        
        return all_importances / all_importances.sum()
