import numpy as np


class BaseSampler:
    def __init__(self, max_samples=1.0, bootstrap=False, random_state=None):
        """
        Parameters
        ----------
        bootstrap : Boolean
            if True then use bootstrap sampling
        max_samples : float in [0;1]
            proportion of sampled examples
        """
        self.random_state = np.random.RandomState(random_state)
        self.bootstrap = bootstrap
        self.max_samples = max_samples


    def sample_indices(self, n_objects):
        """
        Parameters
        ----------
        n_objects : int > 0
            number of sampling objects
        """
        if isinstance(self.max_samples, float):
            number_of_samples = int(n_objects * self.max_samples)
        elif self.max_samples == 'sqrt':
            number_of_samples = int(n_objects**0.5)
        indeces = self.random_state.choice(np.arange(n_objects), number_of_samples, replace = self.bootstrap)
        return indeces


    def sample(self, x, y=None):
        indeces = self.sample_indices(x.shape[0])
        if not isinstance(y, type(None)):
            return x[indeces, :], y[indeces]
        else:
            return x[indeces, :]


class ObjectSampler(BaseSampler):
    def __init__(self, max_samples=1.0, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled, y_sampled : numpy ndarrays of shape (n_samples, n_features) and (n_samples,)
        """
        indeces = self.sample_indices(x.shape[0])
        if not isinstance(y, type(None)):
            return x[indeces, :], y[indeces]
        else:
            return x[indeces, :]
        


class FeatureSampler(BaseSampler):
    def __init__(self, max_samples=1.0, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled : numpy ndarrays of shape (n_objects, n_features_sampled)
        """
        indeces = self.sample_indices(x.shape[1])
        if not isinstance(y, type(None)):
            return x[:, indeces], y
        else:
            return x[:, indeces]
