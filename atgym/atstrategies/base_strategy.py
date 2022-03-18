from abc import abstractmethod


class AbstractStrategy:
    """
    Abstract class for Strategy objects. Must implement predict method.
    """
    names_to_count = {}

    def __init__(self, obs_shape, name=None):
        """Initializes Strategy object and gives it a name."""
        self.obs_shape = obs_shape

        if name is None:
            name = "strategy"

        if name in AbstractStrategy.names_to_count:
            AbstractStrategy.names_to_count[name] += 1
        else:
            AbstractStrategy.names_to_count[name] = 0
        self.name = name + "_" + str(AbstractStrategy.names_to_count[name])

    @abstractmethod
    def predict(self, obs):
        """
        Predicts action - 0 (nothing), 1 (buy), 2 (sell) - from
        observation.
        """
        pass

    def __repr__(self):
        """Returns name as string representation. """
        return self.name
