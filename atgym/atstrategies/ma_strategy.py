from abc import abstractmethod

from atgym.atstrategies import AbstractStrategy
from atgym.atutils import compute_sma, compute_lwa, compute_ewa


class MAStrategy(AbstractStrategy):
    """Moving average strategy. """
    def __init__(self, obs_shape, name=None):
        if name is None:
            name = "ma_strategy"
        super().__init__(obs_shape=obs_shape, name=name)

    def predict(self, obs):
        """
        Computes the moving average from observation and
        places buy/sell signal when MA crosses with price.
        """
        # use only 'Open' column
        price = obs[:, 0]

        avg_before = self.compute_average(price[:-1])
        avg_after = self.compute_average(price[1:])

        price_before, price_after = price[-2], price[-1]

        # buy when price crosses moving average on the way up
        if avg_before > price_before and avg_after < price_after:
            return 1
        # sell when price crosses moving average on the way down
        if avg_before < price_before and avg_after > price_after:
            return 2
        # otherwise, hold
        return 0

    @abstractmethod
    def compute_average(self, price):
        pass


class SMAStrategy(MAStrategy):
    """Simple moving average strategy. """
    def __init__(self, obs_shape, name=None):
        if name is None:
            name = "sma_strategy"
        super().__init__(obs_shape=obs_shape, name=name)

    def compute_average(self, price):
        return compute_sma(price)


class WMAStrategy(MAStrategy):
    """Weighted moving average strategy. """
    def __init__(self, obs_shape, name=None):
        if name is None:
            name = "wma_strategy"
        super().__init__(obs_shape=obs_shape, name=name)

    def compute_average(self, price):
        """Places greater importance on recent data using linear weights. """
        return compute_lwa(price)


class EMAStrategy(MAStrategy):
    """Exponentially weighted moving average strategy. """
    def __init__(self, obs_shape, name=None, gamma=0.99):
        if name is None:
            name = "ema_strategy"
        super().__init__(obs_shape=obs_shape, name=name)
        self.gamma = gamma

    def compute_average(self, price):
        """
        Places greater importance on recent data using exponential weights.
        """
        return compute_ewa(price, self.gamma)
