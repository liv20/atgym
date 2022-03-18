from atgym.atstrategies import AbstractStrategy
from atgym.atutils import compute_rsi


class RSIStrategy(AbstractStrategy):
    """
    Use relative strength index as a momentum indicator to measure
    whether asset is overbought or oversold.
    """
    def __init__(self, obs_shape, name=None,
                 oversold_thresh=30, overbought_thresh=70):
        if name is None:
            name = "rsi_strategy"
        super().__init__(obs_shape=obs_shape, name=name)

        self.oversold_thresh = oversold_thresh
        self.overbought_thresh = overbought_thresh

    def predict(self, obs):
        """
        Computes RSI and places buy/sell signal when RSI crosses threshold.
        """
        # use only 'Open' column
        price = obs[:, 0]

        rsi_before, rsi_after = compute_rsi(price[:-1]), compute_rsi(price[1:])

        # crosses to overbought (rsi > overbought_thresh) region -> sell
        if rsi_before < self.overbought_thresh and \
           rsi_after > self.overbought_thresh:
            return 2
        # cross to oversold (rsi < oversold_thresh) region -> buy
        if rsi_before > self.oversold_thresh and \
           rsi_after < self.oversold_thresh:
            return 1
        # otherwise hold
        return 0
