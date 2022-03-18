from atgym.atstrategies import AbstractStrategy


class EngulfingStrategy(AbstractStrategy):
    """
    Detect bullish/bearish engulfing pattern and place corresponding order.
    """
    def __init__(self, obs_shape, name=None, trend_thresh=0.75):
        if name is None:
            name = "engulfing_strategy"
        super().__init__(obs_shape=obs_shape, name=name)
        self.trend_tresh = trend_thresh

    def num_candlesticks(self, prev_obs):
        """
        Determine number of candlesticks that close higher than open and close
        lower than open. Assumes columns of observation in format
        ['Open', 'High', 'Low', 'Close'].
        """
        white_candlesticks = 0
        for i in range(len(prev_obs)):
            if prev_obs[i, 3] - prev_obs[i, 0] > 0:
                white_candlesticks += 1
        return white_candlesticks, len(prev_obs) - white_candlesticks

    def pattern(self, obs):
        """
        Check last two candles to determine whether there is a reversal.
        """
        first = obs[-2, :]
        second = obs[-1, :]

        # reversal moving up
        if second[3] > first[0] and first[0] > first[3]:
            return "bullish"
        if second[3] < first[0] and first[0] < first[3]:
            return "bearish"
        return "none"

    def predict(self, obs):
        """
        Detects when there is overshadowing in the candlestick, the body of
        which completely overlaps or engulfs the body of the previous step's
        candlestick.
        """
        white_candles, black_candles = self.num_candlesticks(obs[:-2, :])
        frac_white = white_candles / (white_candles + black_candles)
        frac_black = black_candles / (white_candles + black_candles)

        pattern = self.pattern(obs)
        # buy when bullish about a trend reversal
        if pattern == "bullish" and frac_black > self.trend_tresh:
            return 1
        # sell when bearish
        if pattern == "bearish" and frac_white > self.trend_tresh:
            return 2
        return 0
