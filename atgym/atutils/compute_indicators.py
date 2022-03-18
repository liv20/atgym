import numpy as np


def compute_sma(price):
    """Compute simple moving average. """
    return np.mean(price)


def compute_lwa(price):
    """Places greater importance on recent data using linear weights. """
    denom = len(price) * (len(price) + 1) / 2
    num = 0.0
    for i in range(len(price)):
        num += (i + 1) * price[i]
    return num / denom


def compute_ewa(price, gamma):
    """Places greater importance on recent data using exponential weights. """
    weights = np.ones(shape=(len(price),), dtype=np.float32)
    for i in range(len(weights)-2, -1, -1):
        weights[i] = weights[i+1] * gamma
    weights_sum = np.sum(weights)

    tot = 0.0
    for i in range(len(price)):
        tot += weights[i] * price[i]
    return tot / weights_sum


def compute_rsi(price):
    """Compute relative strength index from a list of prices. """
    gains, losses = [], []
    for i in range(len(price) - 1):
        if price[i] < 1e-10:
            continue
        pct_change = 100 * (price[i+1] - price[i]) / price[i]
        if pct_change > 0:
            gains.append(pct_change)
        else:
            losses.append(pct_change)
    avg_gain = sum(gains) / (len(price) - 1)
    avg_loss = np.absolute(sum(losses) / (len(price) - 1))
    if avg_loss < 0.01 * avg_gain:
        ratio = 99
    else:
        ratio = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + ratio))
    return rsi


implemented_str = ["sma", "lwa", "ewa", "rsi"]
implemented_method = [compute_sma, compute_lwa, compute_ewa, compute_rsi]


def generate_indicator(price, method, look_back=19):
    """Computes indicator at each price stamp. """
    if type(method) == str:
        if method not in implemented_str:
            raise ValueError(
                f"invalid method {method}, accepts {implemented_str}")
        method = implemented_method[implemented_str.index(method)]

    indicator = np.zeros(shape=(len(price),), dtype=np.float32)
    for i in range(look_back):
        indicator[i] = np.nan
    for i in range(look_back, len(price)):
        indicator[i] = method(price[i+1-look_back: i+1])
    return indicator
