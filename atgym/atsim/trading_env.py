from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf

from atgym.atutils import generate_indicator

plt.style.use('seaborn-dark')
plt.rcParams["figure.figsize"] = (8, 6)


class TradingEnv(Env):
    """
    Reinforcement learning trading environment.

    State: ndarray of shape (`look_back`, 4)
    - `look_back` - number of past samples
    - columns are ['Open', 'High', 'Low', 'Close']

    Action:
    - 0 (nothing), 1 (buy), 2 (sell)

    Reward:
    - 0 until the end, at which point the change in market value of holdings
      along with trading cost is calculated.
    """
    def __init__(self, data, look_back=20, trading_cost_pct=0.0):
        """
        Arguments:
        data (dict) - data for which trades are simulated on
        - key (string) - name of ticker
        - value (pandas DataFrame) - data loaded by DataLoader
        look_back (int) - number of intervals to look back on
        trading_cost_pct (float) - percentage 0.0 to 100.0 of trading cost
        """
        # load in data
        self.data = data
        # only use Open, High, Low, and Close
        self.data_columns = ['Open', 'High', 'Low', 'Close']
        for ticker in self.data.keys():
            self.data[ticker] = self.data[ticker][self.data_columns]
        self.num_tickers = len(self.data)
        # environment settings
        self.look_back = look_back
        self.trading_cost = trading_cost_pct / 100

        # actions: buy, sell, nothing
        self.action_space = Discrete(3)

        # observations:
        self.box_structure = np.ones(
            shape=(look_back, len(self.data_columns)), dtype=np.float32)
        low = -np.inf * self.box_structure
        high = np.inf * self.box_structure
        self.observation_space = Box(low=low, high=high)

        # states: past open, high, low, and close
        tickers = list(self.data.keys())
        days_timestamps = {}
        # get days to simulate on for each ticker
        for ticker in tickers:
            days_timestamps[ticker] = self.data[ticker].\
                resample('D').mean().dropna().index
        self.days = {}
        for ticker, days_timestamp in days_timestamps.items():
            self.days[ticker] = [ts.strftime('%Y-%m-%d') for ts in days_timestamp]

        self.state = self.reset()

    def reset(self, ticker=None, day=None):
        """
        Resets by starting a new day and new ticker.

        Arguments:
        ticker (str) - ticker to simulate on. Default None for random ticker.
        day (str) - format %Y-%m-%d for day to simulate on. Default None for
            random day.

        Returns:
        state (np.ndarray) - past prices
        """
        # reset tracking variables
        # history to keep count of buys and sells
        self.history = {'buy': [], 'sell': []}
        for _ in range(self.look_back):
            self.history['buy'].append(np.nan)
            self.history['sell'].append(np.nan)
        # number of long positions (negative indicates short position)
        self.position = 0
        
        # randomly choose ticker and day
        # some day data are not long enough to be analyzed (happens rarely)
        # keep choosing until there is one long enough
        if ticker == None and day == None:
            while True:
                # randomly select the ticker/day if both are None
                self.cur_ticker = np.random.choice(list(self.data.keys()))
                # randomly select the current day if day == None
                self.cur_day = np.random.choice(self.days[self.cur_ticker])
                # get current day's data
                self.day_data = self.data[self.cur_ticker].loc[str(self.cur_day)]
                
                if len(self.day_data) >= 100:
                    break
        # current ticker and day already chosen
        else:
            self.cur_ticker = ticker
            self.cur_day = day
            self.day_data = self.data[self.cur_ticker].loc[str(self.cur_day)]

        # normalize values by the first entry
        self.factor = self.day_data.iloc[0]['Open']
        self.day_data = self.day_data / self.factor
        # initialize state
        self.state = self.day_data.head(self.look_back).to_numpy().astype(np.float32)
        # initialize pointer
        self.timestep = self.look_back

        return self.state

    def _get_balance(self, ending_norm_price):
        """Get balance at end of the day. """
        balance = 0.0
        for price in self.history['buy']:
            if not np.isnan(price):
                balance -= price
        for price in self.history['sell']:
            if not np.isnan(price):
                balance += price

        balance += ending_norm_price * self.position
        return balance

    def evaluate(self, strategy, normalized=False, verbose=True):
        """
        Evaluate a strategy over all of the data.

        Arguments:
        strategy (BaseStrategy object) - strategy to evaluate
        normalized (bool) - True for normalized returns, False for raw returns
        verbose (bool) - prints mean rewards for each ticker

        Returns:
        returns (float) - mean reward over tickers and days
        """
        if verbose:
            print(f"evaluating {str(strategy)}...")

        all_rewards = []
        for ticker, days in self.days.items():
            ticker_rewards = []
            for day in days:
                episode_rewards = []
                done = False
                obs = self.reset(ticker, day)
                if len(self.day_data) < 100:
                    continue
                while not done:
                    action = strategy.predict(obs)
                    obs, reward, done, info = self.step(action)
                    episode_rewards.append(reward)
                
                episode_rewards = sum(episode_rewards)
                if not normalized:
                    episode_rewards *= self.factor
                
                ticker_rewards.append(episode_rewards)

            if verbose:
                if ticker_rewards:
                    ticker_rewards_print = np.array(ticker_rewards).mean()
                    print(ticker.rjust(10) + ": " + str(ticker_rewards_print))
                else:
                    print(ticker.rjust(10))
            all_rewards += ticker_rewards
        all_rewards_mean = np.array(all_rewards).mean()
        if verbose:
            print("total rewards: " + str(all_rewards_mean))

        return all_rewards_mean

    def step(self, action):
        """Get next 2-minute interval price. """
        # last price as simple average between open and close price
        cur_day_data = self.day_data.iloc[self.timestep-1]
        cur_price = (cur_day_data['Open'] + cur_day_data['Close']) / 2
        # new price
        new_day_data = self.day_data.iloc[self.timestep]
        new_price = (new_day_data['Open'] + new_day_data['Close']) / 2

        # apply action
        if action == 0:
            self.history['buy'].append(np.nan)
            self.history['sell'].append(np.nan)
        elif action == 1:
            self.history['buy'].append(cur_price)
            self.history['sell'].append(np.nan)
            self.position += 1
        else:
            self.history['buy'].append(np.nan)
            self.history['sell'].append(cur_price)
            self.position -= 1

        # shift down and fill next row with the new data
        self.state[:-1] = self.state[1:]
        self.state[-1] = new_day_data.to_numpy()
        # normalize by first entry
        self.state /= self.state[0, 0]

        # increment timestep
        self.timestep += 1

        # calculate reward
        if action == 1 or action == 2:
            reward = - self.trading_cost * cur_price
        else:
            reward = 0.0

        # check if done
        if self.timestep >= len(self.day_data):
            done = True
            # update reward
            reward += self._get_balance(new_price)
            # update dataframe
            self.day_data['Buy'] = self.history['buy']
            self.day_data['Sell'] = self.history['sell']
        else:
            done = False

        info = {
            "position": self.position,
            "cur_day": self.cur_day,
            "cur_ticker": self.cur_ticker,
            "day_data": self.day_data,
            "num_positions": self.position,
            "history": self.history
        }

        # return step information
        return self.state, reward, done, info

    def render(self, normalized=False,
               plot_rsi=True, robt=70.0, rost=30.0,
               plot_ma=True, ma_method="s"):
        """
        Plot buy and sell points on mpf plot.

        Arguments:
        plot_rsi (bool) - plots relative strength index as a subplot
        robt (float) - between 0.0 and 100.0 - plots overbought threshold
        rost (float) - between 0.0 and 100.0 - plots oversold threshold
        plot_ma (bool) - plots moving average on plot
        ma_method (string) - moving average method
        - "s": simple moving average
        - "l": linearly weighted average
        - "e": exponentially weighted average
        """

        # calculate average price and plot it
        price = self.day_data
        if not normalized:
            price *= self.factor
        s = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'axes.grid': False})

        # get buy/sell signals
        signals = []
        if sum(np.isnan(price['Buy'])) < len(price['Buy']):
            signals.append(
                mpf.make_addplot(price['Buy'], type='scatter', markersize=30, marker='^', color='green')
            )
        if sum(np.isnan(price['Sell'])) < len(price['Sell']):
            signals.append(
                mpf.make_addplot(price['Sell'], type='scatter', markersize=30, marker='v', color='red')
            )

        # plot moving average
        ma_plots = []
        if plot_ma:
            ma = generate_indicator(price['Open'], method=ma_method + "ma", look_back=self.look_back-1)
            ma_plots.append(
                mpf.make_addplot(ma, color='blue')
            )

        # plot rsi
        rsi_plots = []
        if plot_rsi:
            rsi_plots.append(
                mpf.make_addplot([rost for _ in range(len(price))], panel=1, ylabel='rsi', ylim=[0, 120], color='yellow'),
            )
            rsi_plots.append(
                mpf.make_addplot([robt for _ in range(len(price))], panel=1, ylim=[0, 120], color='orange'),
            )
            # compute rsi
            rsi = generate_indicator(price['Open'], method="rsi", look_back=self.look_back-1)
            rsi_plots.append(
                mpf.make_addplot(rsi, panel=1, ylim=[0, 120], color='blue')
            )

        mpf.plot(price,
                 addplot=signals + ma_plots + rsi_plots,
                 type='candle',
                 title=f'{self.cur_ticker} stock on {self.cur_day}')
