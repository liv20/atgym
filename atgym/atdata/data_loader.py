import os
import datetime as dt

import requests
from bs4 import BeautifulSoup

import pandas as pd

import yfinance as yf

from sklearn.model_selection import train_test_split


class TradingDataLoader:
    save_path = "yf_data"
    url = 'https://companiesmarketcap.com/tech/largest-tech-companies-by-market-cap/'

    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Loads the market data for specified tickers using yahoo finance api.

        Arguments:
        tickers (list of str) - tickers to fetch, default will fetch top 100
            tickers on NASDAQ
        start_date (str) "%Y-%m-%d" - beginning of period (default: 60 days ago)
        end_date (str) "%Y-%m-%d" - end of period (default: current day)
        """
        self.interval = '2m'

        now = dt.datetime.now()
        if start_date is None:
            self.start_date = (now - dt.timedelta(days=55)).strftime("%Y-%m-%d")

        if end_date is None:
            self.end_date = now.strftime("%Y-%m-%d")

        if not os.path.exists(TradingDataLoader.save_path):
            os.makedirs(TradingDataLoader.save_path)

        self.tickers = tickers
        if self.tickers is None:
            self.tickers = self.fetch_tickers()
        self.tickers = sorted(self.tickers)

    def fetch_tickers(self):
        r = requests.get(TradingDataLoader.url)
        soup = BeautifulSoup(r.text, 'html.parser')
        tickers = []
        for tag in soup.find_all("div", {"class": "company-code"}):
            tickers.append(tag.get_text())
        return tickers

    def _return_dict(self):
        return {ticker: df for (ticker, df) in zip(self.tickers, self.dfs)}

    def download(self, save=True):
        """Returns list of pandas DataFrame of downloaded data. """
        self.dfs = []
        for ticker in self.tickers:
            df = yf.download(ticker,
                             start=self.start_date, end=self.end_date,
                             interval=self.interval)
            self.dfs.append(df)

        if save:
            self.save_data()
            print(f"data saved in folder {TradingDataLoader.save_path}")
            print("use load_data to load from folder")
        return self._return_dict()

    def build_filename(self, ticker):
        fname = f'{TradingDataLoader.save_path}/'
        fname += ticker + "_"
        fname += self.start_date + "_" + self.end_date + ".csv"
        return fname

    def ticker_in_folder(self, ticker):
        for file in os.listdir(TradingDataLoader.save_path):
            if ticker in file:
                return True
        return False

    def save_data(self):
        """Save data locally. """
        for i in range(len(self.tickers)):
            found = False
            for file in os.listdir(TradingDataLoader.save_path):
                if self.tickers[i] in file:
                    found = True
            if not found:
                self.dfs[i].to_csv(self.build_filename(self.tickers[i]))

    def load_data(self):
        """Load local data. """
        self.dfs, self.tickers = [], []
        files = os.listdir(TradingDataLoader.save_path)
        for file in files:
            filepath = TradingDataLoader.save_path + "/" + file
            df = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
            self.tickers.append(file.split("_")[0])
            self.dfs.append(df)

        return self._return_dict()

    def train_test_split(self, train_size=None, test_size=0.2,
                         split_by_time=False, random_state=None):
        """
        Split data into train and test by day, returning two dictionaries.

        Arguments:
        train_size (float) - proportion of dataset to include in train split.
          If None, the value is automatically set to the complement of test
          size.
        test_size (float) - proportion of data to include in test split.
          Default 0.2.
        split_by_time (bool) - default splits train and test randomly. Otherwise,
          train is the earlier portion of the dataset.

        Returns:
        (d1, d2) (two dictionaries) - train data and test data
        """
        def collect(df, days):
            """Gathers days data in a list and returns pandas DataFrame. """
            data = []
            for day in days:
                day_data = df.loc[day]
                data.append(day_data)
            return pd.concat(data)

        train, test = {}, {}
        for i in range(len(self.dfs)):
            days = self.dfs[i].resample('D').mean().dropna().index.\
                map(lambda x: str(x).split(' ')[0])
            if split_by_time:
                if not train_size:
                    train_size = 1 - test_size

                bound = int(len(days) * train_size)

                days_train = days[:bound]
                days_test = days[bound:]
            else:
                days_train, days_test = train_test_split(days, train_size=train_size, test_size=test_size,
                                                         random_state=random_state)

            ticker_train = collect(self.dfs[i], days_train)
            ticker_test = collect(self.dfs[i], days_test)

            train[self.tickers[i]] = ticker_train
            test[self.tickers[i]] = ticker_test

        return train, test


if __name__ == "__main__":
    if not os.path.exists(TradingDataLoader.save_path):
        tdl = TradingDataLoader()
        tdl.download()
