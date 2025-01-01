import os
import sys
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# simulates the stock market
# reads in a bunch of stocks, and only outputs prices for stocks that are on the market
#   => e.g. if one Stock A opened in 1999 and Stock B in 2007, our bot should only buy Stock A until we get to 2007
# should also output all prices of each stock on a given day, as well as their previous price

class StockMarket():
    def __init__(self, stock_fns = ["aa.us.txt"], window_size = 250, sma_or_ema = 1, smoothing_window_size = 26, trading_start_date = None):
        self.stock_fns = stock_fns if type(stock_fns) is list else [stock_fns] # a user can pass in a single stock fn, or a list of stock fns, but make sure to always convert it to a list
        self.WINDOW_SIZE = window_size
        self.sma_or_ema = sma_or_ema # 0 = use Simple Moving Average, 1 = use Exponential Moving Average, any other number = else don't use either SMA or EMA
        self.smoothing_window_size = smoothing_window_size
        self.stock_market = None
        self.stock_scalers = {}
        self.trading_start_date = trading_start_date

        self.normalize_stock = False
        self.normalization_window_size = 2500

        def process_stock(stock_fn):
            stock_ticker = stock_fn.split(".")[0] # transform "aa.us.txt" into ["aa", "us", "txt"] into "aa"
            path = os.path.join("data", "Stocks", stock_fn)
            try:
                data_csv = pd.read_csv(path, header=0).sort_values('Date')
            except: # error is most likely "pandas.errors.EmptyDataError: No columns to parse from file"
                return None, None, None, None

            if self.trading_start_date is not None and type(self.trading_start_date) is str and len(self.trading_start_date) >= 4:
                data_csv = data_csv[data_csv['Date'] >= self.trading_start_date] # only begin trading this stock at a specific start date
            close_prices = data_csv.loc[:, 'Close'].to_numpy()

            stock_fn_index = self.stock_fns.index(stock_fn)
            print("({}/{}) Num rows in {}: {}".format(stock_fn_index+1, len(self.stock_fns), stock_fn, len(data_csv)))
            if len(data_csv) < self.WINDOW_SIZE - 1: # or len(data_csv) < self.normalization_window_size:
                return None, None, None, None

            # extract training and testing windows, and concatenate them onto our already existing training/testing set
            windows = self.preprocess_stocks(stock_ticker, close_prices)
            historical_prices = data_csv.loc[:, ['Date', 'Close']]
            historical_prices.rename(columns={'Close': stock_ticker}, inplace=True)

            stock_starting_date = historical_prices.iloc[0]['Date']
            return stock_ticker, historical_prices, windows, stock_starting_date

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_stock, stock_fn): stock_fn for stock_fn in self.stock_fns}
            for future in as_completed(futures):
                stock_ticker, historical_prices, windows, stock_starting_date = future.result()
                if stock_ticker is None:
                    continue

                if self.stock_market is None:
                    self.stock_market = historical_prices
                else:
                    self.stock_market = pd.merge(self.stock_market, historical_prices, how='outer', on='Date')
                self.stock_market.sort_values('Date', inplace=True, ascending=True)
                self.stock_market.reset_index(inplace=True, drop=True)

                stock_market_start_index = self.stock_market.loc[self.stock_market['Date'] == stock_starting_date].index[0]
                self.stock_market[stock_ticker] = np.nan
                self.stock_market[stock_ticker][(stock_market_start_index + self.WINDOW_SIZE):(stock_market_start_index + self.WINDOW_SIZE + len(windows))] = windows

        self.stock_ticker_column_names = self.stock_market.columns[1:]

    def preprocess_stocks(self, stock_ticker, stock_data):
        # select training and testing data
        prices = stock_data.reshape(-1, 1)
        prices_copy = prices.copy()

        if self.sma_or_ema == 0: # perform simple moving average smoothing
            prices = self.simple_mov_avg(prices)
        elif self.sma_or_ema == 1: # perform exponential moving average smoothing
            prices = self.exp_mov_avg(prices)

        if self.normalize_stock:
            # scale the data between 0 and 1
            # also, reshape the data and transform the test set
            self.stock_scalers[stock_ticker] = MinMaxScaler()
            prices = self.stock_scalers[stock_ticker].fit_transform(prices).reshape(-1)

        prices_windows = self.create_windows(prices, prices_copy)

        return prices_windows

    # optional -- Exponential Moving Average (EMA)
    def exp_mov_avg(self, stock_data):
        EMA = 0.0
        gamma = 2 / (self.smoothing_window_size + 1) # general formula = 2 / (window_size + 1) (e.g. 20 days = 0.0952, 50 days = 0.0392, and 100 days = 0.0198), typically 12-day or 26-days are used for short-term EMA and 50-day and 100-day are used for long-term EMA
        for index in range(len(stock_data)):
            EMA = gamma * stock_data[index] + (1 - gamma) * EMA
            stock_data[index] = EMA
        return stock_data

    # optional -- Simple Moving Average (SMA)
    def simple_mov_avg(self, stock_data):
        smoothed_data = [np.average(stock_data[(i-self.smoothing_window_size):i]) for i in range(self.smoothing_window_size, len(stock_data)+1)]
        smoothed_data = np.reshape(smoothed_data, (-1, 1))
        return smoothed_data

    def create_windows(self, stock_data, stock_data_untransformed):
        output = []
        if self.normalize_stock:
            for index in range(len(stock_data) - self.WINDOW_SIZE - 1):
                data_input = stock_data[index : (index + self.WINDOW_SIZE)]
                data_label = stock_data[index + self.WINDOW_SIZE]
                unsmoothed_gt_price = stock_data_untransformed[index + self.WINDOW_SIZE][0] # stock_data_untransformed is the price data that is unsmoothed & unnormalized
                output.append((data_input, data_label, unsmoothed_gt_price))
        else:
            #for index in range(len(stock_data) - self.WINDOW_SIZE - 1):
            #for index in range(len(stock_data) - self.normalization_window_size - 1):
            for index in range(len(stock_data) - self.WINDOW_SIZE - 1):
                new_stock_data = stock_data[0 : (index + self.WINDOW_SIZE)]
                scaler = MinMaxScaler()
                scaler.fit(new_stock_data)
                data_input = scaler.transform(new_stock_data[index : (index + self.WINDOW_SIZE)]).reshape(-1)
                data_label = scaler.transform(stock_data[index + self.WINDOW_SIZE].reshape(1, -1))
                unsmoothed_gt_price = float(stock_data[index + self.WINDOW_SIZE])

                '''new_stock_data = stock_data[index : (index + self.normalization_window_size)]
                scaler = MinMaxScaler()
                scaler.fit(new_stock_data)
                data_input = scaler.transform(new_stock_data[(self.normalization_window_size - self.WINDOW_SIZE):]).reshape(-1)
                data_label = scaler.transform(stock_data[index + self.normalization_window_size].reshape(1, -1))
                unsmoothed_gt_price = float(stock_data[index + self.normalization_window_size])'''

                '''new_stock_data = stock_data[index : (index + 100)]
                scaler = MinMaxScaler()
                scaler.fit(new_stock_data)
                data_input = scaler.transform(new_stock_data[75:]).reshape(-1)
                data_label = scaler.transform(stock_data[index + 100].reshape(1, -1))
                unsmoothed_gt_price = float(stock_data[index + 100])'''

                output.append((data_input, data_label, unsmoothed_gt_price))
        return output

    def get_iteration_prices(self, index):
        return self.stock_market['Date'][index], self.stock_market.iloc[index][self.stock_ticker_column_names] # get all the prices for a given day, but remove the 'Date' column

    def get_stock_scalers(self):
        return self.stock_scalers

    def __len__(self):
        return len(self.stock_market) # get number of days we can possibly buy/sell
