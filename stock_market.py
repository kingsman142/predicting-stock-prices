import os
import numpy as np
import pandas as pd

# simulates the stock market
# reads in a bunch of stocks, and only outputs prices for stocks that are on the market
#   => e.g. if one Stock A opened in 1999 and Stock B in 2007, our bot should only buy Stock A until we get to 2007
# should also output all prices of each stock on a given day, as well as their previous price

class StockMarket():
    def __init__(self, stock_fns = ["aa.us.txt"], window_size = 250, sma_or_ema = 1, smoothing_window_size = 26):
        self.stock_fns = stock_fns if type(stock_fns) is list else [stock_fns] # a user can pass in a single stock fn, or a list of stock fns, but make sure to always convert it to a list
        self.WINDOW_SIZE = window_size
        self.sma_or_ema = sma_or_ema # 0 = use Simple Moving Average, 1 = use Exponential Moving Average, any other number = else don't use either SMA or EMA
        self.smoothing_window_size = smoothing_window_size
        self.stock_market = None

        # iterate over all the stock files that belong to this dataset
        for stock_fn in self.stock_fns:
            stock_ticker = stock_fn.split(".")[0] # transform "aa.us.txt" into ["aa", "us", "txt"] into "aa"

            # read in this stock's data into a pandas dataframe
            path = os.path.join("data", "Stocks", stock_fn)
            data_csv = pd.read_csv(path, header = 0).sort_values('Date')
            close_prices = data_csv.loc[:, 'Close'].as_matrix()

            print("Num rows in {}: {}".format(stock_fn, len(data_csv)))

            # extract training and testing windows, and concatenate them onto our already existing training and testing data
            windows = self.preprocess_stocks(close_prices)

            historical_prices = data_csv.loc[:, ['Date', 'Close']].as_matrix() # similar matrix to the above but with the dates included
            historical_prices.rename(columns = {'Close': stock_ticker}, inplace = True) # rename the 'Close' column to the stock's ticker name so we can make a stock matrix involving different stocks later

            if self.stock_market is None:
                self.stock_market = historical_prices
            else:
                self.stock_market = pd.merge(self.stock_market, historical_prices, how = 'outer', on = 'Date')

            stock_starting_date = historical_prices.iloc[0]['Date'] # get the date on the 0th row (assuming this data is sorted by Date)
            stock_market_start_index = self.stock_market.loc[self.stock_market['Date'] == stock_starting_date].index[0] # get the row number, after merging in this stock's prices, where this stock began to be sold on the stock market

            self.stock_market[stock_ticker] = np.nan # clear all the elements in the column (just floats) with NaN so we can replace them with their corresponding windows below
            self.stock_market[stock_ticker][stock_market_start_index + self.WINDOW_SIZE : stock_market_start_index + self.WINDOW_SIZE + len(windows)] = windows # replace this stock's prices with actual window (previous K prices) and the ground-truth price
        self.stock_market.sort_values('Date') # sanity check to make sure all the prices are in chronological order

    def preprocess_stocks(self, stock_data):
        # select training and testing data
        prices = stock_data.reshape(-1, 1)

        # scale the data between 0 and 1
        # also, reshape the data and transform the test set
        scaler = MinMaxScaler()
        prices = scaler.fit_transform(prices).reshape(-1)

        if self.sma_or_ema == 0: # perform simple moving average smoothing
            prices = self.simple_mov_avg(prices)
        elif self.sma_or_ema == 1: # perform exponential moving average smoothing
            prices = self.exp_mov_avg(prices)

        prices_windows = self.create_windows(prices)

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
        return [np.average(stock_data[(i-self.smoothing_window_size):i]) for i in range(self.smoothing_window_size, len(stock_data)+1)]

    def create_windows(self, stock_data):
        output = []
        for index in range(len(stock_data) - self.WINDOW_SIZE - 1):
            data_input = stock_data[index : (index + self.WINDOW_SIZE)]
            data_label = stock_data[index + self.WINDOW_SIZE]
            output.append((data_input, data_label))
        return output

    def get_iteration(self, index):
        return self.stock_market.iloc[index] # get all the prices for a given day

    def get_num_iterations(self):
        return len(self.stock_market) # get number of days we can possibly buy/sell
