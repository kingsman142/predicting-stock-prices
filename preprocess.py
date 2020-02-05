import os
import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class StockPreprocessor():
    def __init__(self, stock_fns = ["aa.us.txt"], window_size = 250, train = 0.8, test = 0.2, sma_or_ema = 1, smoothing_window_size = 50):
        self.stock_fns = stock_fns if type(stock_fns) is list else [stock_fns] # a user can pass in a single stock fn, or a list of stock fns, but make sure to always convert it to a list
        self.WINDOW_SIZE = window_size
        self.TRAIN = train
        self.TEST = test
        self.data = []
        self.train_data = []
        self.test_data = []
        self.sma_or_ema = sma_or_ema # 0 = use Simple Moving Average, 1 = use Exponential Moving Average, any other number = else don't use either SMA or EMA
        self.smoothing_window_size = smoothing_window_size

        # iterate over all the stock files that belong to this dataset
        for stock_fn in self.stock_fns:
            # read in this stock's data into a pandas dataframe
            path = os.path.join("data", "Stocks", stock_fn)
            data_csv = pd.read_csv(path, header = 0).sort_values('Date')
            close_prices = data_csv.loc[:, 'Close'].as_matrix()

            # make sure this dataset belongs to the overall data variable belonging to this class so we can plot them later on
            self.data.append(data_csv)
            print("Num rows in {}: {}".format(stock_fn, len(data_csv)))

            # extract training and testing windows, and concatenate them onto our already existing training and testing data
            train_windows, test_windows = self.preprocess_stocks(close_prices)
            self.train_data += train_windows
            self.test_data += test_windows

    def preprocess_stocks(self, stock_data):
        # select training and testing data
        train = stock_data[: int(self.TRAIN * len(stock_data))].reshape(-1, 1)
        test = stock_data[int(self.TRAIN * len(stock_data)): ].reshape(-1, 1)

        # scale the data between 0 and 1
        # also, reshape the data and transform the test set
        scaler = MinMaxScaler()
        train = scaler.fit_transform(train).reshape(-1)
        test = scaler.transform(test).reshape(-1)

        if self.sma_or_ema == 0: # perform simple moving average smoothing
            train = self.simple_mov_avg(train)
            test = self.simple_mov_avg(test)
        elif self.sma_or_ema == 1: # perform exponential moving average smoothing
            train = self.exp_mov_avg(train)
            test = self.exp_mov_avg(test)

        train_windows = self.create_windows(train)
        test_windows = self.create_windows(test)

        return train_windows, test_windows

    # optional -- Exponential Moving Average (EMA)
    def exp_mov_avg(self, stock_data):
        EMA = 0.0
        gamma = 2 / (self.smoothing_window_size + 1) # general formula = 2 / (window_size + 1) (e.g. 20 days = 0.0952, 50 days = 0.0392, and 100 days = 0.0198)
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

    def get_splits(self):
        return self.train_data, self.test_data

    def get_all_data(self):
        return self.train_data + self.test_data
