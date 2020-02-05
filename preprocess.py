import os
import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class StockPreprocessor():
    def __init__(self, stock_fns = ["aa.us.txt"], window_size = 250, train = 0.8, test = 0.2):
        self.stock_fns = stock_fns if type(stock_fns) is list else [stock_fns] # a user can pass in a single stock fn, or a list of stock fns, but make sure to always convert it to a list
        self.WINDOW_SIZE = window_size
        self.TRAIN = train
        self.TEST = test
        self.data = []
        self.train_data = []
        self.test_data = []

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

        # perform exponential moving average smoothing
        '''EMA = 0.0
        gamma = 0.1
        for index in range(len(train)):
            EMA = gamma * train[index] + (1 - gamma) * EMA
            train[index] = EMA'''

        # perform simple moving average smoothing
        train = self.sim_mov_avg(train, 50)
        test = self.sim_mov_avg(test, 50)

        train_windows = self.create_windows(train, self.WINDOW_SIZE)
        test_windows = self.create_windows(test, self.WINDOW_SIZE)

        return train_windows, test_windows

    # optional -- Simple Moving Average (SMA)
    def sim_mov_avg(self, stock_data, averaging_window_size):
        return [np.average(stock_data[(i-averaging_window_size):i]) for i in range(averaging_window_size, len(stock_data)+1)]

    def create_windows(self, stock_data, window_size):
        output = []
        for index in range(len(stock_data) - window_size - 1):
            data_input = stock_data[index : (index + window_size)]
            data_label = stock_data[index + window_size]
            output.append((data_input, data_label))
        return output

    def get_splits(self):
        return self.train_data, self.test_data