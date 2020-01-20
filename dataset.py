import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from utils import plot_stock
from sklearn.preprocessing import MinMaxScaler

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_fn = "aa.us.txt", window_size = 250, train = 0.8, test = 0.2):
        super(StockDataset, self).__init__()

        self.stock_fn = stock_fn
        self.WINDOW_SIZE = window_size
        self.TRAIN = train
        self.TEST = test

        path = os.path.join("data", "Stocks", stock_fn)
        self.data = pd.read_csv(path, header = 0).sort_values('Date')
        close_prices = self.data.loc[:, 'Close'].as_matrix()

        print("Num rows: {}".format(len(self.data)))

        self.preprocess_stocks(close_prices)

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
        EMA = 0.0
        gamma = 0.1
        for index in range(len(train)):
            EMA = gamma * train[index] + (1 - gamma) * EMA
            train[index] = EMA
        all_closing_data = np.concatenate([train, test], axis = 0)

        train_windows = self.create_windows(train, self.WINDOW_SIZE)
        test_windows = self.create_windows(test, self.WINDOW_SIZE)

        self.train = train_windows
        self.test = test_windows

    def plot_stock(self):
        plot_stock(self.data)
        return

    def create_windows(self, stock_data, window_size):
        output = []
        for index in range(len(stock_data) - window_size - 1):
            data_input = stock_data[index : (index + window_size)]
            data_label = stock_data[index + window_size]
            output.append((data_input, data_label))
        return output

    def __getitem__(self, index):
        price, label = self.train[index]
        return {'prices': torch.FloatTensor(price), 'labels': torch.as_tensor(label)}

    def __len__(self):
        return len(self.train)
