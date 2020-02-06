import os
import torch
import numpy as np

from utils import plot_stock_raw, plot_stock_clean

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_windows):
        super(StockDataset, self).__init__()
        self.data = stock_windows

    # (optional) custom_data = any stock data you want to plot that has a 'Close' and 'Date' column
    # (optional) index = if the user provides several stock filenames to the dataset's constructor, index allows the user to plot a specific stock data's history
    #                    if index is not provided, the first stock's history will be used
    #                    if a list is provided, the stocks at those indices are plotted
    def plot_stock_clean(self, index = 0):
        if type(index) is int: # only a singular index was provided, instead of a list of indices
            index = self.__len__() if index > self.__len__() else index # control for out of bounds errors
            plot_stock_clean(self.data[index])
        else: # a list of indices was provided
            for ind in index:
                ind = self.__len__() if ind > self.__len__() else ind # control for out of bounds errors
                plot_stock_clean(self.data[ind])

    def __getitem__(self, index):
        price, label = self.data[index]
        return {'prices': torch.FloatTensor(price), 'labels': torch.as_tensor(label)}

    def __len__(self):
        return len(self.data)
