##### RULES #####
# 1) No buying of more than 10 stocks of a company per day
# 2) Every stock you invest in, you update its price at closing every day
# 3) Every stock you've invested in today, check its price every 2 hours to make sure it's not crashing
#################

##### NOTES #####
# 1) One iteration from the data loader does not necessarily imply one day's change in the stock price; the stock market is closed on weekends
#    and is only one for 242 days of the year for trading. (e.g. 410 iterations / 242 trading days = 1.69 trading years to get the reported RoI)
#################

import os
import glob
import torch
import requests
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd

from model import StockPredictor
from buyer_seller import BuyerSeller
from preprocess import StockPreprocessor
from dataset import StockDataset
from utils import plot_stock_raw, plot_roi
from stock_market import StockMarket

# set MODEL_LOAD_NAME to a specific name to load a specific model or set it to None to load the newest trained model
MODEL_LOAD_NAME = None # "train80_windowsize50_epochs1_batchsize1_hiddensize150_lr0.001_smoothing1_smoothingsize50"
if MODEL_LOAD_NAME is None:
    list_of_files = glob.glob(os.path.join("models", "*")) # select all models
    latest_file = max(list_of_files, key = os.path.getctime) # sort them by newest date and pick the most recent one
    MODEL_LOAD_NAME = os.path.split(latest_file)[1] # convert 'models/model_name' to 'model_name'
MODEL_HIDDEN_SIZE = int(MODEL_LOAD_NAME.split("_")[4][10:]) # convert "train{}_windowsize{}_epochs{}_batchsize{}_hiddensize{}_lr{}" to "hiddensize{}" to "{}" and then to an int

print("Loading model {} ...".format(MODEL_LOAD_NAME))

# set network hyperparameters
TRAIN = 0.8
WINDOW_SIZE = 50
SMA_OR_EMA = 1 # 0 = use Simple Moving Average, 1 = use Exponential Moving Average, any other number = else don't use either SMA or EMA
SMOOTHING_WINDOW_SIZE = 26 # 12-day = 30% RoI, 26-day = 10% RoI, 50-day = 5.5% RoI, 100-day = 5% RoI
INITIAL_MONEY = 10

# set up model
model = StockPredictor(hidden_size = MODEL_HIDDEN_SIZE)
model.load_state_dict(torch.load(os.path.join("models", MODEL_LOAD_NAME)))

# determine which OOD stocks to use
ood_stock_fns = ["acbi.us.txt", "googl.us.txt", "jpm.us.txt", "goex.us.txt", "goro.us.txt", "lea.us.txt", "tsla.us.txt"] #["acbi.us.txt", "googl.us.txt", "jpm.us.txt"] #["acbi.us.txt", "googl.us.txt"] #["acbi.us.txt", "googl.us.txt", "jpm.us.txt"] # ["acbi.us.txt"]

# preprocess the dataset and set up a stock market so we can pull prices on a daily basis
market = StockMarket(stock_fns = ood_stock_fns, window_size = WINDOW_SIZE, sma_or_ema = SMA_OR_EMA, smoothing_window_size = SMOOTHING_WINDOW_SIZE)
agent = BuyerSeller(initial_money = INITIAL_MONEY)

# test the model
curr_prices = None
fluctuation_correct = 0
buy_correct = 0
buy_total = 0
sell_correct = 0
sell_total = 0
increase_correct = 0
increase_total = 0
decrease_correct = 0
decrease_total = 0
for day_number in range(len(market)): # iterate over batches
    date, day_prices = market.get_iteration_prices(day_number) # get the price of each stock for today

    # store the predicted and ground-truth prices for each stock on this day
    pred_prices = []
    true_prices = []

    for stock_ticker in day_prices.index: # the .index attribute for a Series is equivalent to .columns for a DataFrame
        pred = 0
        label = 0
        if not pd.isna(day_prices[stock_ticker]): # detect if NaN value (the stock has not been opened yet by this date)
            prices, label = day_prices[stock_ticker] # input prices and ground-truth price prediction

            prices = torch.FloatTensor(prices).unsqueeze(0) # convert to tensor so we can pass it into the model
            pred = model(prices).item() # predicted price

        pred_prices.append(pred)
        true_prices.append(label)

    if curr_prices is None: # will only be true on the first iteration
        curr_prices = [0] * len(pred_prices)
    stock_analysis_model = pd.DataFrame([curr_prices, pred_prices], columns = day_prices.index)

    # buying/selling logic
    stocks_bought, stocks_sold, num_active_stocks = agent.buy_sell_or_stay(stock_analysis_model)
    if day_number % 100 == 0:
        print("Date: {}, Day {}/{} -- # active stocks: {}, # bought: {}, # sold: {}, Curr money: {}".format( \
                date, day_number + 1, len(market), num_active_stocks, len(stocks_bought), len(stocks_sold), \
                round(agent.get_curr_investment_money(), 2)))
        print("Pred prices: {}".format(pred_prices))
        print("True prices: {}".format(true_prices))
        print(agent.get_stock_counts())
        print(agent.get_stock_prices())

    # prepare for next iteration
    curr_prices = true_prices

agent.sell_all() # sell all stocks
profit_loss_amount = agent.get_total_money() - INITIAL_MONEY # amount of money we gained or lost at the end of the simulation
roi_percent = profit_loss_amount / INITIAL_MONEY # compared to the initial investment, how much of the initial investment did we make in profit/loss?
print("Final total money: {}, Final savings: {}, Profit/Loss: {}, RoI%: {}".format(round(agent.get_total_money(), 4), round(agent.get_curr_savings(), 4), round(profit_loss_amount, 4), round(roi_percent, 4)))
