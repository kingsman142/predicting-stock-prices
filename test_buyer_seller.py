##### RULES #####
# 1) No buying of more than 10 units of a company's stock per day
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
import random
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

if torch.cuda.is_available():
    print("Testing on GPU...")
else:
    print("No GPU found. Training on CPU...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set MODEL_LOAD_NAME to a specific name to load a specific model or set it to None to load the newest trained model
MODEL_LOAD_NAME = "train80_windowsize25_epochs20_batchsize1_hiddensize200_lr0.0005_smoothing2_smoothingsize26_avgloss0.0145763301583338" #"train80_windowsize25_epochs20_batchsize1_hiddensize200_lr0.0005_smoothing2_smoothingsize26_avgloss0.0145763301583338"
if MODEL_LOAD_NAME is None:
    list_of_files = glob.glob(os.path.join("models", "*")) # select all models
    latest_file = max(list_of_files, key = os.path.getctime) # sort them by newest date and pick the most recent one
    MODEL_LOAD_NAME = os.path.split(latest_file)[1] # convert 'models/model_name' to 'model_name'
MODEL_HIDDEN_SIZE = int(MODEL_LOAD_NAME.split("_")[4][10:]) # convert "train{}_windowsize{}_epochs{}_batchsize{}_hiddensize{}_lr{}" to "hiddensize{}" to "{}" and then to an int

print("Loading model {} ...".format(MODEL_LOAD_NAME))

# set network hyperparameters
TRAIN = 0.8
WINDOW_SIZE = 25
SMA_OR_EMA = 2 # 0 = use Simple Moving Average, 1 = use Exponential Moving Average, any other number = else don't use either SMA or EMA
SMOOTHING_WINDOW_SIZE = 25 # 12-day = 30% RoI, 26-day = 10% RoI, 50-day = 5.5% RoI, 100-day = 5% RoI
INITIAL_MONEY = 10000
TRADING_START_DATE = "2012" # begin trading at this date or later (e.g. "2015", "2007-01", "2007-01-01", "YYYY-MM-DD"); None => start date is the earliest start date across all stocks

# set up model
model = StockPredictor(hidden_size = MODEL_HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load(os.path.join("models", MODEL_LOAD_NAME)))

# determine which OOD stocks to use
all_stock_fns = [os.path.split(path)[1] for path in glob.glob(os.path.join("data", "Stocks", "*"))]
random.shuffle(all_stock_fns)
ood_stock_fns = all_stock_fns[0:1000] #["acbi.us.txt", "googl.us.txt", "goex.us.txt", "goro.us.txt", "lea.us.txt", "tsla.us.txt"] #["acbi.us.txt", "googl.us.txt", "jpm.us.txt"] #["acbi.us.txt", "googl.us.txt"] #["acbi.us.txt", "googl.us.txt", "jpm.us.txt"] # ["acbi.us.txt"]
print("Stocks traded: {}".format(ood_stock_fns))

# preprocess the dataset and set up a stock market so we can pull prices on a daily basis
market = StockMarket(stock_fns = ood_stock_fns, window_size = WINDOW_SIZE, sma_or_ema = SMA_OR_EMA, smoothing_window_size = SMOOTHING_WINDOW_SIZE, trading_start_date = TRADING_START_DATE)
agent = BuyerSeller(initial_money = INITIAL_MONEY, market = market)

# test the model
curr_prices = None
curr_untransformed_prices = None
stock_scalers = market.get_stock_scalers()
for day_number in range(len(market)): # iterate over batches
    date, day_prices = market.get_iteration_prices(day_number) # get the price of each stock for today

    # store the predicted and ground-truth prices for each stock on this day
    pred_prices = []
    true_prices = []
    true_untransformed_prices = []

    for stock_ticker in day_prices.index: # the .index attribute for a Series is equivalent to .columns for a DataFrame
        pred = 0
        label = 0
        label_untransformed = 0
        if not pd.isna(day_prices[stock_ticker]): # detect if NaN value (the stock has not been opened yet by this date)
            prices, label, label_untransformed = day_prices[stock_ticker] # input prices and ground-truth price prediction

            prices = torch.FloatTensor(prices).unsqueeze(0) # convert to tensor so we can pass it into the model
            pred = model(prices.to(device)).item() # predicted price

        pred_prices.append(pred)
        true_prices.append(label)
        true_untransformed_prices.append(label_untransformed)

    if curr_prices is None: # will only be true on the first iteration
        curr_prices = [0] * len(pred_prices)
        curr_untransformed_prices = [0] * len(pred_prices)
    stock_analysis_model = pd.DataFrame([curr_prices, pred_prices, curr_untransformed_prices], columns = day_prices.index)

    # buying/selling logic
    stocks_bought, stocks_sold, num_active_stocks = agent.buy_sell_or_stay(stock_analysis_model)
    if day_number == 0 or (day_number+1) % 100 == 0:
        print("Date: {}, Day {}/{} -- # active stocks: {}, # bought: {}, # sold: {}, Curr money: {}".format( \
                date, day_number + 1, len(market), num_active_stocks, len(stocks_bought), len(stocks_sold), \
                round(agent.get_curr_investment_money(), 2)))

    # prepare for next iteration
    curr_prices = true_prices
    curr_untransformed_prices = true_untransformed_prices

agent.sell_all() # sell all stocks
profit_loss_amount = agent.get_total_money() - INITIAL_MONEY # amount of money we gained or lost at the end of the simulation
roi_percent = (profit_loss_amount / INITIAL_MONEY) * 100 # compared to the initial investment, how much of the initial investment did we make in profit/loss?
print("Final total money: {}, Final savings: {}, Profit/Loss: {}, RoI%: {}".format(round(agent.get_total_money(), 4), round(agent.get_curr_savings(), 4), round(profit_loss_amount, 4), round(roi_percent, 4)))
