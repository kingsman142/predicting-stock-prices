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
ood_stock_fns = ["acbi.us.txt"] # ["goog.us.txt"]

# preprocess the dataset
preprocessor = StockPreprocessor(stock_fns = ood_stock_fns, window_size = WINDOW_SIZE, train = TRAIN, sma_or_ema = SMA_OR_EMA, smoothing_window_size = SMOOTHING_WINDOW_SIZE)
stock_windows = preprocessor.get_all_data()
dataset = StockDataset(stock_windows = stock_windows)

# set up hyperparameters
loss_func = nn.L1Loss(reduction = 'mean')
loader = data.DataLoader(dataset, batch_size = 1, shuffle = False)

# test the model
agent = BuyerSeller(initial_money = INITIAL_MONEY)
curr_price = 0
fluctuation_correct = 0
buy_correct = 0
buy_total = 0
sell_correct = 0
sell_total = 0
increase_correct = 0
increase_total = 0
decrease_correct = 0
decrease_total = 0
for batch_id, samples in enumerate(loader): # iterate over batches
    # input prices and ground-truth price prediction
    prices = samples['prices']
    label = samples['labels'].item()

    # make predictions and calculate loss
    pred = model(prices).item()

    # buying/selling logic
    action, amount = agent.buy_sell_or_stay(ood_stock_fns[0], curr_price, pred)
    print("Iteration {}/{} -- Curr price: {}, Pred price: {}, GT price: {} -- Action: {}, Amount: {}, Curr money: {}".format(batch_id+1, len(loader), round(curr_price, 4), round(pred, 4), round(label, 4), action, amount, round(agent.get_curr_investment_money(), 2)))

    # update accuracy counts to print out some statistics at the end
    if (pred > curr_price and  label > curr_price) or (pred < curr_price and label < curr_price):
        fluctuation_correct += 1

    if label > curr_price:
        increase_correct += 1 if (pred > curr_price and amount > 0) else 0
        increase_total += 1
    elif label < curr_price:
        decrease_correct += 1 if (pred < curr_price and amount > 0) else 0
        decrease_total += 1

    if pred > curr_price and amount > 0:
        buy_correct += 1 if label > curr_price else 0
        buy_total += 1
    elif pred < curr_price and amount > 0:
        sell_correct += 1 if label < curr_price else 0
        sell_total += 1

    # prepare for next iteration
    curr_price = label

fluctuation_accuracy = fluctuation_correct / len(loader)
buy_accuracy = buy_correct / buy_total
sell_accuracy = sell_correct / sell_total
increase_accuracy = increase_correct / increase_total
decrease_accuracy = decrease_correct / decrease_total

agent.sell_all() # sell all stocks
profit_loss_amount = agent.get_total_money() - INITIAL_MONEY # amount of money we gained or lost at the end of the simulation
roi_percent = profit_loss_amount / INITIAL_MONEY # compared to the initial investment, how much of the initial investment did we make in profit/loss?
print("Final total money: {}, Final savings: {}, Profit/Loss: {}, RoI%: {}".format(round(agent.get_total_money(), 4), round(agent.get_curr_savings(), 4), round(profit_loss_amount, 4), round(roi_percent, 4)))
print("Fluctuation accuracy: {}%".format(round(fluctuation_accuracy * 100.0, 2))) # percent of the time we predicted correctly whether the stock would increase or decrease
print("Buy accuracy: {}%".format(round(buy_accuracy * 100.0, 2))) # when we bought, how many times did the stock actually increase the next day?
print("Sell accuracy: {}%".format(round(sell_accuracy * 100.0, 2))) # when we sold, how many times did the stock actually decrease the next day?
print("Increase accuracy: {}%".format(round(increase_accuracy * 100.0, 2))) # when the stock increased, how many times did we predict that correctly and actually approve a buy?
print("Decrease accuracy: {}%".format(round(decrease_accuracy * 100.0, 2))) # when the stock decreased, how many times did we predict that correctly and actually approve a sell?
