import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import StockPredictor
from dataset import StockDataset
from preprocess import StockPreprocessor

# set network hyperparameters
TRAIN = 0.8 #0.0209
TEST = 0.2
WINDOW_SIZE = 50 #250 #250
EPOCHS = 20
BATCH_SIZE = 1 #4
HIDDEN_SIZE = 150 #200
LEARNING_RATE = 0.001 #0.0001
SMA_OR_EMA = 2 # 0 = use Simple Moving Average, 1 = use Exponential Moving Average, any other number = else don't use either SMA or EMA
SMOOTHING_WINDOW_SIZE = 26

MODEL_LOAD_NAME = None # change to load in a custom model
MODEL_SAVE_NAME = "train{}_windowsize{}_epochs{}_batchsize{}_hiddensize{}_lr{}_smoothing{}_smoothingsize{}".format(int(TRAIN*100), WINDOW_SIZE, EPOCHS, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, SMA_OR_EMA, SMOOTHING_WINDOW_SIZE)

# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, no moving average, 1 stock => 0.005 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average, smoothing window = 50, 1 stock => 0.0022 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average, smoothing window = 50, 2 stocks => 0.0021 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average, smoothing window = 50, 6 stocks => 0.0013 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, exponential moving average (gamma = 0.1), smoothing window = 50, 6 stocks => 0.0019 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, exponential moving average (gamma = 0.0392), smoothing window = 50, 6 stocks => 0.0014 L1 train loss, 0.0024 test loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, exponential moving average (gamma = 0.2), smoothing window = 50, 6 stocks, SGD => 0.0051 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, exponential moving average (gamma = standard), smoothing window = 26, 6 stocks => 0.0015 L1 train loss, 0.0016 test loss ('sum' reduction)

# set up dataset and model
stock_fns = ["aa.us.txt", "msft.us.txt", "goog.us.txt", "gpic.us.txt", "rfdi.us.txt", "aal.us.txt"] # chosen somewhat randomly
model = StockPredictor(hidden_size = HIDDEN_SIZE)
train_windows, test_windows = StockPreprocessor(stock_fns = stock_fns, window_size = WINDOW_SIZE, train = TRAIN, sma_or_ema = SMA_OR_EMA, smoothing_window_size = SMOOTHING_WINDOW_SIZE).get_splits()
train_dataset = StockDataset(stock_windows = train_windows)
test_dataset = StockDataset(stock_windows = test_windows)

# (OPTIONAL) uncomment to plot the stock data -- NOTE: only plots the first stock's history if a list of stocks is provided
#train_dataset.plot_stock_raw()
#test_dataset.plot_stock_raw()

# load pre-trained model weights
if MODEL_LOAD_NAME is not None:
    model.load_state_dict(torch.load(os.path.join("models", MODEL_LOAD_NAME)))

# set up hyperparameters
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss_func = nn.L1Loss(reduction = 'mean') #nn.MSELoss(reduction = 'sum')
train_loader = data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = True)

# train the model
for epoch in range(EPOCHS): # iterate over epochs
    avg_loss = 0.0
    for batch_id, samples in enumerate(train_loader): # iterate over batches
        # input prices and ground-truth price prediction
        prices = samples['prices']
        labels = samples['labels']

        # make predictions and calculate loss
        pred = model(prices)
        loss = loss_func(pred, labels)

        # backpropagate loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # print out useful logging information for user
        avg_loss += loss.item()
        if batch_id % 50 == 0:
            if BATCH_SIZE == 1:
                print("(train) Epoch {}/{} -- Batch {}/{} -- Loss: {} -- Pred: {}, True: {}".format(epoch+1, EPOCHS, batch_id+1, len(train_loader), loss.item(), pred.item(), labels.item()))
            else:
                print("(train) Epoch {}/{} -- Batch {}/{} -- Loss: {}".format(epoch+1, EPOCHS, batch_id+1, len(train_loader), loss.item()))
    avg_loss /= len(train_loader)
    print("--- Epoch {} avg loss: {}".format(epoch+1, avg_loss))

# test the model
avg_loss = 0.0
for batch_id, samples in enumerate(test_loader): # iterate over batches
    # input prices and ground-truth price prediction
    prices = samples['prices']
    labels = samples['labels']

    # make predictions and calculate loss
    pred = model(prices)
    loss = loss_func(pred, labels)

    # print out useful logging information for user
    avg_loss += loss.item()
    if batch_id % 50 == 0:
        print("(test) Batch {}/{} -- Loss: {} -- Pred: {}, True: {}".format(batch_id+1, len(test_loader), loss.item(), pred.item(), labels.item()))
avg_loss /= len(test_loader)
print("(test) avg loss: {}".format(avg_loss))

# save our model
if not os.path.exists("models"):
    os.mkdir("models")
torch.save(model.state_dict(), "models/{}".format(MODEL_SAVE_NAME))
