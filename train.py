import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import StockPredictor
from dataset import StockDataset

# set network hyperparameters
TRAIN = 0.8 #0.0209
TEST = 0.2
WINDOW_SIZE = 50 #250 #250
EPOCHS = 20
BATCH_SIZE = 1 #4
HIDDEN_SIZE = 150 #200
LEARNING_RATE = 0.001 #0.0001

MODEL_LOAD_NAME = None # change to load in a custom model
MODEL_SAVE_NAME = "train{}_windowsize{}_epochs{}_batchsize{}_hiddensize{}_lr{}".format(int(TRAIN*100), WINDOW_SIZE, EPOCHS, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE)

# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, no moving average, 1 stock => 0.005 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average, 1 stock => 0.0022 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average, 2 stocks => 0.0021 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average, 6 stocks => 0.0013 L1 loss ('sum' reduction)

# set up dataset and model
stock_fns = ["aa.us.txt", "msft.us.txt", "goog.us.txt", "gpic.us.txt", "rfdi.us.txt", "aal.us.txt"] # chosen somewhat randomly
model = StockPredictor(hidden_size = HIDDEN_SIZE)
dataset = StockDataset(stock_fns = stock_fns, window_size = WINDOW_SIZE, train = TRAIN)

# (OPTIONAL) uncomment to plot the stock data -- NOTE: only plots the first stock's history if a list of stocks is provided
#dataset.plot_stock_raw()

# load pre-trained model weights
if MODEL_LOAD_NAME is not None:
    model.load_state_dict(torch.load(os.path.join("models", MODEL_LOAD_NAME)))

# set up hyperparameters
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss_func = nn.L1Loss(reduction = 'mean') #nn.MSELoss(reduction = 'sum')
loader = data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# train the model
for epoch in range(EPOCHS): # iterate over epochs
    avg_loss = 0.0
    for batch_id, samples in enumerate(loader): # iterate over batches
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
                print("Epoch {}/{} -- Batch {}/{} -- Loss: {} -- Pred: {}, True: {}".format(epoch+1, EPOCHS, batch_id+1, len(loader), loss.item(), pred.item(), labels.item()))
            else:
                print("Epoch {}/{} -- Batch {}/{} -- Loss: {}".format(epoch+1, EPOCHS, batch_id+1, len(loader), loss.item()))
    avg_loss /= len(loader)
    print("--- Epoch {} avg loss: {}".format(epoch+1, avg_loss))

# save our model
if not os.path.exists("models"):
    os.mkdir("models")
torch.save(model.state_dict(), "models/{}".format(MODEL_SAVE_NAME))
