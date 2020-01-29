import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import StockPredictor
from dataset import StockDataset

TRAIN = 0.8 #0.0209
TEST = 0.2
WINDOW_SIZE = 50 #250 #250
EPOCHS = 20
BATCH_SIZE = 1 #4
HIDDEN_SIZE = 150 #200
LEARNING_RATE = 0.001 #0.0001

# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, no moving average => 0.005 L1 loss ('sum' reduction)
# train = 0.8, window = 50, epochs = 20, batch size = 1, hidden size = 100, lr = 0.001, simple moving average => 0.0022 L1 loss ('sum' reduction)

model = StockPredictor(hidden_size = HIDDEN_SIZE)
dataset = StockDataset(stock_fns = "aa.us.txt", window_size = WINDOW_SIZE, train = TRAIN)
#dataset.plot_stock_raw()
#print(len(dataset))

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss_func = nn.L1Loss(reduction = 'mean') #nn.MSELoss(reduction = 'sum')

loader = data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

for epoch in range(EPOCHS):
    avg_loss = 0.0
    for batch_id, samples in enumerate(loader):
        prices = samples['prices']
        labels = samples['labels']

        pred = model(prices)

        loss = loss_func(pred, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        if batch_id % 50 == 0:
            if BATCH_SIZE == 1:
                print("Epoch {}/{} -- Batch {}/{} -- Loss: {} -- Pred: {}, True: {}".format(epoch+1, EPOCHS, batch_id+1, len(loader), loss.item(), pred.item(), labels.item()))
            else:
                print("Epoch {}/{} -- Batch {}/{} -- Loss: {}".format(epoch+1, EPOCHS, batch_id+1, len(loader), loss.item()))
    avg_loss /= len(loader)
    print("--- Epoch {} avg loss: {}".format(epoch+1, avg_loss))

# NOTE: pytorch LSTM units take input in the form of [window_length, batch_size, num_features], which will end up being [WINDOW_SIZE, batch_size, 1] for our dataset
