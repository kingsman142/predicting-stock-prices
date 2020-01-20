import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import StockPredictor
from dataset import StockDataset

TRAIN = 0.8
TEST = 0.2
WINDOW_SIZE = 250
EPOCHS = 10
BATCH_SIZE = 4

model = StockPredictor()
dataset = StockDataset("aa.us.txt")
#dataset.plot_stock()

optimizer = optim.Adam(model.parameters(), lr = 0.0001)
loss_func = nn.L1Loss(reduction = 'mean') #nn.MSELoss(reduction = 'sum')

loader = data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

for epoch in range(EPOCHS):
    for batch_id, samples in enumerate(loader):
        prices = samples['prices']
        labels = samples['labels']

        pred = model(prices)

        loss = loss_func(pred, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 50 == 0:
            if BATCH_SIZE == 1:
                print("Epoch {}/{} -- Batch {}/{} -- Loss: {} -- Pred: {}, True: {}".format(epoch+1, EPOCHS, batch_id+1, len(loader), loss.item(), pred.item(), labels.item()))
            else:
                print("Epoch {}/{} -- Batch {}/{} -- Loss: {}".format(epoch+1, EPOCHS, batch_id+1, len(loader), loss.item()))

# NOTE: pytorch LSTM units take input in the form of [window_length, batch_size, num_features], which will end up being [WINDOW_SIZE, batch_size, 1] for our dataset
