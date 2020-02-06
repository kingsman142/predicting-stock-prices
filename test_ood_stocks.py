import os
import glob
import torch
import torch.nn as nn
import torch.utils.data as data

from model import StockPredictor
from preprocess import StockPreprocessor
from dataset import StockDataset
from utils import plot_predictions

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
SMOOTHING_WINDOW_SIZE = 50

# set up model
model = StockPredictor(hidden_size = MODEL_HIDDEN_SIZE)
model.load_state_dict(torch.load(os.path.join("models", MODEL_LOAD_NAME)))

# determine which OOD stocks to use
ood_stock_fns = ["acbi.us.txt", "hscz.us.txt", "qvcb.us.txt", "qsr.us.txt"] # ["acbi.us.txt"]

# preprocess the dataset
stock_windows = StockPreprocessor(stock_fns = ood_stock_fns, window_size = WINDOW_SIZE, train = TRAIN, sma_or_ema = SMA_OR_EMA, smoothing_window_size = SMOOTHING_WINDOW_SIZE).get_all_data()
dataset = StockDataset(stock_windows = stock_windows)

# set up hyperparameters
loss_func = nn.L1Loss(reduction = 'mean')
loader = data.DataLoader(dataset, batch_size = 1, shuffle = False)

# test the model
avg_loss = 0.0
predictions = []
ground_truth = []
for batch_id, samples in enumerate(loader): # iterate over batches
    # input prices and ground-truth price prediction
    prices = samples['prices']
    labels = samples['labels']

    # make predictions and calculate loss
    pred = model(prices)
    loss = loss_func(pred, labels).item()
    predictions.append(pred.item())
    ground_truth.append(labels.item())

    # print out useful logging information for user
    avg_loss += loss
    if batch_id % 50 == 0:
        print("(test) Batch {}/{} -- Loss: {} -- Pred: {}, True: {}".format(batch_id+1, len(loader), loss, pred.item(), labels.item()))
avg_loss /= len(loader)
print("(test) avg loss: {}".format(avg_loss))

if len(ood_stock_fns) == 1: # only have to plot one stock's predictions
    plot_predictions(ground_truth, predictions)
