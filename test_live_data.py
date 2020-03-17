import os
import glob
import torch
import requests
import numpy as np

from model import StockPredictor

# set MODEL_LOAD_NAME to a specific name to load a specific model or set it to None to load the newest trained model
MODEL_LOAD_NAME = None # "train80_windowsize50_epochs1_batchsize1_hiddensize150_lr0.001_smoothing1_smoothingsize50"
if MODEL_LOAD_NAME is None:
    list_of_files = glob.glob(os.path.join("models", "*")) # select all models
    latest_file = max(list_of_files, key = os.path.getctime) # sort them by newest date and pick the most recent one
    MODEL_LOAD_NAME = os.path.split(latest_file)[1] # convert 'models/model_name' to 'model_name'
MODEL_HIDDEN_SIZE = int(MODEL_LOAD_NAME.split("_")[4][10:]) # convert "train{}_windowsize{}_epochs{}_batchsize{}_hiddensize{}_lr{}" to "hiddensize{}" to "{}" and then to an int

model = StockPredictor(hidden_size = MODEL_HIDDEN_SIZE)
model.load_state_dict(torch.load(os.path.join("models", MODEL_LOAD_NAME)))
