import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 200, output_size = 1):
        super(StockPredictor, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, prices):
        batch_size = len(prices)
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_size),
                             torch.zeros(1, batch_size, self.hidden_size))
        new_prices = prices.permute(1, 0).unsqueeze(-1)

        # NOTE: pytorch LSTM units take input in the form of [window_length, batch_size, num_features], which will end up being [WINDOW_SIZE, batch_size, 1] for our dataset
        lstm_out, self.hidden_cell = self.lstm(new_prices, self.hidden_cell)
        pred = self.linear(lstm_out[-1])
        return pred[-1]
