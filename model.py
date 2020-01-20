import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 200, output_size = 1):
        super(StockPredictor, self).__init__()

        '''self.model = nn.Sequential(
            nn.LSTM(input_size = 1, hidden_size = 200, num_layers = 1, bias = True, dropout = 0.0),
            nn.ReLU(),
            nn.LSTM(input_size = 200, hidden_size = 200, num_layers = 1, bias = True, dropout = 0.0),
            nn.ReLU(),
            nn.LSTM(input_size = 200, hidden_size = 100, num_layers = 1, bias = True, dropout = 0.0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = 100, out_features = 1, bias = True)
        )''' # consider removing two the LSTM lines, I don't thinkw e need them

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, prices):
        batch_size = len(prices)
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_size),
                             torch.zeros(1, batch_size, self.hidden_size))
        new_prices = prices.permute(1, 0).unsqueeze(-1)

        #print("(Before LSTM) Prices shape: {}, Prices new shape: {}".format(prices.size(), new_prices.size()))
        lstm_out, self.hidden_cell = self.lstm(new_prices, self.hidden_cell)
        #print("(After LSTM) Prices shape: {}".format(lstm_out[-1].size()))
        pred = self.linear(lstm_out[-1]) # self.linear(lstm_ut[:, -1, :]) TRY SWAPPING THIS BIT OF CODE IN INSTEAD IF THE CURRENT CODE DOESN'T WORK
        # maybe consider doing the loop approach where we iterate over each element in the window
        print("Pred shape: {}".format(pred.size()))
        return pred[-1]

        '''print("(Before LSTM) Prices shape: {}, Prices new shape: {}".format(prices.size(), prices.view(len(prices), 1, -1).size()))
        lstm_out, self.hidden_cell = self.lstm(prices.view(len(prices), 1, -1), self.hidden_cell)
        print("(After LSTM) Prices shape: {}".format(lstm_out.view(len(prices), -1).size()))
        pred = self.linear(lstm_out.view(len(prices), -1)) # self.linear(lstm_ut[:, -1, :]) TRY SWAPPING THIS BIT OF CODE IN INSTEAD IF THE CURRENT CODE DOESN'T WORK
        # maybe consider doing the loop approach where we iterate over each element in the window
        print("Pred shape: {}".format(pred.size()))
        return pred[-1]'''
