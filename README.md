# predicting-stock-prices
Using an LSTM to predict stock prices

0. Download data from https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data and place it in a data/ folder in the project root directory
1. Run train.py --stocks [stock_name]

TODO:
* Implement exponential moving average (EMA)
* Investigate common fault signals in stock prediction ( https://www.investopedia.com/terms/f/false-signal.asp and https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp)
* Make a small trading bot and show applicability
* Maybe try SGD optimizer over Adam optimizer

Possible analysis:
* Plot effect of simple moving average
* Plot effect of exponential moving average
* Show results on predicting of one stock
* Show results on predicting of multiple stocks
* Investigate use of L1 vs. MSE loss
* Write study based around performance of different window sizes (or even hidden state sizes, batch sizes, etc.)
* Investigate whether there's a difference in fluctuations/spikes between popular stocks and non-popular stocks
