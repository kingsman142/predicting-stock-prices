# predicting-stock-prices
Using an LSTM to predict stock prices

0. Download data from https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data and place it in a data/ folder in the project root directory such that you have a "data/Data", "data/ETFs", and "data/Stocks" folder
1. Run `train.py`
2. (optional) Run `test_ood_stocks.py` to test your trained model on OOD (out-of-distribution) stocks, which are stocks your model was not trained on
3. (optional) Run `test_live_data.py` to pull live prices off of a stock exchange site and compare the success of the model on a daily basis
4. (optional) Run `test_buy_seller.py` to simulate a buying and selling bot in an auction house

TODO:
* Investigate common fault signals in stock prediction ( https://www.investopedia.com/terms/f/false-signal.asp and https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp)
* Make a small trading bot and show applicability

Possible analysis:
* Plot effect of simple moving average
* Plot effect of exponential moving average
* Show results on predicting of one stock
* Show results on predicting of multiple stocks
* Investigate use of L1 vs. MSE loss
* Write study based around performance of different window sizes (or even hidden state sizes, batch sizes, etc.)
* Investigate whether there's a difference in fluctuations/spikes between popular stocks and non-popular stocks
