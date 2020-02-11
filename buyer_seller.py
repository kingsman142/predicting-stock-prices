import numpy as np

class BuyerSeller():
    def __init__(self, initial_money = 0):
        self.stock_counts = {} # number of stocks owned per company
        self.stock_prices = {} # price of each stock
        self.money = initial_money # amount of current money
        self.savings = 0
        self.savings_percent = 0.0001

        self.STOCK_TOTAL_BUY_CAP = 10000 # maximum number of stocks you can own of an individual company at a given time
        self.stock_buy_minimum = 1 # minimum number of stocks you can buy of a company at a given time

    # decide whether to buy, sell, or stay
    def buy_sell_or_stay(self, stock_analysis_model):
        stock_tickers = stock_analysis_model.columns

        # keep track of which stocks have a positive margin, then later sort then based on margin and buy as much of each one as possible
        stocks_to_buy = []

        # keep track of which stocks we bought, sold, and stayed today
        stocks_bought = []
        stocks_sold = []
        num_active_stocks = 0

        for stock_ticker in stock_tickers:
            curr_price, pred = stock_analysis_model[stock_ticker] # extract the current price and predicted price of this stock from the analysis model
            self.stock_prices[stock_ticker] = curr_price # update the price of this stock
            num_active_stocks += 1 if curr_price > 0 else 0 # if this stock's price is greater than 0, then we can buy it
            if curr_price == 0:
                continue

            margin = pred - curr_price # how much do the predicted price and curr_price differ

            # main logic of whether to buy or sell
            if margin > 0: # we should buy
                buy_amount = self.determine_buy_amount(stock_ticker, curr_price, pred)
                if buy_amount > 0:
                    stocks_to_buy.append([stock_ticker, margin, buy_amount])
            elif margin < 0: # we should sell
                sell_amount = self.determine_sell_amount(stock_ticker, curr_price, pred)
                stocks_sold.append((stock_ticker, sell_amount))

        # now that we've found the one stock with the best margin, buy as much of it as possible
        stocks_to_buy.sort(key = lambda stock_info : stock_info[1], reverse = True) # sort by margin in descending order
        for stock in stocks_to_buy:
            stock_ticker, buy_amount = stock[0], stock[2]

            buy_successful = self.make_purchase(stock_ticker, buy_amount) # return whether this purchase was successful
            if not buy_successful and self.money < 0.0001: # we can't buy anymore stocks
                break
            elif buy_successful:
                stocks_bought.append((stock_ticker, buy_amount))

        return stocks_bought, stocks_sold, num_active_stocks

    # decide whether to buy and how much
    def determine_buy_amount(self, stock_ticker, curr_price, pred):
        if curr_price == 0:
            return 0

        threshold = 0.0001 # 0.0001 OR 0.0004 for buy
        if (pred / curr_price) >= 1.0001:
            amount = min(self.get_stock_buy_cap(curr_price), int(self.money / curr_price)) # int(self.money / curr_price) -- this approach produces about a 0.5% smaller RoI
            amount = max(amount, self.stock_buy_minimum) # we have to buy at least self.stock_buy_minimum stocks at a given time, this helps to prevent buying 1e-10 stocks
            if self.money > (curr_price * amount) and amount > 0 and (self.stock_counts.get(stock_ticker, 0) + amount) < self.STOCK_TOTAL_BUY_CAP: # cap out at N stocks of this company
                return amount
            return 0
        return 0

    # decide whether to sell and how much
    def determine_sell_amount(self, stock_ticker, curr_price, pred):
        if curr_price == 0:
            return 0

        threshold = -0.002 # -0.002 OR -0.0025 for sell
        if (pred / curr_price) <= 0.998 and stock_ticker in self.stock_counts and self.stock_counts[stock_ticker] > 0:
            sell_amount = self.stock_counts[stock_ticker]
            self.money += self.stock_counts[stock_ticker] * curr_price
            #print("SELLING {} of {} at price {}, new money at {}".format(sell_amount, stock_ticker, curr_price, self.money))

            self.savings += self.stock_counts[stock_ticker] * curr_price * self.savings_percent
            self.money -= self.stock_counts[stock_ticker] * curr_price * self.savings_percent

            self.stock_counts[stock_ticker] = 0
            return sell_amount # sell all stocks
        return 0

    def make_purchase(self, stock_ticker, amount):
        curr_price = self.stock_prices[stock_ticker]
        if curr_price < 1e-10 or self.money < 1e-10:
            return False

        tmp = self.get_stock_buy_cap(curr_price)
        new_amount = min(self.get_stock_buy_cap(curr_price), self.money / curr_price - 1e-10) # calibrate, because a previously bought stock might make this stock's amount decrease
        new_amount = max(new_amount, self.stock_buy_minimum) # we have to buy at least self.stock_buy_minimum stocks at a given time, this helps to prevent buying 1e-10 stocks
        old_amount = amount
        amount = int(min(new_amount, amount))
        total_cost = curr_price * amount

        if self.money >= total_cost and amount > 0:
            self.money -= total_cost
            if stock_ticker in self.stock_counts:
                self.stock_counts[stock_ticker] += amount
            else:
                self.stock_counts[stock_ticker] = amount

            if self.money < 1e-10:
                self.money = 0.0

            #print("BUYING {} OF {} AT {}, with total cost {} and new money {} -- {}, {}, {}".format(amount, stock_ticker, curr_price, total_cost, self.money, tmp, self.money / curr_price - 1e-10, old_amount))
            return True
        return False

    def sell_all(self):
        for stock_ticker, num_stocks in self.stock_counts.items():
            self.money += num_stocks * self.stock_prices[stock_ticker]

    def get_curr_investment_money(self):
        return self.money

    def get_curr_savings(self):
        return self.savings

    def get_stock_counts(self):
        return self.stock_counts

    def get_stock_prices(self):
        return self.stock_prices

    def get_total_money(self):
        return self.money + self.savings

    # can only spend X% of your money on each stock per day
    # maximum amount of a stock you can buy on a given day for a company
    def get_stock_buy_cap(self, curr_price):
        return int(self.money / curr_price)
