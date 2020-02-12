import numpy as np

class BuyerSeller():
    def __init__(self, market, initial_money = 0):
        self.stock_counts = {} # number of stocks owned per company
        self.stock_prices = {} # price of each stock
        self.money = initial_money # amount of current money
        self.savings = 0
        self.savings_percent = 0.0 # 0.0001
        self.market = market # the stock market

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
            curr_price, pred, curr_price_untransformed = stock_analysis_model[stock_ticker] # extract the current price and predicted price of this stock from the analysis model
            if curr_price_untransformed == 0:
                continue

            curr_price = self.unscale_stock_price(stock_ticker, curr_price)
            pred = self.unscale_stock_price(stock_ticker, pred)

            self.stock_prices[stock_ticker] = curr_price_untransformed # update the price of this stock
            num_active_stocks += 1 if self.stock_prices[stock_ticker] > 0 else 0 # if this stock's price is greater than 0, then we can buy it

            margin = pred / curr_price # how much do the predicted price and curr_price differ

            # main logic of whether to buy or sell
            if margin > 1: # we should buy
                buy_amount = self.determine_buy_amount(stock_ticker, margin, curr_price_untransformed)
                if buy_amount > 0:
                    stocks_to_buy.append([stock_ticker, margin, buy_amount])
            elif margin < 1: # we should sell
                sell_amount = self.determine_sell_amount(stock_ticker, margin, curr_price_untransformed)
                if sell_amount > 0:
                    self.make_sale(stock_ticker, sell_amount)
                stocks_sold.append((stock_ticker, sell_amount))

        # now that we've found the one stock with the best margin, buy as much of it as possible
        stocks_to_buy.sort(key = lambda stock_info : (stock_info[1]*stock_info[2]), reverse = True) # sort by margin in descending order
        for stock in stocks_to_buy:
            stock_ticker, buy_amount = stock[0], stock[2]

            buy_successful = self.make_purchase(stock_ticker, buy_amount) # return whether this purchase was successful
            if buy_successful:
                stocks_bought.append((stock_ticker, buy_amount))
            elif not buy_successful and self.money < 0.0001: # we can't buy anymore stocks
                break

        return stocks_bought, stocks_sold, num_active_stocks

    # decide whether to buy and how much
    def determine_buy_amount(self, stock_ticker, margin, curr_price_untransformed):
        if curr_price_untransformed == 0:
            return 0

        threshold = 0.0001 # 0.0001 OR 0.0004 for buy
        if margin >= 1.0002:
            amount = max(self.get_stock_buy_cap(self.stock_prices[stock_ticker]), self.stock_buy_minimum) # we have to buy at least self.stock_buy_minimum stocks at a given time, this helps to prevent buying 1e-10 stocks
            if self.money > (self.stock_prices[stock_ticker] * amount) and amount > 0 and (self.stock_counts.get(stock_ticker, 0) + amount) < self.STOCK_TOTAL_BUY_CAP: # cap out at N stocks of this company
                return amount
            return 0
        return 0

    # decide whether to sell and how much
    def determine_sell_amount(self, stock_ticker, margin, curr_price_untransformed):
        if curr_price_untransformed == 0:
            return 0

        threshold = -0.002 # -0.002 OR -0.0025 for sell
        if margin <= 0.994 and stock_ticker in self.stock_counts and self.stock_counts[stock_ticker] > 0:
            sell_amount = self.stock_counts[stock_ticker]
            return sell_amount # sell all stocks
        return 0

    def make_purchase(self, stock_ticker, amount):
        curr_price = self.stock_prices[stock_ticker]
        if curr_price < 1e-10 or self.money < 1e-10:
            return False

        new_amount = max(self.get_stock_buy_cap(curr_price) , self.stock_buy_minimum) # we have to buy at least self.stock_buy_minimum stocks at a given time, this helps to prevent buying 1e-10 stocks
        amount = int(min(new_amount, amount)) # calibrate, because a previously bought stock might make this stock's amount decrease
        total_cost = curr_price * amount

        if self.money >= total_cost and amount > 0:
            self.money -= total_cost
            if stock_ticker in self.stock_counts:
                self.stock_counts[stock_ticker] += amount
            else:
                self.stock_counts[stock_ticker] = amount

            if self.money < 1e-2:
                self.money = 0.0

            #print("BUYING {} OF {} AT {}, with total cost {} and new money {} -- {}, {}, {}".format(amount, stock_ticker, curr_price, total_cost, self.money, tmp, self.money / curr_price - 1e-10, old_amount))
            return True
        return False

    def make_sale(self, stock_ticker, amount):
        self.money += amount * self.stock_prices[stock_ticker]
        #print("SELLING {} of {} at price {}, new money at {}, curr_price_untransformed: {}".format(sell_amount, stock_ticker, curr_price, self.money, curr_price_untransformed))

        self.savings += amount * self.stock_prices[stock_ticker] * self.savings_percent
        self.money -= amount * self.stock_prices[stock_ticker] * self.savings_percent

        self.stock_counts[stock_ticker] -= amount

    def sell_all(self):
        for stock_ticker, num_stocks in self.stock_counts.items():
            self.money += num_stocks * self.stock_prices[stock_ticker]

    def unscale_stock_price(self, stock_ticker, scaled_price):
        stock_scaler = self.market.get_stock_scalers()[stock_ticker]
        scaled_price = np.array(scaled_price).reshape(-1, 1)
        unscaled_price = stock_scaler.inverse_transform(scaled_price)
        return unscaled_price[0][0]

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
