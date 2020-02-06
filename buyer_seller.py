# TODO:
# attach stock name to its price/label pairing from the dataloader

class BuyerSeller():
    def __init__(self, initial_money = 0):
        self.stock_counts = {} # number of stocks owned per company
        self.stock_prices = {} # price of each stock
        self.money = initial_money # amount of current money
        self.savings = 0
        self.savings_percent = 0 #0.01

    # decide whether to buy, sell, or stay
    def buy_sell_or_stay(self, stock_ticker, curr_price, pred):
        self.stock_prices[stock_ticker] = curr_price
        if pred > curr_price:
            buy_amount = self.determine_buy_amount(stock_ticker, curr_price, pred)
            return "buy", buy_amount
        elif pred < curr_price:
            sell_amount = self.determine_sell_amount(stock_ticker, curr_price, pred)
            return "sell", sell_amount
        else:
            return "stay", None

    # decide whether to buy and how much
    def determine_buy_amount(self, stock_ticker, curr_price, pred):
        if curr_price == 0:
            return 0

        threshold = 0.0001 # 0.0001 OR 0.0004 for buy
        if pred - curr_price >= threshold:
            amount = self.money / curr_price - 0.00001 # int(self.money / curr_price) -- this approach produces about a 0.5% smaller RoI
            if self.money > (curr_price * amount) and amount > 0:
                self.money -= (curr_price * amount)
                if stock_ticker in self.stock_counts:
                    self.stock_counts[stock_ticker] += amount
                else:
                    self.stock_counts[stock_ticker] = amount
                return amount
            return 0
        return 0

    # decide whether to sell and how much
    def determine_sell_amount(self, stock_ticker, curr_price, pred):
        if curr_price == 0:
            return 0

        threshold = -0.002 # -0.002 OR -0.0025 for sell
        if pred - curr_price <= threshold and stock_ticker in self.stock_counts and self.stock_counts[stock_ticker] > 0:
            sell_amount = self.stock_counts[stock_ticker]
            self.money += self.stock_counts[stock_ticker] * curr_price

            self.savings += self.stock_counts[stock_ticker] * curr_price * self.savings_percent
            self.money -= self.stock_counts[stock_ticker] * curr_price * self.savings_percent

            self.stock_counts[stock_ticker] = 0
            return sell_amount # sell all stocks
        return 0

    def sell_all(self):
        for stock_ticker, num_stocks in self.stock_counts.items():
            self.money += num_stocks * self.stock_prices[stock_ticker]

    def get_curr_investment_money(self):
        return self.money

    def get_curr_savings(self):
        return self.savings

    def get_total_money(self):
        return self.money + self.savings
