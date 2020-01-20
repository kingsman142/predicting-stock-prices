import matplotlib.pyplot as plt

def plot_stock(stock_data):
    plt.plot(range(len(stock_data)), (stock_data['Close']) / 2.0)
    plt.xticks(range(0, len(stock_data), 500), stock_data['Date'].loc[::500], rotation = 45)
    plt.xlabel('Date', fontsize = 10)
    plt.ylabel('Avg. Price', fontsize = 11)
    plt.show()
