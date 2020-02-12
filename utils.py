import matplotlib.pyplot as plt

# only used for raw data with the column names, such as 'Close' and 'Date'
def plot_stock_raw(stock_data):
    plt.plot(range(len(stock_data)), stock_data['Close'])
    plt.xticks(range(0, len(stock_data), 500), stock_data['Date'].loc[::500], rotation = 45)
    plt.xlabel('Date', fontsize = 10)
    plt.ylabel('Avg. Price', fontsize = 11)
    plt.show()

# only used for data with only number (such as vanilla python lists)
def plot_stock_clean(stock_data):
    plt.plot(range(len(stock_data)), stock_data)
    plt.xticks(range(0, len(stock_data), 500), range(0, len(stock_data), 500), rotation = 45)
    plt.xlabel('Timestep', fontsize = 10)
    plt.ylabel('Avg. Price', fontsize = 11)
    plt.show()

def plot_predictions(stock_ticker, filename, ground_truth, predictions):
    plt.plot(range(len(ground_truth)), ground_truth)
    plt.plot(range(len(ground_truth)), predictions)
    plt.xticks(range(0, len(ground_truth), 50), range(0, len(ground_truth), 50), rotation = 45)
    plt.xlabel('Timestep', fontsize = 10)
    plt.ylabel('Avg. Price', fontsize = 11)
    plt.title("Predictions for {}".format(stock_ticker))
    plt.savefig("{}.png".format(filename))
    plt.show()

# as we shift the smoothing window size, how does that impact our RoI?
def plot_roi(window_sizes, rois):
    plt.plot(window_sizes, rois)
    plt.xlabel("Smoothing window size")
    plt.ylabel("RoI %")
    plt.savefig("smoothing_window_roi.png")
    plt.show()
