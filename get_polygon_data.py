import requests
import pandas as pd
import datetime
from datetime import datetime as dt

class PolygonAPI():
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "api.polygon.io/v1/"

    def get_stock_historical_open(self, symbol, start_date):
        data = []
        dates = self.get_all_dates(start_date, dt.today().date().isoformat())

        for date in dates:
            request_url = self.base_url + "open-close/{}/{}?apiKey={}".format(symbol.upper(), start_date, self.api_key)
            request_json = requests.get(request_url).json()

            if request_json["status"] == "OK":
                open_price = request_json["open"]
                data.append([date, open_price])
        data = pd.DataFrame(data, columns = ['Date', symbol.lower()])
        return data

    def get_all_dates(self, begin_date, end_date):
        begin_date = dt.strptime(begin_date, "%Y-%m-%d").date() # turn this into a Date object
        end_date = dt.strptime(end_date, "%Y-%m-%d").date() # turn this into a Date object
        diff = end - begin # get the number of days between the begin and end date
        all_dates_between = [(begin_date + datetime.timedelta(i)).isoformat() for i in range(diff.days + 1)] # iterate between begin and end dates and convert it into a date string
        return all_dates_between
