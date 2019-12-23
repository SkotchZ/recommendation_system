import pandas as pd
import numpy as np
import DataLoader
import re

class DataPipeline:
    """
        Load data and perform transformation for further work.
    """
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def prepare_data(self, starting_date, prefixes_to_remove=[], sku_to_remove=[]):
        orders = self.data_loader.load_data()
        orders = orders[orders["user_id"] != 0]
        orders = orders[orders["order_id"] != 0]
        print(orders)
        orders["sku_prefix"] = orders["order_item_sku"].map(lambda x: re.findall("([a-zA-Z]+|[-]|\d+)", str(x))[0])
        orders = orders[~orders["sku_prefix"].isin(prefixes_to_remove)]
        orders = orders[~orders["order_item_sku"].isin(sku_to_remove)]
        # orders["weekday"] = orders["cdate"].map(lambda x: x.weekday())
        # orders["day"] = orders["cdate"].map(lambda x: x.day)
        # orders["month"] = orders["cdate"].map(lambda x: x.month)
        # orders["year"] = orders["cdate"].map(lambda x: x.year)
        orders = orders[orders["cdate"] > starting_date]
        return orders
