import pandas as pd
import numpy as np


class DataLoader:
    """
        Process all actions related with data loading. Also remove unnecessary columns from data.
    """
    def __init__(self, source_of_data, **data):
        """
        Init source of data and necessary data for that source.
        Args:
             source_of_data (string): Type of source data.
                                      Pass 'sql' for loading for sql server.
                                      Pass 'local storage' for loading from local storage
             data (dict, kwargs**):   Data necessary for chosen source of data
                                      For 'sql' pass ....
                                      For 'local storage' pass path to file with data or path to two files.
                                      First of them contains table with orders, and second constrains table with SKU.
        """
        self.source_of_data = source_of_data
        self.data = data

    def load_data(self):
        """
        Load data from source.
        return: pd.DataFrame with users, orders, cdate and SKU
        """
        if self.source_of_data == "sql":
            raise Exception("Loading from sql is not implemented yet")
        elif self.source_of_data == "local storage":
            orders = pd.read_csv(self.data["first"])
            orders = orders[["order_id", "user_id", "cdate"]]
            orders["cdate"] = pd.to_datetime(orders["cdate"], errors='coerce', unit='s')
            order_items = pd.read_csv(self.data["second"])
            order_items["order_item_sku"] = order_items["order_item_sku"].astype(str)
            user_type = pd.read_csv(self.data["third"])
            user_type["shopper_group_id"] = user_type["shopper_group_id"].apply(lambda x: x != 5)
            user_type.rename(columns={"shopper_group_id": "is_corp"}, inplace=True)
            return orders.merge(order_items, on="order_id", how="left").merge(user_type, on="user_id",how="inner")
        else:
            raise Exception("Unknown type of data source")
