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

    def prepare_data(self, starting_date, prefixes_to_remove=[], sku_to_remove=[], is_corp=False):
        orders = self.data_loader.load_data()
        orders = orders[orders["user_id"] != 0]
        orders = orders[orders["order_id"] != 0]
        orders["sku_prefix"] = orders["order_item_sku"].map(lambda x: re.findall("([a-zA-Z]+|[-]|\d+)", str(x))[0])
        orders = orders[~orders["sku_prefix"].isin(prefixes_to_remove)]
        orders = orders[~orders["order_item_sku"].isin(sku_to_remove)]
        # orders["weekday"] = orders["cdate"].map(lambda x: x.weekday())
        # orders["day"] = orders["cdate"].map(lambda x: x.day)
        # orders["month"] = orders["cdate"].map(lambda x: x.month)
        # orders["year"] = orders["cdate"].map(lambda x: x.year)
        orders = orders[orders["cdate"] > starting_date]
        orders = orders[orders["is_corp"] == is_corp]
        orders = orders[orders["cdate"] > starting_date]
        return orders

    def remove_unpopular_item_and_users(self, data):
        print("amount of users before sku remove:", len(data["user_id"].unique()))
        print("amount of sku before sku remove:", len(data["order_item_sku"].unique()))

        df_item_summary = data.groupby('order_item_sku')['user_id'].agg(["count"])
        movie_benchmark = round(df_item_summary['count'].quantile(0.8), 0)
        drop_item_list = df_item_summary[df_item_summary['count'] < movie_benchmark].index

        df_user_summary = data.groupby('user_id')['order_item_sku'].agg(["count"])
        user_benchmark = round(df_user_summary['count'].quantile(0.8), 0)
        drop_user_list = df_user_summary[df_user_summary['count'] < user_benchmark].index

        data = data[~data['order_item_sku'].isin(drop_item_list)]
        data = data[~data['user_id'].isin(drop_user_list)]

        print("amount of sku after sku remove:", len(data["order_item_sku"].unique()))
        print("amount of users after sku remove:", len(data["user_id"].unique()))
        return data
