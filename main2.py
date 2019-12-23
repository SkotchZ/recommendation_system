import sys
import pandas as pd
import numpy as np
from datetime import datetime
from DataPipeline import DataPipeline
from DataLoader import DataLoader
from PmiModel import PmiModel
from PmiRecommendator import PmiRecommendator
from MFmodel import CustomDataset, MF, train_loop, MFBias

# Пример работы рекомендательной системы на основе факторизации матрицы.


def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids.
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o: i for i, o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)


def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids.
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ['user_id', 'order_item_sku']:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _, col, _ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df


def main2():
    path_order_info = "../../../all_database_in_csv/jos_vm_orders.csv"
    path_sku_info = "../../../all_database_in_csv/jos_vm_order_item.csv"
    pipeline = DataPipeline(DataLoader("local storage", first=path_order_info, second=path_sku_info))
    data = pipeline.prepare_data(datetime(2019, 1, 1), prefixes_to_remove=["RP"])
    data.sort_values(by=['cdate'], axis=0, inplace=True)
    print(data)
    df_encode = encode_data(data)
    print(df_encode)
    k = round(len(df_encode) * 0.8)
    train = df_encode[:k]
    val = df_encode[k:]
    num_users = len(df_encode.Cust_Id.unique())
    num_movies = len(df_encode.Movie_Id.unique())
    emb_size = 50
    train_ds = CustomDataset(train[['Cust_Id', 'Rating', 'Movie_Id']])
    train_dl = DataLoader(train_ds, batch_size=100000, shuffle=True)

    model = MF(num_users, num_movies, emb_size).cuda()
    train_loop(model, train_dl, val[['Cust_Id', 'Rating', 'Movie_Id']], 3, 0.05, wd=0.00001)  # to do take most popular quantile(0.8)
    # bias_model = MFBias(num_users, num_movies, emb_size).cuda()
    # train_loop(bias_model, train_dl, val, 3, 0.05, wd=0.00001)


if __name__ == "__main__":
    main2()
