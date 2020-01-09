import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from DataPipeline import DataPipeline
from DataLoader import DataLoader
from PmiModel import PmiModel
from PmiRecommendator import PmiRecommendator
from MFmodel import train_loop, MFBias, CustomDataset, MFmodel
from itertools import product

# Пример работы рекомендательной системы на основе факторизации матрицы.




def main2():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path_order_info = "../../../all_database_in_csv/jos_vm_orders.csv"
    path_sku_info = "../../../all_database_in_csv/jos_vm_order_item.csv"
    path_to_user_type = "../../../all_database_in_csv/jos_vm_shopper_vendor_xref.csv"
    pipeline = DataPipeline(DataLoader("local storage", first=path_order_info,
                                       second=path_sku_info, third=path_to_user_type))
    data = pipeline.prepare_data(datetime(2019, 1, 1), prefixes_to_remove=["RP"])
    data = pipeline.remove_unpopular_item_and_users(data)
    model = MFmodel()
    model.fit_model(data, 22)
    #data = data.groupby('user_id').agg({"order_item_sku": [("list", lambda x: list(x)), ("count", "count")]})
    model.load_binary("data.tf")
    print(model.evaluate(6709))



if __name__ == "__main__":
    main2()
