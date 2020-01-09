import sys
import pandas as pd
import numpy as np
from datetime import datetime
from DataPipeline import DataPipeline
from DataLoader import DataLoader
from PmiModel import PmiModel
from PmiRecommendator import PmiRecommendator


# Пример работы для рекомендательной системы на основе PMI модели.


def main():
    path_order_info = "../../../all_database_in_csv/jos_vm_orders.csv"
    path_sku_info = "../../../all_database_in_csv/jos_vm_order_item.csv"
    path_to_user_type = "../../../all_database_in_csv/jos_vm_shopper_vendor_xref.csv"
    # Инициируем конвеер предварительной обработки загрузчиком, который на локальной машине открывает два файла.
    pipeline = DataPipeline(DataLoader("local storage", first=path_order_info,
                                       second=path_sku_info, third=path_to_user_type))
    data = pipeline.prepare_data(datetime(2019, 1, 1), prefixes_to_remove=["RP"])

    # Удаляем непопулярные SKU и user-ов(нужно ли удалять юзеров?)
    data = pipeline.remove_unpopular_item_and_users(data)

    # Инициализируем, обучаем, сохраняем, загружаем модель матрицы PMI
    model = PmiModel()
    model.fit_model(data)
    model.save_binary("data")
    model.load_binary("data")
    # print(model.pmi_matrix.shape)
    # print(model.evaluate("AMX-02"))
    # Находим все sku некоторого пользователся
    grouped_data = data.groupby('user_id').agg({"order_item_sku": [("list", lambda x: list(x)), ("count", "count")]})
    users_sku = grouped_data.loc[6709][("order_item_sku", "list")]
    print(users_sku)
    # Инициализируем рекоменадательную систему моделью. Получаем список предсказаний с очками для каждого из них.
    recomendator = PmiRecommendator(model)
    answer = recomendator.create_list(users_sku)
    print(answer.shape, answer)


if __name__ == "__main__":
    main()
