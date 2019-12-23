import numpy as np
import pandas as pd
import pickle
from PmiModel import PmiModel


class PmiRecommendator:
    """
        Create list of recommendations based on trained PMI_model and user`s SKU list
    """
    def __init__(self, pmi_model):
        """
        Init Recomendator with trained model
        Args:
            pmi_model(PmiModel): trained PmiModel
        """
        self.pmi_model = pmi_model

    def create_list(self, list_of_users_sku, without_input_sku=False):
        """
        Create list of recommendations based on user`s SKU list
        Args:
            list_of_users_sku (list of string): list of user`s sku
            without_input_sku (bool): If True remove list_of_users_sku from suggestions
        Returns:
            list of string: list of SKU for suggestions
        """
        suggestions = []
        for sku in list_of_users_sku:
            suggestions += self.pmi_model.evaluate(sku)
        suggestions.sort(key=(lambda x: x[1]), reverse=True)
        sku_collection = set()
        index_to_save = []
        for key, sku in enumerate(suggestions):
            if sku[0] in sku_collection or (sku[0] in list_of_users_sku and without_input_sku):
                continue
            else:
                index_to_save.append(key)
                sku_collection.add(sku[0])
        print(index_to_save)
        return np.array(suggestions)[index_to_save]
