import pandas as pd
import numpy as np
import pickle


class PmiModel:
    """
        Find list of SKU which are similar to give SKU.
    """
    def __init__(self):
        self.pmi_matrix = None
        self.decoding_dict = None
        self.encoding_dict = None

    def load_binary(self, path_to_binary):
        """
        Load model data from binary
        Args:
            path_to_binary(string): path to binary file
        return: Nothing
        """
        with open(path_to_binary, 'rb') as f:
            temp = pickle.load(f)
            self.pmi_matrix = temp["pmi_matrix"]
            self.decoding_dict = temp["decoding_dict"]
            self.encoding_dict = temp["encoding_dict"]

    def save_binary(self, path_to_binary):
        """
        Save model data to binary file
        Args:
            path_to_binary(string): path to binary file
        return: Nothing
        """
        object_to_save = {"pmi_matrix": self.pmi_matrix,
                          "decoding_dict": self.decoding_dict,
                          "encoding_dict": self.encoding_dict}
        with open(path_to_binary, 'wb') as f:
            pickle.dump(object_to_save, f)

    def fit_model(self, data):
        """
        Fit model for recommendation
        Args:
             data(pd.DataFrame): with info about users and items (SKU) they bought.
        return: nothing
        """
        unique_users = data["user_id"].unique()
        items_user_list = data.groupby(["order_item_sku"]).agg({"user_id": [("list", lambda x: set(x))]})
        items_user_list.reset_index(inplace=True)
        items_user_list = [tuple(x) for x in items_user_list.to_numpy()]
        self.encoding_dict = {item[0]: key for key, item in enumerate(items_user_list)}
        self.decoding_dict = {key: item[0] for key, item in enumerate(items_user_list)}
        self.pmi_matrix = np.zeros((len(items_user_list), len(items_user_list)))
        for i in range(len(items_user_list)):
            for j in range(len(items_user_list)):
                item1 = items_user_list[i][1]
                item2 = items_user_list[j][1]
                self.pmi_matrix[i, j] = len(unique_users) * len(item1 & item2) / (len(item1) * len(item2))  # PMI
        self.pmi_matrix = np.log(self.pmi_matrix)
        print(self.pmi_matrix)

    def evaluate(self, x):
        """
        Find list of SKU which are similar to give SKU.
        Args:
            x(string): SKU for which we are trying to find similar SKU
        return:
        """
        index = self.encoding_dict[x]
        result = self.pmi_matrix[index, :]
        indexes = np.argsort(-result)
        scores = result[indexes][~np.isinf(result[indexes])]
        indexes = indexes[:len(scores)]
        return list(zip(np.vectorize(self.decoding_dict.get)(indexes[:len(scores)]),
                        result[indexes][~np.isinf(result[indexes])]))
