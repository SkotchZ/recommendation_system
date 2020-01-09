import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import numpy as np
import pandas as pd
from test_and_train import train_loop
from MFbias import MFBias
from CustomDataset import CustomDataset


class MFmodel:
    def __init__(self, num_user=None, num_movie=None, emb_size=None):
        self.user_encoding = None
        self.user_decoding = None
        self.item_encoding = None
        self.item_decoding = None
        self.model = None
        self.num_user = num_user
        self.num_movie = num_movie
        self.emb_size = emb_size


    def load_binary(self, path_to_binary):
        """
        Load model data from binary
        Args:
            path_to_binary(string): path to binary file
        return: Nothing
        """
        saved_model = torch.load(path_to_binary)
        self.num_user = saved_model['num_user']
        self.num_movie = saved_model['num_movie']
        self.emb_size = saved_model['emb_size']

        self.user_encoding = saved_model['user_encoding']
        self.user_decoding = saved_model['user_decoding']
        self.item_encoding = saved_model['item_encoding']
        self.item_decoding = saved_model['item_decoding']

        self.model = MFBias(self.num_user, self.num_movie, self.emb_size)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.model.eval()

    def save_binary(self, path_to_binary):
        """
        Save model data to binary file
        Args:
            path_to_binary(string): path to binary file
        return: Nothing
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            "user_encoding": self.user_encoding,
            "user_decoding": self.user_decoding,
            "item_encoding": self.item_encoding,
            "item_decoding": self.item_decoding,
            "num_user": self.num_user,
            "num_movie": self.num_movie,
            "emb_size": self.emb_size}, path_to_binary)


    def __create_encoding_decoding(self, df):
        uniq = df['user_id'].unique()
        user_encode = {o: i for i, o in enumerate(uniq)}
        user_decode = {i: o for i, o in enumerate(uniq)}
        uniq = df['item_id'].unique()
        item_encode = {o: i for i, o in enumerate(uniq)}
        item_decode = {i: o for i, o in enumerate(uniq)}
        return user_encode, user_decode, item_encode, item_decode

    def __format_data(self, df):
        df = df.groupby(["user_id", "order_item_sku"]).count().reset_index()
        df = df[["user_id", "order_item_sku", "order_id"]]
        df.rename(columns={"order_id": "item_count", "order_item_sku": "item_id"}, inplace=True)
        return df

    def __encode_data(self, df):
        df["user_id"] = df["user_id"].apply(lambda x: self.user_encoding[x])
        df["item_id"] = df["item_id"].apply(lambda x: self.item_encoding[x])
        return df

    def fit_model(self, data, num_epoch, emb_size=50):
        """
        Fit model for recommendation
        Args:
             data():
        return: nothing
        """
        data.sort_values(by=['cdate'], axis=0, inplace=True)
        data_size = data.shape[0]
        train = data.iloc[:int(0.8 * data_size)]
        train = self.__format_data(train)

        user_ids = train["user_id"].unique()
        item_ids = train["item_id"].unique()
        num_users = user_ids.shape[0]
        num_movies = item_ids.shape[0]

        fulldata = self.__format_data(data)
        fulldata = fulldata[fulldata["user_id"].isin(user_ids)]
        fulldata = fulldata[fulldata["item_id"].isin(item_ids)]
        fulldata = pd.pivot_table(fulldata, values="item_count",
                                  index="user_id", columns="item_id",
                                  aggfunc="sum").fillna(0).to_numpy().reshape(-1)

        self.user_encoding, self.user_decoding,\
        self.item_encoding, self.item_decoding = self.__create_encoding_decoding(train)
        train = self.__encode_data(train)

        count_of_non_zero = train.shape[0]
        count_of_zeros = num_users * num_movies - count_of_non_zero
        coeff = count_of_zeros / count_of_non_zero


        new_table = pd.pivot_table(train, values="item_count",
                                   index="user_id", columns="item_id",
                                   aggfunc="sum").fillna(0)
        new_table = new_table.to_numpy().reshape(-1)

        weights = new_table.copy()
        weights[weights > 0] = coeff / (1 + coeff)
        weights[weights == 0] = 1 / (1 + coeff)

        X = np.transpose([np.tile(np.arange(num_movies), num_users), np.repeat(np.arange(num_users), num_movies)])
        X = np.concatenate((np.concatenate((X, new_table[:, np.newaxis]), axis=1), weights[:, np.newaxis]), axis=1)
        X = np.concatenate((X, fulldata[:, np.newaxis]), axis=1)

        train_ds = CustomDataset(x=X)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=100000, num_workers=20, shuffle=True)
        del data, train, X, new_table, weights

        model = MFBias(num_users, num_movies, emb_size=emb_size).cuda()
        self.emb_size = emb_size
        self.num_movie = num_movies
        self.num_user = num_users
        train_loop(model, train_dl, num_epoch, 0.05, wd=0.00001)
        self.model = model
        self.save_binary("data.tf")

    def evaluate(self, x):
        """
        Find list of SKU which are similar to give SKU.
        Args:
            x(user_id):
        return:
        """
        self.model.eval()
        size = tuple([self.num_movie])
        encoded_user = torch.full(size, self.user_encoding[x]).long()
        item_vector = torch.arange(start=0, end=self.num_movie).long()
        result = self.model(encoded_user, item_vector).detach().numpy()
        indexes = np.argsort(-result)
        scores = result[indexes][~np.isinf(result[indexes])]
        indexes = indexes[:len(scores)]
        return list(zip(np.vectorize(self.item_decoding.get)(indexes[:len(scores)]),
                        result[indexes][~np.isinf(result[indexes])]))
        return result
