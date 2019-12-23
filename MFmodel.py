import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch.autograd import Variable as V

import matplotlib.pyplot as plt

class MF(nn.Module):
    def __init__(self, num_user, num_movie, emb_size):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_user, emb_size)
        self.movie_emb = nn.Embedding(num_movie, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.movie_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.movie_emb(v)
        return F.sigmoid((u * v).sum(1)) * 4 + 1


class MFBias(nn.Module):
    def __init__(self, num_user, num_movie, emb_size):
        super(MFBias, self).__init__()
        self.user_emb = nn.Embedding(num_user, emb_size)
        self.movie_emb = nn.Embedding(num_movie, emb_size)
        self.user_bias = nn.Embedding(num_user, 1)
        self.movie_bias = nn.Embedding(num_movie, 1)

        self.user_emb.weight.data.uniform_(0, 0.05)
        self.movie_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.movie_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.movie_emb(v)

        b_u = self.user_bias(u).squeeze()
        b_v = self.movie_bias(v).squeeze()

        return F.sigmoid((U * V).sum(1) + b_u + b_v) * 4 + 1


def test_loss(model, val):
    model.eval()
    preds = []
    user = V(torch.LongTensor(val.Cust_Id.values)).cuda()
    movie = V(torch.LongTensor(val.Movie_Id.values)).cuda()
    rating = V(torch.LongTensor(val.Rating.values)).float().cuda()
    y_hat = model(user, movie)
    loss = F.mse_loss(y_hat, rating)
    #     print("Validation loss %.3f " % loss.data[0])
    #     return y_hat
    return loss.data.item()

def train_loop(model, train_dl, val, epochs, learning_rate, wd=0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimiser = torch.optim.Adam(parameters, learning_rate, weight_decay=wd)
    for i in range(epochs):
        model.train()
        for j, d in enumerate(train_dl):
            mb_loss = []
            user = V(d[0]).cuda()
            movie = V(d[1]).cuda()
            rating = V(d[2]).float().cuda()
            y_hat = model(user, movie)
            loss = F.mse_loss(y_hat, rating)
            optimiser.zero_grad()
            loss.backward()
            mb_loss.append(loss.data.item())
            optimiser.step()
        print(f'Training loss for epoch {i} = {np.mean(mb_loss)}')
        print(f'Validation loss for epoch {i} = {test_loss(model, val)}')



class CustomDataset(Dataset):
    def __init__(self, df):
        self.u = torch.LongTensor(df.Cust_Id.values)
        self.v = torch.LongTensor(df.Movie_Id.values)
        self.y = torch.LongTensor(df.Rating.values)

    def __len__(self):
        self.len = len(self.u)
        return self.len

    def __getitem__(self, index):
        return self.u[index], self.v[index], self.y[index]