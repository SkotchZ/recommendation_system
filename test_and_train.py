from other import get_default_visdom_env
from visdom import Visdom
from tqdm import tqdm
import numpy as np
import copy
import torch
import torch.nn.functional as F


def test_loss(model, val):
    model.eval()
    mb_loss = []
    for d in tqdm(val):
        movie = d[:, 0].long().cuda()
        user = d[:, 1].long().cuda()
        rating = d[:, 4].float().cuda()
        y_hat = model(user, movie)
        loss = F.mse_loss(y_hat, rating)
        mb_loss.append(loss.data.item())

    return np.mean(mb_loss)


def train_loop(model, train_dl, epochs, learning_rate, wd=0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    acc_history = {'train': np.array([]), 'val': np.array([])}
    viz = get_default_visdom_env()
    win = viz.line(
        X=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
        Y=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
    )
    optimiser = torch.optim.Adam(parameters, learning_rate, weight_decay=wd)
    for i in range(epochs):
        print("epoch {0}".format(i))
        model.train()
        for d in tqdm(train_dl):
            mb_loss = []
            d = d.cuda()
            movie = d[:, 0].long()
            user = d[:, 1].long()
            rating = d[:, 2].float()
            weight = d[:, 3].float()
            y_hat = model(user, movie)
            loss = torch.mean(weight * (y_hat - rating) ** 2)
            optimiser.zero_grad()
            loss.backward()
            mb_loss.append(loss.data.item())
            optimiser.step()

        train_loss = np.mean(mb_loss)
        val_loss = test_loss(model, train_dl)
        print(f'Training loss for epoch {i} = {train_loss}')
        print(f'Validation loss for epoch {i} = {val_loss}')
        acc_history["train"] = np.append(acc_history["train"], train_loss)
        acc_history["val"] = np.append(acc_history["val"], val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        viz.line(
            X=np.column_stack(
                (np.arange(0, acc_history['train'].shape[0]),
                 np.arange(0, acc_history['val'].shape[0]))),
            Y=np.column_stack((acc_history['train'],
                               acc_history['val'])),
            win=win,
            update='insert',
            opts=dict(title='{}'.format(optimiser.defaults))
        )
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
