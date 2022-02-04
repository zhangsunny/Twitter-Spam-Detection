import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import *
import os
from config import *


def get_network(in_dim, out_dim):
    net = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(DROP),
    )
    return net


def get_loader(data, label, batch_size):
    data = torch.tensor(data, dtype=torch.float32).to(DEVICE)
    label = torch.tensor(label, dtype=torch.float32).to(DEVICE).view(-1, 1)
    dataset = TensorDataset(data, label)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return loader


class AEBiDNN(nn.Module):
    def __init__(self, in_dim, unit=8, bilinear=True, kernels=[2, 2, 3, 3]):
        super().__init__()
        self.in_dim = in_dim
        self.kernels = kernels
        self.convs = nn.ModuleList([nn.Conv1d(1, 1, k) for k in self.kernels])
        self.dim = (1+len(self.kernels)) * in_dim - \
            np.sum(self.kernels) + len(self.kernels)
        self.encode = nn.Sequential(
            # get_network(self.dim, 1024),
            # get_network(1024, 256),
            # get_network(256, 64),
            get_network(self.dim, 128),
            get_network(128, 64),
            get_network(64, unit),
            # nn.Linear(64, unit),
        )
        self.decode = nn.Sequential(
            get_network(unit, 64),
            get_network(64, 128),
            nn.Linear(128, in_dim),
            # get_network(64, 256),
            # get_network(256, 1024),
            # nn.Linear(1024, in_dim),
            nn.Sigmoid(),
        )
        self.nn = BiNN(in_dim, bilinear)

    def forward(self, x):
        x_in = x.unsqueeze(1)
        feats = [x_in]
        for conv in self.convs:
            feats.append(conv(x_in))
        feat = torch.cat(feats, dim=-1).squeeze(1)
        code = self.encode(feat)
        x_ = self.decode(code)
        logit = self.nn(x, x_)
        return x_, code, logit


class BiNN(nn.Module):
    def __init__(self, in_dim, bilinear):
        super().__init__()
        self.in_dim = in_dim
        self.bilinear = bilinear
        if bilinear:
            self.fc_layer = nn.Bilinear(in_dim, in_dim, 128)
        else:
            self.fc_layer = nn.Linear(in_dim*2, 128)
        self.out_layer = nn.Sequential(
            # get_network(512, 128),
            get_network(128, 64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_):
        if self.bilinear:
            feat = self.fc_layer(x, x_)
        else:
            feat = torch.cat([x, x_], dim=-1)
            feat = self.fc_layer(feat)
        logit = self.out_layer(feat)
        return logit


class NN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.fc_layer = nn.Sequential(
            get_network(in_dim, 128),
            get_network(128, 64),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.fc_layer(x)
        logit = self.out_layer(feat)
        return logit


# 只使用局部信息
class AELocal(nn.Module):
    def __init__(self, in_dim, unit=8, bilinear=True, kernels=[2, 2, 3, 3]):
        super().__init__()
        self.in_dim = in_dim
        self.kernels = kernels
        self.convs = nn.ModuleList([nn.Conv1d(1, 1, k) for k in self.kernels])
        self.dim = len(self.kernels) * in_dim - \
            np.sum(self.kernels) + len(self.kernels)
        self.encode = nn.Sequential(
            get_network(self.dim, 128),
            get_network(128, 64),
            get_network(64, unit),
        )
        self.decode = nn.Sequential(
            get_network(unit, 64),
            get_network(64, 128),
            nn.Linear(128, in_dim),
            nn.Sigmoid(),
        )
        self.nn = BiNN(in_dim, bilinear)

    def forward(self, x):
        x_in = x.unsqueeze(1)
        feats = []
        for conv in self.convs:
            feats.append(conv(x_in))
        feat = torch.cat(feats, dim=-1).squeeze(1)
        code = self.encode(feat)
        x_ = self.decode(code)
        logit = self.nn(x, x_)
        return x_, code, logit


# without local information
class AE(nn.Module):
    def __init__(self, in_dim, unit=8, bilinear=True):
        super().__init__()
        self.in_dim = in_dim
        self.encode = nn.Sequential(
            get_network(in_dim, 128),
            get_network(128, 64),
            get_network(64, unit),
        )
        self.decode = nn.Sequential(
            get_network(unit, 64),
            get_network(64, 128),
            nn.Linear(128, in_dim),
            nn.Sigmoid(),
        )
        self.nn = BiNN(in_dim, bilinear)

    def forward(self, x):
        code = self.encode(x)
        x_ = self.decode(code)
        logit = self.nn(x, x_)
        return x_, code, logit


class ABC():
    def __init__(self, dim, unit, bilinear, kernels=[2, 2, 3, 3], ckpt_dir='./ckpt/'):
        super().__init__()
        # set_seed(0)
        self.dim = dim
        self.unit = unit
        self.model = AEBiDNN(dim, unit, bilinear, kernels).to(DEVICE)
        self.lr = LR
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.crit_ae = torch.nn.CosineSimilarity().to(DEVICE)
        self.crit_nn = torch.nn.BCELoss(reduction='none').to(DEVICE)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        # self.writer = SummaryWriter(
        #     f'./logs/{unit}-{LR}-{DROP}-{BILINEAR}', flush_secs=30)

    def train_epoch(self, loader):
        self.model.train()
        sum_loss, count = 0, 0
        for batch_x, batch_y in loader:
            self.optim.zero_grad()
            x_, code, logit = self.model(batch_x)
            loss_ae = self.crit_ae(x_, batch_x).view(-1, 1)
            loss_nn = self.crit_nn(logit, batch_y)
            loss = torch.mean(batch_y*loss_ae - (1-batch_y)*loss_ae + loss_nn)
            loss.backward()
            self.optim.step()
            count += len(batch_y)
            sum_loss += len(batch_y) * loss.item()
        return sum_loss/count

    def fit(self, data, label, epoches=EPOCH, batch_size=BATCH, verbose=VERB):
        loader = get_loader(data, label, batch_size)
        for epoch in np.arange(1, epoches+1):
            loss = self.train_epoch(loader)
            loss_dict = dict([('train', loss)])
            # write_scalar(self.writer, 'loss', epoch, loss_dict)
            torch.save(self.model.state_dict(), self.ckpt_dir+'ABC.ckpt')
            if verbose and (epoch % 50 == 0):
                print(
                    'Epoch: {}; Loss: {:.4f}'.format(
                        epoch, loss))
        # self.writer.close()
        return self

    def predict_proba(self, x_test):
        # self.model.load_state_dict(torch.load(self.ckpt_dir+'ABC.ckpt'))
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_, code, logit = self.model(x_test)
            logit = logit.data.cpu().numpy()
        logit = logit.reshape(1, -1)
        probs_mat = np.zeros([len(x_test), 2])
        probs_mat[:, 0] = 1 - logit
        probs_mat[:, 1] = logit
        return probs_mat

    def predict(self, x_test):
        probs_mat = self.predict_proba(x_test)
        return np.argmax(probs_mat, axis=1)

    def visual(self, x_test):
        self.model.load_state_dict(torch.load(self.ckpt_dir+'ABC.ckpt'))
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_, code, logit = self.model(x_test)
            x_ = x_.data.cpu().numpy()
        return x_


class DNN():
    def __init__(self, in_dim):
        super().__init__()
        # set_seed(0)
        self.model = NN(in_dim).to(DEVICE)
        self.lr = LR
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.crit_nn = torch.nn.BCELoss().to(DEVICE)

    def train_epoch(self, loader):
        self.model.train()
        sum_loss, count = 0, 0
        for batch_x, batch_y in loader:
            self.optim.zero_grad()
            logit = self.model(batch_x)
            loss = self.crit_nn(logit, batch_y)
            loss.backward()
            self.optim.step()
            count += len(batch_y)
            sum_loss += len(batch_y) * loss.item()
        return sum_loss/count

    def fit(self, data, label, epoches=EPOCH, batch_size=BATCH, verbose=VERB):
        loader = get_loader(data, label, batch_size)
        for epoch in np.arange(1, epoches+1):
            loss = self.train_epoch(loader)
            loss_dict = dict([('train', loss)])
            if verbose and (epoch % 50 == 0):
                print(
                    'Epoch: {}; Loss: {:.4f}'.format(
                        epoch, loss))
        return self

    def predict_proba(self, x_test):
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logit = self.model(x_test)
            logit = logit.data.cpu().numpy()
        logit = logit.reshape(1, -1)
        probs_mat = np.zeros([len(x_test), 2])
        probs_mat[:, 0] = 1 - logit
        probs_mat[:, 1] = logit
        return probs_mat

    def predict(self, x_test):
        probs_mat = self.predict_proba(x_test)
        return np.argmax(probs_mat, axis=1)


class AutoEncoder():
    def __init__(self, dim, unit, bilinear, kernels=[2, 2, 3, 3], kind='local'):
        super().__init__()
        # set_seed(0)
        self.dim = dim
        self.unit = unit
        if kind == 'local':
            self.model = AELocal(dim, unit, bilinear, kernels).to(DEVICE)
        elif kind == 'global':
            self.model = AE(dim, unit, bilinear).to(DEVICE)
        self.lr = LR
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.crit_ae = torch.nn.CosineSimilarity().to(DEVICE)
        self.crit_nn = torch.nn.BCELoss(reduction='none').to(DEVICE)

    def train_epoch(self, loader):
        self.model.train()
        sum_loss, count = 0, 0
        for batch_x, batch_y in loader:
            self.optim.zero_grad()
            x_, code, logit = self.model(batch_x)
            loss_ae = self.crit_ae(x_, batch_x).view(-1, 1)
            loss_nn = self.crit_nn(logit, batch_y)
            loss = torch.mean(batch_y*loss_ae - (1-batch_y)*loss_ae + loss_nn)
            loss.backward()
            self.optim.step()
            count += len(batch_y)
            sum_loss += len(batch_y) * loss.item()
        return sum_loss/count

    def fit(self, data, label, epoches=EPOCH, batch_size=BATCH, verbose=VERB):
        loader = get_loader(data, label, batch_size)
        for epoch in np.arange(1, epoches+1):
            loss = self.train_epoch(loader)
            loss_dict = dict([('train', loss)])
            if verbose and (epoch % 50 == 0):
                print(
                    'Epoch: {}; Loss: {:.4f}'.format(
                        epoch, loss))
        return self

    def predict_proba(self, x_test):
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_, code, logit = self.model(x_test)
            logit = logit.data.cpu().numpy()
        logit = logit.reshape(1, -1)
        probs_mat = np.zeros([len(x_test), 2])
        probs_mat[:, 0] = 1 - logit
        probs_mat[:, 1] = logit
        return probs_mat

    def predict(self, x_test):
        probs_mat = self.predict_proba(x_test)
        return np.argmax(probs_mat, axis=1)


class CNN():
    def __init__(self, dim, unit, kernels=[2]):
        super().__init__()
        # set_seed(0)
        self.dim = dim
        self.unit = unit
        self.model = AELocal(dim, unit, False, kernels).to(DEVICE)
        self.lr = LR
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.crit_nn = torch.nn.BCELoss(reduction='none').to(DEVICE)

    def train_epoch(self, loader):
        self.model.train()
        sum_loss, count = 0, 0
        for batch_x, batch_y in loader:
            self.optim.zero_grad()
            x_, code, logit = self.model(batch_x)
            loss_nn = self.crit_nn(logit, batch_y)
            loss = torch.mean(loss_nn)
            loss.backward()
            self.optim.step()
            count += len(batch_y)
            sum_loss += len(batch_y) * loss.item()
        return sum_loss/count

    def fit(self, data, label, epoches=EPOCH, batch_size=BATCH, verbose=VERB):
        loader = get_loader(data, label, batch_size)
        for epoch in np.arange(1, epoches+1):
            loss = self.train_epoch(loader)
            loss_dict = dict([('train', loss)])
            if verbose and (epoch % 50 == 0):
                print(
                    'Epoch: {}; Loss: {:.4f}'.format(
                        epoch, loss))
        return self

    def predict_proba(self, x_test):
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_, code, logit = self.model(x_test)
            logit = logit.data.cpu().numpy()
        logit = logit.reshape(1, -1)
        probs_mat = np.zeros([len(x_test), 2])
        probs_mat[:, 0] = 1 - logit
        probs_mat[:, 1] = logit
        return probs_mat

    def predict(self, x_test):
        probs_mat = self.predict_proba(x_test)
        return np.argmax(probs_mat, axis=1)
