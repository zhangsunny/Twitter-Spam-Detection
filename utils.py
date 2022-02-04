import torch
import os
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
import time
import logging
from logging import Logger
import matplotlib.pyplot as plt


plt.rc('font', family='Times New Roman', size='12', weight='normal')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_test_single(data, label, cf, name, test_size=0.4, save_path=None):
    # print('data, label, sum(label):', data.shape, label.shape, np.sum(label))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    tic = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=test_size, random_state=0, stratify=label)
    probas_ = cf.fit(x_train, y_train).predict_proba(x_test)
    pred = cf.predict(x_test)
    toc = time.time()
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * recall * precision) / (recall + precision)
    # info = '{}-Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1: {:.4f}; AUC: {:.4f}'.format(
    #     name, accuracy, precision, recall, f1, roc_auc)
    # print(info)
    # print('Time Cost: {:.2f} s'.format(toc-tic))
    print(f'{name},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{roc_auc:.4f},{toc-tic:.2f}')
    save_path = './results/auc' if save_path is None else './results/auc/'+save_path
    os.makedirs(save_path, exist_ok=True)
    np.savez(save_path+'/{}.npz'.format(name), fpr=fpr, tpr=tpr, auc=roc_auc)
    return roc_auc, f1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda_mode():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


def now_datetime(ms=False):
    now = datetime.now()
    if not ms:
        now = now.replace(microsecond=0)
    return now


def get_logger(path='./logs/', name=None, level=logging.WARN, logger_name='logger'):
    formatter = logging.Formatter('%(asctime)s %(filename)s,%(message)s', "%Y-%m-%d %H:%M:%S")
    if name is None:
        now = now_datetime()
        name = str(now)+'.log'
    log_file = os.path.join(path, name)
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    vlog = logging.getLogger(logger_name)
    vlog.setLevel(level)
    vlog.addHandler(fileHandler)
    return vlog


def trans_cuda(*tensors):
    out = []
    float_type = [torch.float64, torch.float16]
    int_type = [torch.int, torch.int8, torch.int16, torch.int32]
    for tensor in tensors:
        if tensor.dtype in float_type:
            tensor = tensor.float()
        elif tensor.dtype in int_type:
            tensor = tensor.long()
        tensor = tensor.cuda()
        out.append(tensor)
    return out


def write_scalar(writer, name, epoch, metrics: dict):
    for k, v in metrics.items():
        writer.add_scalar(name+'/'+k, v, epoch)
