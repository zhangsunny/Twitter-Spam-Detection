from model import *
from utils import *
from config import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.io import arff
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.use('gtk3agg')


def load_arff(file_path):
    data_arff = arff.loadarff(file_path)
    df = pd.DataFrame(data_arff[0])
    df['class'] = df['class'] == b'spammer'
    df['class'] = np.array(df['class'], dtype=int)
    col_names = list(df.columns)
    col_names.pop(col_names.index('class'))
    data = df[col_names].values
    label = df['class'].values
    return data, label


def scatter_points(ax, code, label):
    for i in range(len(code)):
        color = 'blue' if label[i] == 0 else 'red'
        ax.scatter(*code[i], c=color, marker='o', s=5)


def tsne_visualize(model, data, label, name='tmp', num=1000, unit=3):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    test_data = np.concatenate([data[label == 0][:num], data[label == 1][:num]])
    test_label = np.concatenate([np.zeros(num), np.ones(num)])
    recons_data = classifier.visual(test_data)
    vis_model = TSNE(unit)
    code_rec = vis_model.fit_transform(recons_data)
    vis_model = TSNE(unit)
    code_raw = vis_model.fit_transform(test_data)
    fig = plt.figure(figsize=(10, 5))
    if unit == 3:
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    elif unit == 2:
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    scatter_points(ax, code_rec, test_label)
    scatter_points(ax2, code_raw, test_label)
    plt.savefig(f'./figs/tsne_{name}_{unit}d.svg', format='svg')
    plt.show()


if __name__ == "__main__":
    path_5r = './ICC/5k-random.arff'
    ckpt_dir = './ckpt/5r/'
    kernels = [2, 3] * 4
    data, label = load_arff(path_5r)
    classifier = ABC(data.shape[-1], UNIT, BILINEAR, kernels, ckpt_dir)
    tsne_visualize(classifier, data, label, '5r', unit=2)

    # EXT = True
    # df = pd.read_csv('./data.csv')
    # col_names = list(df.columns)
    # col_names.remove('is_polluter')
    # col_names.remove('id')
    # if not EXT:
    #     col_names.remove('age')
    #     col_names.remove('reputation')
    #     col_names.remove('ff_ratio')
    #     col_names.remove('freq_tweet')
    #     kernels = [2, 3] * 2
    # else:
    #     kernels = [2, 3] * 2
    # data = df[col_names].values
    # label = df['is_polluter'].values
    # ckpt_dir = './ckpt/d1/' if EXT else './ckpt/d0/'
    # classifier = ABC(data.shape[-1], UNIT, BILINEAR, kernels, ckpt_dir)
    # tsne_visualize(classifier, data, label, 'd1' if EXT else 'd0', unit=2)
