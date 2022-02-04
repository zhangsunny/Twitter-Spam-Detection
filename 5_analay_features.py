"""
seaborn for visualizing the features
Ref from: https://blog.csdn.net/mago2015/article/details/84290362
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('GTK3Agg')
plt.rc('font', family='Times New Roman')


if __name__ == "__main__":
    df = pd.read_csv('./data.csv')
    col_names = list(df.columns)
    col_names.pop(col_names.index('is_polluter'))
    col_names.pop(col_names.index('id'))
    col_names.append('is_polluter')
    # col_names = col_names[::-1]
    # data = df[col_names].values
    # label = df['is_polluter'].values
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)
    corr_feature = df[col_names].corr()
    print(corr_feature)

    fig = plt.figure(figsize=(8, 7))
    sns.set(font_scale=0.8, style='ticks', context='notebook')
    hm = sns.heatmap(
        corr_feature, cbar=True, annot=True, fmt='.2f', square=True,
        cmap=plt.cm.gray,
        annot_kws={'size': 10}, yticklabels=True, xticklabels=True, vmin=-0.5,
        cbar_kws={"shrink": 0.8})
    plt.ylim(10, 0)
    plt.savefig('./figs/heatmap_feature.svg', format='svg')
    plt.show()

    # data = df[col_names[:-1]].values
    # label = df['is_polluter'].values
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)
    # df[col_names[:-1]] = data

    # label2 = ['legal']*len(label)
    # label2 = np.array(label2)
    # label2[label == 1] = 'spammer'
    # df['is_polluter'] = label2
    # # col_pair = ['age', 'reputation', 'ff_ratio', 'freq_tweet', 'is_polluter']
    # col_pair = ['age', 'reputation', 'is_polluter']
    # sns.set(style='ticks', context='notebook')
    # sns.pairplot(df[col_pair], hue='is_polluter', markers=['.', 'D'], diag_kind='hist',
    #              plot_kws=dict(s=15, color='k'))
    # plt.savefig('./figs/pairplot_feature.jpg', format='jpg')
    # plt.show()
    # # plt.close()
