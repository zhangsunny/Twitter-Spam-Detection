from scipy.io import arff
import pandas as pd
import numpy as np
from model import *
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import MultinomialNB


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


def load_honey(ext=True):
    df = pd.read_csv('./data.csv')
    col_names = list(df.columns)
    col_names.remove('is_polluter')
    col_names.remove('id')
    if not ext:
        col_names.remove('age')
        col_names.remove('reputation')
        col_names.remove('ff_ratio')
        col_names.remove('freq_tweet')
    data = df[col_names].values
    label = df['is_polluter'].values
    return data, label


if __name__ == "__main__":
    print('Model,Accuracy,Precision,Recall,F1,AUC,Time')
    paths = {
        '5c': './ICC/5k-continuous.arff',
        '5r': './ICC/5k-random.arff',
    }
    data_key = '5r'
    data, label = load_arff(paths[data_key])
    # data_key = 'honey_ext'
    # data, label = load_honey(True)

    set_seed(0)
    kernels = [2] * 4 + [3] * 5
    classifier = ABC(data.shape[-1], UNIT, BILINEAR, kernels)
    train_test_single(data, label, classifier, 'ABC', TEST_SIZE, None)

    # classifier = ABC(data.shape[-1], UNIT, BILINEAR, [2])
    # train_test_single(data, label, classifier, 'ABC(win=2)', TEST_SIZE, data_key)
    # classifier = ABC(data.shape[-1], UNIT, BILINEAR, [3])
    # train_test_single(data, label, classifier, 'ABC(win=3)', TEST_SIZE, data_key)
    # classifier = AutoEncoder(data.shape[-1], UNIT, False, kernels, 'local')
    # train_test_single(data, label, classifier, 'ABC(local-concat)', TEST_SIZE, data_key)
    # classifier = AutoEncoder(data.shape[-1], UNIT, False, kernels, 'global')
    # train_test_single(data, label, classifier, 'ABC(global-concat)', TEST_SIZE, data_key)

    classifier = CNN(data.shape[-1], UNIT)
    train_test_single(data, label, classifier, 'CNN', TEST_SIZE, data_key)

    # classifier = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=0)
    # train_test_single(data, label, classifier, 'Random Fields', TEST_SIZE, data_key)

    classifier = MultinomialNB()
    train_test_single(data, label, classifier, 'Naive Bayes', TEST_SIZE, data_key)

    # classifier = DNN(data.shape[-1])
    # train_test_single(data, label, classifier, 'DNN', TEST_SIZE, data_key)

    classifier = SVC(kernel='linear', probability=True, random_state=0)
    train_test_single(data, label, classifier, 'SVM', TEST_SIZE, data_key)
    classifier = KNeighborsClassifier(5)
    train_test_single(data, label, classifier, 'KNN', TEST_SIZE, data_key)
    classifier = DecisionTreeClassifier(random_state=0)
    train_test_single(data, label, classifier, 'DT', TEST_SIZE, data_key)
