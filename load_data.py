# file kfkd.py
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from collections import namedtuple

import pickle

FTRAIN = './data/Q3/training.csv'
FTEST = './data/Q3/test.csv'
PICKLE = './data/Q3/dataset.pickle'


Dataset = namedtuple('Dataset', ['train_x', 'train_y', 'val_x', 'val_y', 'test_x'])


def load_data(load_pickle=True):

    if not os.path.exists(PICKLE):
        load_pickle = False

    if load_pickle:
        pickle_fname = PICKLE
        with open(pickle_fname, 'rb') as f:
            dataset = pickle.load(f)

    else:
        train_x = None
        val_x = None
        test_x = None
        train_y = None
        val_y = None

        for f_name in [FTEST, FTRAIN]:
            df = read_csv(os.path.expanduser(f_name))
            df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
            df = df.dropna()
            x = np.vstack(df['Image'].values) / 255.
            x = x.astype(np.float32)
            if f_name == FTRAIN:
                y = df[df.columns[:-1]].values
                y = (y - 48) / 48
                x, y = shuffle(x, y, random_state=42)
                y = y.astype(np.float32)
                num_val = x.shape[0] // 10

                val_x = x[:num_val]
                train_x = x[num_val:]

                val_y = y[:num_val]
                train_y = y[num_val:]
            else: # test
                test_x = x

        dataset = Dataset(train_x, train_y, val_x, val_y, test_x)
        pickle_fname = PICKLE
        with open(pickle_fname, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset


def example():
    dataset = load_data(load_pickle=False)
    print('Dataset:')
    for k, v in dataset._asdict().items():
        print('{:10} --> shape: {}'.format(k, v.shape))

    dataset = load_data(load_pickle=True)
    print('Dataset (Reloaded!):')
    for k, v in dataset._asdict().items():
        print('{:10} --> shape: {}'.format(k, v.shape))


if __name__ == '__main__':
    example()
