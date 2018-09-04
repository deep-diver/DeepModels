from urllib.request import urlretrieve
from os.path import isfile, isdir

from tqdm import tqdm
import gzip
import shutil
import struct
import tarfile
import pickle
import numpy as np

import skimage
import skimage.io
import skimage.transform

from dataset.dataset import Dataset
from dataset.dataset import DownloadProgress

class Mnist(Dataset):
    def __init__(self):
        Dataset.__init__(self, name='MNIST', path='mnist_dataset', num_classes=10, num_batch=1)
        self.width = 28
        self.height = 28

    def __download__(self):
        # training dataset
        if not isfile('train-images-idx3-ubyte.gz'):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='MNIST training dataset (images)') as pbar:
                urlretrieve(
                    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                    'train-images-idx3-ubyte.gz',
                    pbar.hook)
        else:
            print('train-images-idx3-ubyte.gz already exists')

        if not isfile('train-labels-idx1-ubyte.gz'):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='MNIST training dataset (labels)') as pbar:
                urlretrieve(
                    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    pbar.hook)
        else:
            print('train-labels-idx1-ubyte.gz already exists')

        # testing dataset
        if not isfile('t10k-images-idx3-ubyte.gz'):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='MNIST testing dataset (images)') as pbar:
                urlretrieve(
                    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    pbar.hook)
        else:
            print('t10k-images-idx3-ubyte.gz already exists')

        if not isfile('t10k-labels-idx1-ubyte.gz'):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='MNIST testing dataset (labels)') as pbar:
                urlretrieve(
                    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz',
                    pbar.hook)
        else:
            print('t10k-labels-idx1-ubyte.gz already exists')

        # unzip
        with gzip.open('./train-images-idx3-ubyte.gz', 'rb') as gz_in:
            with open('./train-images-idx3-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

        with gzip.open('./train-labels-idx1-ubyte.gz', 'rb') as gz_in:
            with open('./train-labels-idx1-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

        with gzip.open('./t10k-images-idx3-ubyte.gz', 'rb') as gz_in:
            with open('./t10k-images-idx3-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

        with gzip.open('./t10k-labels-idx1-ubyte.gz', 'rb') as gz_in:
            with open('./t10k-labels-idx1-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

    def __preprocess_and_save_data__(self):
        raise NotImplementedError

    def __load_batch__(self, batch_id):
        raise NotImplementedError

    def load_valid_set(self):
        raise NotImplementedError

    def batch_features_labels(self, features, labels, batch_size):
        raise NotImplementedError

    def load_preprocess_training_batch(self, batch_id, batch_size, scale_to_imagenet=False):
        raise NotImplementedError
