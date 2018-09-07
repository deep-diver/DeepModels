from urllib.request import urlretrieve
from os.path import isfile, isdir
import os

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
        Dataset.__init__(self, name='MNIST', path='mnist_dataset', num_classes=10, num_batch=5)
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

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        # unzip
        with gzip.open('./train-images-idx3-ubyte.gz', 'rb') as gz_in:
            with open(self.path + '/train-images-idx3-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

        with gzip.open('./train-labels-idx1-ubyte.gz', 'rb') as gz_in:
            with open(self.path + '/train-labels-idx1-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

        with gzip.open('./t10k-images-idx3-ubyte.gz', 'rb') as gz_in:
            with open(self.path + '/t10k-images-idx3-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

        with gzip.open('./t10k-labels-idx1-ubyte.gz', 'rb') as gz_in:
            with open(self.path + '/t10k-labels-idx1-ubyte', 'wb') as gz_out:
                shutil.copyfileobj(gz_in, gz_out)

    def __load_batch__(self, batch_id=1):
        with open(self.path + '/train-labels-idx1-ubyte', 'rb') as train_label_file:
            magic, num = struct.unpack(">II", train_label_file.read(8))
            labels = np.fromfile(train_label_file, dtype=np.int8)

        with open(self.path + '/train-images-idx3-ubyte', mode='rb') as train_image_file:
            magic, num, rows, cols = struct.unpack(">IIII", train_image_file.read(16))
            features = np.fromfile(train_image_file, dtype=np.uint8).reshape(len(labels), rows, cols)

        return features, labels

    def __preprocess_and_save_data__(self, valid_ratio=0.1):
        valid_features = []
        valid_labels = []
        flag = True

        features, labels = self.__load_batch__()

        tmp_features = []
        for feature in features:
            tmp_features.append(np.resize(feature, (28, 28, 3)))

        features = np.asarray(tmp_features)
        features = features.reshape(len(labels), 3, 28, 28).transpose(0, 2, 3, 1)

        index_of_validation = int(len(features) * valid_ratio)
        total_train_num = len(features) - int(len(features) * valid_ratio)
        n_batches = self.num_batch

        for batch_i in range(1, n_batches + 1):
            batch_filename = self.path + '/mnist_preprocess_batch_' + str(batch_i) + '.p'

            if isfile(batch_filename):
                print(batch_filename + ' already exists')
                flag = False
            else:
                start_index = int((batch_i-1) * total_train_num/n_batches)
                end_index = int(start_index + total_train_num/n_batches)

                self.save_preprocessed_data(features[start_index:end_index], labels[start_index:end_index], batch_filename)

        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

        # preprocess the all stacked validation dataset
        self.save_preprocessed_data(np.array(valid_features), np.array(valid_labels), self.path + '/mnist_preprocess_validation.p')

        # load the test dataset
        with open(self.path + '/t10k-labels-idx1-ubyte', 'rb') as test_label_file:
            magic, num = struct.unpack(">II", test_label_file.read(8))
            test_labels = np.fromfile(test_label_file, dtype=np.int8)

        with open(self.path + '/t10k-images-idx3-ubyte', mode='rb') as test_image_file:
            magic, num, rows, cols = struct.unpack(">IIII", test_image_file.read(16))
            test_features = np.fromfile(test_image_file, dtype=np.uint8).reshape(len(test_labels), rows, cols)

        tmp_features = []
        for feature in test_features:
            tmp_features.append(np.resize(feature, (28, 28, 3)))

        test_features = np.asarray(tmp_features)
        test_features = test_features.reshape(len(test_labels), 3, 28, 28).transpose(0, 2, 3, 1)

        # Preprocess and Save all testing data
        self.save_preprocessed_data(np.array(test_features), np.array(test_labels), self.path + '/mnist_preprocess_test.p')

    def get_batches_from(self, features, labels, batch_size):
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

    def get_training_batches_from_preprocessed(self, batch_id, batch_size, scale_to_imagenet=False):
        filename = self.path + '/mnist_preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        if scale_to_imagenet:
            features = self.convert_to_imagenet_size(features)

        return self.get_batches_from(features, labels, batch_size)

    def get_valid_set(self, scale_to_imagenet=False):
        filename = self.path + '/mnist_preprocess_validation.p'
        valid_features, valid_labels = pickle.load(open(filename, mode='rb'))

        if scale_to_imagenet:
            valid_features = self.convert_to_imagenet_size(valid_features)

        return valid_features, valid_labels
