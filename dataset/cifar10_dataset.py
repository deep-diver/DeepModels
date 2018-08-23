from urllib.request import urlretrieve
from os.path import isfile, isdir

from tqdm import tqdm
import tarfile
import pickle
import numpy as np

import skimage
import skimage.io
import skimage.transform

from dataset.dataset import Dataset
from dataset.dataset import DownloadProgress

class Cifar10(Dataset):
    def __init__(self):
        Dataset.__init__(self, name='Cifar-10', path='cifar-10-batches-py', num_batch=5)
        self.width = 32
        self.height = 32
        self.num_classes = 10

    def download(self):
        if not isfile('cifar-10-python.tar.gz'):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                urlretrieve(
                    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                    'cifar-10-python.tar.gz',
                    pbar.hook)
        else:
            print('cifar-10-python.tar.gz already exists')

        if not isdir(self.path):
            with tarfile.open('cifar-10-python.tar.gz') as tar:
                tar.extractall()
                tar.close()
        else:
            print('cifar10 dataset already exists')

    def load_batch(self, batch_id):
        with open(self.path + '/data_batch_' + str(batch_id), mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')

        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']

        return features, labels

    def preprocess_and_save_data(self, valid_ratio=0.1):
        n_batches = 5
        valid_features = []
        valid_labels = []
        flag = True

        for batch_i in range(1, n_batches + 1):
            batch_filename = 'cifar10_preprocess_batch_' + str(batch_i) + '.p'

            if isfile(batch_filename):
                print(batch_filename + ' already exists')
                flag = False
            else:
                features, labels = self.load_batch(batch_i)
                index_of_validation = int(len(features) * valid_ratio)

                self.save_preprocessed_data(features[:-index_of_validation], labels[:-index_of_validation], batch_filename)

                valid_features.extend(features[-index_of_validation:])
                valid_labels.extend(labels[-index_of_validation:])

        if flag:
            self.save_preprocessed_data(np.array(valid_features), np.array(valid_labels),
                                        'cifar10_preprocess_validation.p')

        # load the test dataset
        with open(self.path + '/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        test_filename = 'cifar10_preprocess_testing.p'

        if isfile(test_filename):
            print(test_filename + ' already exists')
        else:
            # preprocess the testing data
            test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            test_labels = batch['labels']

            # Preprocess and Save all testing data
            self.save_preprocessed_data(np.array(test_features), np.array(test_labels), test_filename)

    def batch_features_labels(self, features, labels, batch_size):
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

    def load_preprocess_training_batch(self, batch_id, batch_size, scale_to_imagenet=False):
        filename = 'cifar10_preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        if scale_to_imagenet:
            tmpFeatures = []

            for feature in features:
                tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
                tmpFeatures.append(tmpFeature)

            features = tmpFeatures

        return self.batch_features_labels(features, labels, batch_size)

    def load_valid_set(self):
        valid_features, valid_labels = pickle.load(open('cifar10_preprocess_validation.p', mode='rb'))
        tmpValidFeatures = self.convert_to_imagenet_size(valid_features)

        return tmpValidFeatures, valid_labels
