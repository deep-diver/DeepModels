from urllib.request import urlretrieve
from os.path import isfile, isdir
import os

from tqdm import tqdm
import tarfile
import pickle
import numpy as np

import skimage
import skimage.io
import skimage.transform

from dataset.dataset import Dataset
from dataset.dataset import DownloadProgress

class Cifar100(Dataset):
    def __init__(self):
        Dataset.__init__(self, name='Cifar-100', path='cifar-100-python',  num_classes=100, num_batch=5)
        self.width = 32
        self.height = 32

    def __download__(self):
        if not isfile('cifar-100-python.tar.gz'):
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-100 Dataset') as pbar:
                urlretrieve(
                    'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                    'cifar-100-python.tar.gz',
                    pbar.hook)
        else:
            print('cifar-100-python.tar.gz already exists')

        if not isdir(self.path):
            with tarfile.open('cifar-100-python.tar.gz') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar)
                tar.close()
        else:
            print('cifar100 dataset already exists')

    def __load_batch__(self, batch_id=1):
        with open(self.path + '/train', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')

        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['fine_labels']

        return features, labels

    def __preprocess_and_save_data__(self, valid_ratio=0.1):
        valid_features = []
        valid_labels = []
        flag = True

        features, labels = self.__load_batch__()

        index_of_validation = int(len(features) * valid_ratio)
        total_train_num = len(features) - int(len(features) * valid_ratio)
        n_batches = self.num_batch

        for batch_i in range(1, n_batches + 1):
            batch_filename = self.path + '/cifar100_preprocess_batch_' + str(batch_i) + '.p'

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
        self.save_preprocessed_data(np.array(valid_features), np.array(valid_labels), self.path + '/cifar100_preprocess_validation.p')

        # load the test dataset
        with open(self.path + '/test', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # preprocess the testing data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['fine_labels']

        # Preprocess and Save all testing data
        self.save_preprocessed_data(np.array(test_features), np.array(test_labels), self.path + '/cifar100_preprocess_testing.p')

    def get_batches_from(self, features, labels, batch_size):
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

    def get_training_batches_from_preprocessed(self, batch_id, batch_size, scale_to_imagenet=False):
        filename = self.path + '/cifar100_preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        if scale_to_imagenet:
            features = self.convert_to_imagenet_size(features)

        return self.get_batches_from(features, labels, batch_size)

    def get_valid_set(self, scale_to_imagenet=False):
        valid_features, valid_labels = pickle.load(open(self.path + '/cifar100_preprocess_validation.p', mode='rb'))

        if scale_to_imagenet:
            valid_features = self.convert_to_imagenet_size(valid_features)

        return valid_features, valid_labels
