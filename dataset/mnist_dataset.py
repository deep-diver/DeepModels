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

class Mnist(Dataset):
    def __init__(self):
        Dataset.__init__(self, name='MNIST', path='mnist_dataset', num_classes=10, num_batch=1)
        self.width = 28
        self.height = 28

    def download(self):
        raise NotImplementedError

    def preprocess_and_save_data(self):
        raise NotImplementedError

    def load_batch(self, batch_id):
        raise NotImplementedError

    def load_valid_set(self):
        raise NotImplementedError

    def batch_features_labels(self, features, labels, batch_size):
        raise NotImplementedError

    def load_preprocess_training_batch(self, batch_id, batch_size, scale_to_imagenet=False):
        raise NotImplementedError
