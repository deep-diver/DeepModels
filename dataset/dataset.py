from urllib.request import urlretrieve
from os.path import isfile, isdir

from tqdm import tqdm
import pickle
import numpy as np

import skimage
import skimage.io
import skimage.transform

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

class Dataset:
    def __init__(self, name, path, num_classes=-1, num_batch=1):
        self.name = name
        self.path = path
        self.num_batch = num_batch
        self.num_classes = num_classes

        self.download()
        self.preprocess_and_save_data()

    def download(self):
        raise NotImplementedError

    def preprocess_and_save_data(self):
        raise NotImplementedError

    def load_batch(self, batch_id):
        raise NotImplementedError

    def load_valid_set(self):
        raise NotImplementedError

    def convert_to_imagenet_size(self, images):
        tmp_images = []
        for image in images:
            tmp_image = skimage.transform.resize(image, (224, 224), mode='constant')
            tmp_images.append(tmp_image)

        return np.array(tmp_images)

    def save_preprocessed_data(self, features, labels, filename):
        labels = self.one_hot_encode(labels)
        pickle.dump((features, labels), open(filename, 'wb'))

    def one_hot_encode(self, x):
        encoded = np.zeros((len(x), self.num_classes))

        for idx, val in enumerate(x):
            encoded[idx][val] = 1

        return encoded

    def batch_features_labels(self, features, labels, batch_size):
        raise NotImplementedError

    def load_preprocess_training_batch(self, batch_id, batch_size, scale_to_imagenet=False):
        raise NotImplementedError
