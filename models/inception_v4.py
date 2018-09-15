from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class InceptionV4(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input):
        raise NotImplementedError
