from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

"""
    Implementation of Residual Network from ILSVRC 2015. The original architecture is invented by Kaiming He @Microsoft.

    The main technical contributions from this architecture are "identity mapping", and "making network very very deep"
"""
class ResNet(ImgClfModel):
    def __init__(self, model_type='50'):
        ImgClfModel.__init__(self, scale_to_imagenet=True, model_type=model_type)

    def create_model(self, input):
        raise NotImplementedError
