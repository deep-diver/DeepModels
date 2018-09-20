from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class DenseNet(ImgClfModel):
    # model_type = [121 | 169 | 201 \ 264]
    def __init__(self, model_type='121'):
        ImgClfModel.__init__(self, scale_to_imagenet=True, model_type=model_type)

    def create_model(self, input):
        raise NotImplementedError
