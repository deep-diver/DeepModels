from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class InceptionV2(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=False)

    def create_model(self, input, options=None):
        # STEM Network
        with tf.variable_scope('stem'):
            self.conv2d_1 = conv2d(input, num_outputs=32,
                        kernel_size=[3,3], stride=2, padding="VALID",
                        activation_fn=tf.nn.relu)
            self.conv2d_2 = conv2d(self.conv2d_1, num_outputs=32,
                                    kernel_size=[3,3], stride=1, padding="VALID",
                                    activation_fn=tf.nn.relu)
            self.conv2d_3 = conv2d(self.conv2d_2, num_outputs=64,
                                    kernel_size=[3,3], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)
            self.pool_1 = max_pool2d(self.conv2d_3, kernel_size=[3,3], stride=2, padding='VALID')
            self.conv2d_4 = conv2d(self.pool_1, num_outputs=80,
                                    kernel_size=[3,3], stride=1, padding="VALID",
                                    activation_fn=tf.nn.relu)
            self.conv2d_5 = conv2d(self.conv2d_4, num_outputs=192,
                                    kernel_size=[3,3], stride=2, padding="VALID",
                                    activation_fn=tf.nn.relu)
            self.conv2d_6 = conv2d(self.conv2d_5, num_outputs=288,
                                    kernel_size=[3,3], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)    
        raise NotImplementedError