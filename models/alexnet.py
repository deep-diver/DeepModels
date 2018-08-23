from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class AlexNet(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input, options=None):
        # 1st
        self.conv1 = conv2d(input, num_outputs=96,
                    kernel_size=[11,11], stride=4, padding="VALID",
                    activation_fn=tf.nn.relu)
        self.lrn1 = tf.nn.local_response_normalization(self.conv1, bias=2, alpha=0.0001,beta=0.75)
        self.pool1 = max_pool2d(self.lrn1, kernel_size=[3,3], stride=2)

        # 2nd
        self.conv2 = conv2d(self.pool1, num_outputs=256,
                    kernel_size=[5,5], stride=1, padding="VALID",
                    biases_initializer=tf.ones_initializer(),
                    activation_fn=tf.nn.relu)
        self.lrn2 = tf.nn.local_response_normalization(self.conv2, bias=2, alpha=0.0001, beta=0.75)
        self.pool2 = max_pool2d(self.lrn2, kernel_size=[3,3], stride=2)

        #3rd
        self.conv3 = conv2d(self.pool2, num_outputs=384,
                    kernel_size=[3,3], stride=1, padding="VALID",
                    activation_fn=tf.nn.relu)

        #4th
        self.conv4 = conv2d(self.conv3, num_outputs=384,
                    kernel_size=[3,3], stride=1, padding="VALID",
                    biases_initializer=tf.ones_initializer(),
                    activation_fn=tf.nn.relu)

        #5th
        self.conv5 = conv2d(self.conv4, num_outputs=256,
                    kernel_size=[3,3], stride=1, padding="VALID",
                    biases_initializer=tf.ones_initializer(),
                    activation_fn=tf.nn.relu)
        self.pool5 = max_pool2d(self.conv5, kernel_size=[3,3], stride=2)

        #6th
        self.flat = flatten(self.pool5)
        self.fcl1 = fully_connected(self.flat, num_outputs=4096,
                                biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
        self.dr1 = tf.nn.dropout(self.fcl1, 0.5)

        #7th
        self.fcl2 = fully_connected(self.dr1, num_outputs=4096,
                                biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
        self.dr2 = tf.nn.dropout(self.fcl2, 0.5)

        #output
        self.out = fully_connected(self.dr2, num_outputs=self.num_classes, activation_fn=None)
        return self.out
