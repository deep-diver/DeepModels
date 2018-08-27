from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class GoogLeNet(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input, options):
        # STEM Network
        self.conv2d_1 = conv2d(input, num_outputs=64,
                    kernel_size=[7,7], stride=2, padding="SAME",
                    activation_fn=tf.nn.relu)
        self.pool_1 = max_pool2d(self.conv2d_1, kernel_size=[3,3], stride=2)
        self.lrn_1 = tf.nn.local_response_normalization(self.pool_1, bias=2, alpha=0.0001,beta=0.75)

        self.conv2d_2 = conv2d(input, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        self.conv2d_3 = conv2d(input, num_outputs=192,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        self.lrn_2 = tf.nn.local_response_normalization(self.conv2d_3, bias=2, alpha=0.0001,beta=0.75)
        self.pool_2 = max_pool2d(self.lrn_2, kernel_size=[3,3], stride=2)

        # Inception (3a)
        conv2d_1 = conv2d(self.pool_2, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.pool_2, num_outputs=96,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=128,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.pool_2, num_outputs=16,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=32,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.pool_2, kernel_size=[3,3], stride=1)
        conv2d_4 = conv2d(conv2d_4, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        self.inception_1 = tf.concat(3, [conv2d_1, conv2d_2, conv2d_3, conv2d_4])

        # inception (3b)
        conv2d_1 = conv2d(self.inception_1, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_1, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=192,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_1, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=96,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_1, kernel_size=[3,3], stride=1)
        conv2d_4 = conv2d(conv2d_4, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        self.inception_2 = tf.concat(3, [conv2d_1, conv2d_2, conv2d_3, conv2d_4])
        self.inception_2_pool_1 = max_pool2d(self.inception_2, kernel_size=[3,3], stride=2)

        # inception (4a)

    def load_pretrained_model(self, save_model_from, options):
        raise NotImplementedError
