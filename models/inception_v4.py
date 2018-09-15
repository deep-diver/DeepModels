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
        # STEM Network - separate into 3 parts by filter concatenation points
        with tf.variable_scope('stem'):
            stem_a = conv2d(input, num_outputs=32,
                            kernel_size=[3,3], stride=2, padding='VALID',
                            activation_fn=tf.nn.relu)
            stem_a = conv2d(stem_a, num_outputs=32,
                            kernel_size=[3,3], stride=1, padding='VALID',
                            activation_fn=tf.nn.relu)
            stem_a = conv2d(stem_a, num_outputs=64,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
            branch_a = max_pool2d(stem_a, kernel_size=[3,3], stride=2, padding='VALID')
            branch_b = conv2d(stem_a, num_outputs=96,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)
            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            stem_b = tf.concat(layers_concat, 3)

            branch_a = conv2d(stem_b, num_outputs=64,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_a = conv2d(branch_a, num_outputs=96,
                              kernel_size=[3,3], stride=1, padding='VALID',
                              activation_fn=tf.nn.relu)
            branch_b = conv2d(stem_b, num_outputs=64,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_b = conv2d(branch_b, num_outputs=64,
                              kernel_size=[7,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_b = conv2d(branch_b, num_outputs=64,
                              kernel_size=[1,7], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_b = conv2d(branch_b, num_outputs=96,
                              kernel_size=[3,3], stride=1, padding='VALID',
                              activation_fn=tf.nn.relu)
            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            stem_c = tf.concat(layers_concat, 3)

            branch_a = conv2d(stem_c, num_outputs=192,
                              kernel_size=[3,3], stride=1, padding='VALID',
                              activation_fn=tf.nn.relu)
            branch_b = max_pool2d(stem_c, kernel_size=[3,3], stride=2, padding='VALID')
            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            prev = tf.concat(layers_concat, 3)

            
