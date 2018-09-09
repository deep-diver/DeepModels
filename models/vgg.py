from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

"""
    Implementation of VGGs from ILSVRC 2014. The original architecture is invented by VGG (Visual Geometry Group) @Oxford.
    This one didnt' win the ILSVRC 2014, but it took the 2nd place. It is very popular and well-known to lots of new comers in deep learning area.

    The main technical contributions from this architecture are "3x3 filters", and very simple architecture with deeper depth.
"""
class VGG(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    """
        types
        A : 11 weight layers
        A-LRN : 11 weight layers with Local Response Normalization
        B : 13 weight layers
        C : 16 weight layers with 1D conv layers
        D : 16 weight layers
        E : 19 weight layers
    """
    def create_model(self, input, options):
        if options is None:
            raise TypeError

        model_type = options['model_type']
        self.model_type = model_type

        self.group1 = []
        self.group2 = []
        self.group3 = []
        self.group4 = []
        self.group5 = []                        

        with tf.variable_scope('group1'):
            # LAYER GROUP #1
            group_1 = conv2d(input, num_outputs=64,
                        kernel_size=[3,3], stride=1, padding='SAME',
                        activation_fn=tf.nn.relu)
            self.group1.append(group_1)

            if model_type == 'A-LRN':
                group_1 = tf.nn.local_response_normalization(group_1,
                                                             bias=2, alpha=0.0001, beta=0.75)
                self.group1.append(group_1)

            if model_type != 'A' and model_type == 'A-LRN':
                group_1 = conv2d(group_1, num_outputs=64,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group1.append(group_1)

            group_1 = max_pool2d(group_1, kernel_size=[2,2], stride=2)
            self.group1.append(group_1)

        with tf.variable_scope('group2'):
            # LAYER GROUP #2
            group_2 = conv2d(group_1, num_outputs=128,
                                kernel_size=[3, 3], padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group2.append(group_2)

            if model_type != 'A' and model_type == 'A-LRN':
                group_2 = conv2d(group_2, num_outputs=128,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
                self.group2.append(group_2)

            group_2 = max_pool2d(group_2, kernel_size=[2,2], stride=2)
            self.group2.append(group_2)

        with tf.variable_scope('group3'):
            # LAYER GROUP #3
            group_3 = conv2d(group_2, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group3.append(group_3)
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group3.append(group_3)

            if model_type == 'C':
                group_3 = conv2d(group_3, num_outputs=256,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group3.append(group_3)

            if model_type == 'D' or model_type == 'E':
                group_3 = conv2d(group_3, num_outputs=256,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group3.append(group_3)

            if model_type == 'E':
                group_3 = conv2d(group_3, num_outputs=256,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group3.append(group_3)

            group_3 = max_pool2d(group_3, kernel_size=[2,2], stride=2)
            self.group3.append(group_3)

        with tf.variable_scope('group4'):
            # LAYER GROUP #4
            group_4 = conv2d(group_3, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group4.append(group_4)
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group4.append(group_4)

            if model_type == 'C':
                group_4 = conv2d(group_4, num_outputs=512,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group4.append(group_4)

            if model_type == 'D' or model_type == 'E':
                group_4 = conv2d(group_4, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group4.append(group_4)

            if model_type == 'E':
                group_4 = conv2d(group_4, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group4.append(group_4)

            group_4 = max_pool2d(group_4, kernel_size=[2,2], stride=2)
            self.group4.append(group_4)

        with tf.variable_scope('group5'):
            # LAYER GROUP #5
            group_5 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group5.append(group_5)
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group5.append(group_5)

            if model_type == 'C':
                group_5 = conv2d(group_5, num_outputs=512,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group5.append(group_5)

            if model_type == 'D' or model_type == 'E':
                group_5 = conv2d(group_5, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group5.append(group_5)

            if model_type == 'E':
                group_5 = conv2d(group_5, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group5.append(group_5)

            group_5 = max_pool2d(group_5, kernel_size=[2,2], stride=2)
            self.group5.append(group_5)

        with tf.variable_scope('fcl'):
            # 1st FC 4096
            self.flat = flatten(group_5)
            self.fcl1 = fully_connected(self.flat, num_outputs=4096, activation_fn=tf.nn.relu)
            self.dr1 = tf.nn.dropout(self.fcl1, 0.5)

            # 2nd FC 4096
            self.fcl2 = fully_connected(self.dr1, num_outputs=4096, activation_fn=tf.nn.relu)
            self.dr2 = tf.nn.dropout(self.fcl2, 0.5)

        with tf.variable_scope('final'):
            # 3rd FC 1000
            self.out = fully_connected(self.dr2, num_outputs=self.num_classes, activation_fn=None)

        return [self.out]