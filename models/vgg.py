from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class VGG(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input, options):
        if options is None:
            raise TypeError

        model_type = options['model_type']

        # LAYER GROUP #1
        group_1 = conv2d(input, num_outputs=64,
                    kernel_size=[3,3], stride=1, padding='SAME',
                    activation_fn=tf.nn.relu)

        if model_type == 'A-LRN':
            group_1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001,beta=0.75)

        if model_type != 'A' and model_type == 'A-LRN':
            group_1 = conv2d(group_1, num_outputs=64,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_1 = max_pool2d(group_1, kernel_size=[2,2], stride=2)

        # LAYER GROUP #2
        group_2 = conv2d(group_1, num_outputs=128,
                            kernel_size=[3, 3], padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type != 'A' and model_type == 'A-LRN':
            group_2 = conv2d(group_2, num_outputs=128,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)

        group_2 = max_pool2d(group_2, kernel_size=[2,2], stride=2)

        # LAYER GROUP #3
        group_3 = conv2d(group_2, num_outputs=256,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
        group_3 = conv2d(group_3, num_outputs=256,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type == 'C':
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'D' or model_type == 'E':
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'E':
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_3 = max_pool2d(group_3, kernel_size=[2,2], stride=2)

        # LAYER GROUP #4
        group_4 = conv2d(group_3, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
        group_4 = conv2d(group_4, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type == 'C':
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'D' or model_type == 'E':
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'E':
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_4 = max_pool2d(group_4, kernel_size=[2,2], stride=2)

        # LAYER GROUP #5
        group_5 = conv2d(group_4, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
        group_5 = conv2d(group_5, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type == 'C':
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'D' or model_type == 'E':
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'E':
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_5 = max_pool2d(group_5, kernel_size=[2,2], stride=2)

        # 1st FC 4096
        flat = flatten(group_5)
        fcl1 = fully_connected(flat, num_outputs=4096, activation_fn=tf.nn.relu)
        dr1 = tf.nn.dropout(fcl1, 0.5)

        # 2nd FC 4096
        fcl2 = fully_connected(dr1, num_outputs=4096, activation_fn=tf.nn.relu)
        dr2 = tf.nn.dropout(fcl2, 0.5)

        # 3rd FC 1000
        out = fully_connected(dr2, num_outputs=self.num_classes, activation_fn=None)

        return out
