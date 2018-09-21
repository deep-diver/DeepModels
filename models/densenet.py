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
    def __init__(self, model_type='121', k=32, theta=0.5):
        ImgClfModel.__init__(self, scale_to_imagenet=True, model_type=model_type)
        self.k = k
        self.theta = theta

    def create_model(self, input):
        k = self.k
        theta = self.theta

        with tf.variable_scope('initial_block'):
            conv = tf.layers.batch_normalization(input)
            conv = tf.nn.relu(conv)
            conv = conv2d(input, num_outputs=2*k,
                          kernel_size=[7,7], stride=2, padding='SAME',
                          activation_fn=None)

            pool = max_pool2d(conv, kernel_size=[3,3], stride=2, padding='SAME')
            prev_kernels = 2*k
            input_kernels = prev_kernels

        cur_layer = pool
        layers_concat = list()

        with tf.variable_scope('dense_block_1'):
            for i in range(6):
                cur_kernels = 4 * k
                bottlenect = tf.layers.batch_normalization(cur_layer)
                bottlenect = tf.nn.relu(bottlenect)
                bottlenect = conv2d(bottlenect, num_outputs=cur_kernels,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=None)
                bottlenect = tf.nn.dropout(bottlenect, 0.2)

                cur_kernels = input_kernels + (k * i)
                conv = tf.layers.batch_normalization(bottlenect)
                conv = tf.nn.relu(conv)
                conv = conv2d(conv, num_outputs=cur_kernels,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=None)
                conv = tf.nn.dropout(conv, 0.2)

                layers_concat.append(conv)
                cur_layer = tf.concat(layers_concat, 3)
                prev_kernels = cur_kernels

        with tf.variable_scope('transition_block_1'):
            bottlenect = tf.layers.batch_normalization(cur_layer)
            bottlenect = conv2d(bottlenect, num_outputs=int(prev_kernels*theta),
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            bottlenect = tf.nn.dropout(bottlenect, 0.2)

            pool = avg_pool2d(bottlenect, kernel_size=[2,2], stride=2, padding='SAME')
            prev_kernels = int(prev_kernels*theta)
            input_kernels = prev_kernels

        cur_layer = pool
        layers_concat = list()

        with tf.variable_scope('dense_block_2'):
            for i in range(12):
                cur_kernels = 4 * k
                bottlenect = tf.layers.batch_normalization(cur_layer)
                bottlenect = tf.nn.relu(bottlenect)
                bottlenect = conv2d(bottlenect, num_outputs=cur_kernels,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=None)
                bottlenect = tf.nn.dropout(bottlenect, 0.2)

                cur_kernels = input_kernels + (k * i)
                conv = tf.layers.batch_normalization(bottlenect)
                conv = tf.nn.relu(conv)
                conv = conv2d(conv, num_outputs=cur_kernels,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=None)
                conv = tf.nn.dropout(conv, 0.2)

                layers_concat.append(conv)
                cur_layer = tf.concat(layers_concat, 3)
                prev_kernels = cur_kernels

        with tf.variable_scope('transition_block_2'):
            bottlenect = tf.layers.batch_normalization(cur_layer)
            bottlenect = conv2d(bottlenect, num_outputs=int(prev_kernels*theta),
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            bottlenect = tf.nn.dropout(bottlenect, 0.2)

            pool = avg_pool2d(bottlenect, kernel_size=[2,2], stride=2, padding='SAME')
            prev_kernels = int(prev_kernels*theta)
            input_kernels = prev_kernels

        cur_layer = pool
        layers_concat = list()
        dense_block_3_iter = 24
        if self.model_type is "169":
            dense_block_3_iter = 32
        elif self.model_type is "201":
            dense_block_3_iter = 48
        elif self.model_type is "264":
            dense_block_3_iter = 64

        with tf.variable_scope('dense_block_3'):
            for i in range(dense_block_3_iter):
                cur_kernels = 4 * k
                bottlenect = tf.layers.batch_normalization(cur_layer)
                bottlenect = tf.nn.relu(bottlenect)
                bottlenect = conv2d(bottlenect, num_outputs=cur_kernels,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=None)
                bottlenect = tf.nn.dropout(bottlenect, 0.2)

                cur_kernels = input_kernels + (k * i)
                conv = tf.layers.batch_normalization(bottlenect)
                conv = tf.nn.relu(conv)
                conv = conv2d(conv, num_outputs=cur_kernels,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=None)
                conv = tf.nn.dropout(conv, 0.2)

                layers_concat.append(conv)
                cur_layer = tf.concat(layers_concat, 3)
                prev_kernels = cur_kernels

        with tf.variable_scope('transition_block_3'):
            bottlenect = tf.layers.batch_normalization(cur_layer)
            bottlenect = conv2d(bottlenect, num_outputs=int(prev_kernels*theta),
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            bottlenect = tf.nn.dropout(bottlenect, 0.2)

            pool = avg_pool2d(bottlenect, kernel_size=[2,2], stride=2, padding='SAME')
            prev_kernels = int(prev_kernels*theta)
            input_kernels = prev_kernels

        cur_layer = pool
        layers_concat = list()
        dense_block_4_iter = 16
        if self.model_type is "169" or self.model_type is "201":
            dense_block_4_iter = 32
        elif self.model_type is "264":
            dense_block_4_iter = 48

        with tf.variable_scope('dense_block_4'):
            for i in range(dense_block_4_iter):
                cur_kernels = 4 * k
                bottlenect = tf.layers.batch_normalization(cur_layer)
                bottlenect = tf.nn.relu(bottlenect)
                bottlenect = conv2d(bottlenect, num_outputs=cur_kernels,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=None)
                bottlenect = tf.nn.dropout(bottlenect, 0.2)

                cur_kernels = input_kernels + (k * i)
                conv = tf.layers.batch_normalization(bottlenect)
                conv = tf.nn.relu(conv)
                conv = conv2d(conv, num_outputs=cur_kernels,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=None)
                conv = tf.nn.dropout(conv, 0.2)

                layers_concat.append(conv)
                cur_layer = tf.concat(layers_concat, 3)
                prev_kernels = cur_kernels

        with tf.variable_scope('final'):
            pool = avg_pool2d(cur_layer, kernel_size=[7,7], stride=1, padding='SAME')
            flat = flatten(pool)
            self.out = fully_connected(flat, num_outputs=self.num_classes, activation_fn=None)

        return [self.out]
