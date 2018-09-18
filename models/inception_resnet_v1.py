from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class Inception_ResnetV1(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input):
        with tf.variable_scope('stem'):
            stem = conv2d(input, num_outputs=32,
                          kernel_size=[3,3], stride=2, padding='VALID',
                          activation_fn=tf.nn.relu)
            stem = conv2d(stem, num_outputs=32,
                          kernel_size=[3,3], stride=1, padding='VALID',
                          activation_fn=tf.nn.relu)
            stem = conv2d(stem, num_outputs=64,
                          kernel_size=[3,3], stride=1, padding='SAME',
                          activation_fn=tf.nn.relu)
            stem = max_pool2d(stem, kernel_size=[3,3], stride=2, padding='VALID')
            stem = conv2d(stem, num_outputs=80,
                          kernel_size=[1,1], stride=1, padding='SAME',
                          activation_fn=tf.nn.relu)
            stem = conv2d(stem, num_outputs=192,
                          kernel_size=[3,3], stride=2, padding='VALID',
                          activation_fn=tf.nn.relu)
            stem = conv2d(stem, num_outputs=256,
                          kernel_size=[3,3], stride=2, padding='VALID',
                          activation_fn=tf.nn.relu)
            prev = stem

        with tf.variable_scope('inception_resnet_a'):
            for i in range(5):
                identity = prev

                branch_a = conv2d(prev, num_outputs=32,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)

                branch_b = conv2d(prev, num_outputs=32,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = conv2d(branch_b, num_outputs=32,
                                  kernel_size=[3,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)

                branch_c = conv2d(prev, num_outputs=32,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_c = conv2d(branch_c, num_outputs=32,
                                  kernel_size=[3,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_c = conv2d(branch_c, num_outputs=32,
                                  kernel_size=[3,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                layers_concat.append(branch_c)
                merge = tf.concat(layers_concat, 3)
                merge = conv2d(merge, num_outputs=256,
                               kernel_size=[1,1], stride=1, padding='SAME',
                               activation_fn=tf.nn.relu)

                prev = tf.nn.relu(merge + identity)

        with tf.variable_scope('reduction_a'):
            branch_a = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='VALID')

            branch_b = conv2d(prev, num_outputs=384,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)

            branch_c = conv2d(prev, num_outputs=192,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_c = conv2d(branch_c, num_outputs=192,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_c = conv2d(branch_c, num_outputs=256,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)
            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            layers_concat.append(branch_c)
            prev = tf.concat(layers_concat, 3)

        with tf.variable_scope('inception_resnet_b'):
            for i in range(10):
                identity = prev

                branch_a = conv2d(prev, num_outputs=128,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)

                branch_b = conv2d(prev, num_outputs=128,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = conv2d(branch_b, num_outputs=128,
                                  kernel_size=[1,7], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = conv2d(branch_b, num_outputs=128,
                                  kernel_size=[7,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                merge = tf.concat(layers_concat, 3)
                merge = conv2d(merge, num_outputs=896,
                               kernel_size=[1,1], stride=1, padding='SAME',
                               activation_fn=tf.nn.relu)

                prev = tf.nn.relu(merge + identity)

        with tf.variable_scope('reduction_b'):
            branch_a = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='VALID')

            branch_b = conv2d(prev, num_outputs=256,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_b = conv2d(branch_b, num_outputs=384,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)

            branch_c = conv2d(prev, num_outputs=256,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_c = conv2d(branch_c, num_outputs=256,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)

            branch_d = conv2d(prev, num_outputs=256,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_d = conv2d(branch_d, num_outputs=256,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_d = conv2d(branch_d, num_outputs=256,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)
            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            layers_concat.append(branch_c)
            layers_concat.append(branch_d)
            prev = tf.concat(layers_concat, 3)

        with tf.variable_scope('inception_resnet_c'):
            for i in range(5):
                identity = prev

                branch_a = conv2d(prev, num_outputs=192,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)

                branch_b = conv2d(prev, num_outputs=192,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = conv2d(branch_b, num_outputs=192,
                                  kernel_size=[1,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = conv2d(branch_b, num_outputs=192,
                                  kernel_size=[3,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                merge = tf.concat(layers_concat, 3)
                merge = conv2d(merge, num_outputs=1792,
                               kernel_size=[1,1], stride=1, padding='SAME',
                               activation_fn=tf.nn.relu)

                prev = tf.nn.relu(merge + identity)

        with tf.variable_scope('final'):
            prev = avg_pool2d(prev, kernel_size=[3,3], stride=2, padding='SAME')
            flat = flatten(prev)
            dr = tf.nn.dropout(flat, 0.8)
            self.out = fully_connected(dr, num_outputs=self.num_classes, activation_fn=None)

        return [self.out]
