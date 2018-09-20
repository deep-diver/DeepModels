from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class Inception_ResnetV2(ImgClfModel):
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
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)
            branch_b = max_pool2d(stem_c, kernel_size=[3,3], stride=2, padding='VALID')
            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            prev = tf.concat(layers_concat, 3)

        with tf.variable_scope('inception_resnet_a'):
            for i in range(5):
                identity = prev

                branch_a = conv2d(prev, num_outputs=32,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_a = tf.layers.batch_normalization(branch_a)

                branch_b = conv2d(prev, num_outputs=32,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)
                branch_b = conv2d(branch_b, num_outputs=32,
                                  kernel_size=[3,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)

                branch_c = conv2d(prev, num_outputs=32,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_c = tf.layers.batch_normalization(branch_c)
                branch_c = conv2d(branch_c, num_outputs=48,
                                  kernel_size=[3,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_c = tf.layers.batch_normalization(branch_c)
                branch_c = conv2d(branch_c, num_outputs=64,
                                  kernel_size=[3,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_c = tf.layers.batch_normalization(branch_c)

                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                layers_concat.append(branch_c)
                merge = tf.concat(layers_concat, 3)
                merge = conv2d(merge, num_outputs=384,
                               kernel_size=[1,1], stride=1, padding='SAME',
                               activation_fn=tf.nn.relu)
                merge = tf.layers.batch_normalization(merge)

                prev = tf.nn.relu(merge + identity)
                prev = tf.layers.batch_normalization(prev)

        with tf.variable_scope('reduction_a'):
            branch_a = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='VALID')

            branch_b = conv2d(prev, num_outputs=384,
                              kernel_size=[3,3], stride=2, padding='VALID',
                              activation_fn=tf.nn.relu)

            branch_c = conv2d(prev, num_outputs=256,
                              kernel_size=[1,1], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_c = conv2d(branch_c, num_outputs=256,
                              kernel_size=[3,3], stride=1, padding='SAME',
                              activation_fn=tf.nn.relu)
            branch_c = conv2d(branch_c, num_outputs=384,
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

                branch_a = conv2d(prev, num_outputs=192,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_a = tf.layers.batch_normalization(branch_a)

                branch_b = conv2d(prev, num_outputs=128,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)
                branch_b = conv2d(branch_b, num_outputs=160,
                                  kernel_size=[1,7], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)
                branch_b = conv2d(branch_b, num_outputs=192,
                                  kernel_size=[7,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)

                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                merge = tf.concat(layers_concat, 3)
                merge = conv2d(merge, num_outputs=1152,
                               kernel_size=[1,1], stride=1, padding='SAME',
                               activation_fn=tf.nn.relu)
                merge = tf.layers.batch_normalization(merge)

                prev = tf.nn.relu(merge + identity)
                prev = tf.layers.batch_normalization(prev)

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
                branch_a = tf.layers.batch_normalization(branch_a)

                branch_b = conv2d(prev, num_outputs=192,
                                  kernel_size=[1,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)
                branch_b = conv2d(branch_b, num_outputs=224,
                                  kernel_size=[1,3], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)
                branch_b = tf.layers.batch_normalization(branch_b)
                branch_b = conv2d(branch_b, num_outputs=256,
                                  kernel_size=[3,1], stride=1, padding='SAME',
                                  activation_fn=tf.nn.relu)

                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                merge = tf.concat(layers_concat, 3)
                merge = conv2d(merge, num_outputs=2048,
                               kernel_size=[1,1], stride=1, padding='SAME',
                               activation_fn=tf.nn.relu)
                merge = tf.layers.batch_normalization(merge)

                prev = tf.nn.relu(merge + identity)
                prev = tf.layers.batch_normalization(prev)


        with tf.variable_scope('final'):
            prev = avg_pool2d(prev, kernel_size=[3,3], stride=2, padding='SAME')
            flat = flatten(prev)
            dr = tf.nn.dropout(flat, 0.8)
            self.out = fully_connected(dr, num_outputs=self.num_classes, activation_fn=None)

        return [self.out]
