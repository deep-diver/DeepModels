from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class GoogLeNet(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input, options=None):
        # STEM Network
        with tf.variable_scope('stem'):
            self.conv2d_1 = conv2d(input, num_outputs=64,
                        kernel_size=[7,7], stride=2, padding="SAME",
                        activation_fn=tf.nn.relu)
            self.pool_1 = max_pool2d(self.conv2d_1, kernel_size=[3,3], stride=2, padding='SAME')
            self.lrn_1 = tf.nn.local_response_normalization(self.pool_1, bias=2, alpha=0.0001,beta=0.75)

            self.conv2d_2 = conv2d(self.lrn_1, num_outputs=64,
                                    kernel_size=[1,1], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)
            self.conv2d_3 = conv2d(self.conv2d_2, num_outputs=192,
                                    kernel_size=[3,3], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)
            self.lrn_2 = tf.nn.local_response_normalization(self.conv2d_3, bias=2, alpha=0.0001,beta=0.75)
            self.pool_2 = max_pool2d(self.lrn_2, kernel_size=[3,3], stride=2, padding='SAME')

        # Inception 3
        # a, b
        inception_3_nums = {
            'conv2d_1'  : [64 , 128],
            'conv2d_2_1': [96 , 128],
            'conv2d_2_2': [128, 192],
            'conv2d_3_1': [16 ,  32],
            'conv2d_3_2': [32 ,  96],
            'conv2d_4'  : [32 ,  64]
        }

        with tf.variable_scope('inception_3'):
            prev = self.pool_2
            for i in range(2):
                conv2d_1_kernels    = inception_3_nums['conv2d_1'][i]
                conv2d_2_1_kernels  = inception_3_nums['conv2d_2_1'][i]
                conv2d_2_2_kernels  = inception_3_nums['conv2d_2_2'][i]
                conv2d_3_1_kernels  = inception_3_nums['conv2d_3_1'][i]
                conv2d_3_2_kernels  = inception_3_nums['conv2d_3_2'][i]
                conv2d_4_kernels    = inception_3_nums['conv2d_4'][i]

                conv2d_1 = conv2d(prev, num_outputs=conv2d_1_kernels,
                                    kernel_size=[1,1], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)

                conv2d_2 = conv2d(prev, num_outputs=conv2d_2_1_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)
                conv2d_2 = conv2d(conv2d_2, num_outputs=conv2d_2_2_kernels,
                            kernel_size=[3,3], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                conv2d_3 = conv2d(prev, num_outputs=conv2d_3_1_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)
                conv2d_3 = conv2d(conv2d_3, num_outputs=conv2d_3_2_kernels,
                            kernel_size=[5,5], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                conv2d_4 = max_pool2d(prev, kernel_size=[3,3], stride=1, padding='SAME')
                conv2d_4 = conv2d(conv2d_4, num_outputs=conv2d_4_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                layers_concat = list()
                layers_concat.append(conv2d_1)
                layers_concat.append(conv2d_2)
                layers_concat.append(conv2d_3)
                layers_concat.append(conv2d_4)
                prev = tf.concat(layers_concat, 3)

                if i is 0:
                    self.inception_3a = prev

            prev = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='SAME')
            self.inception_3b = prev

        # Inception (4)
        # a, b, c, d, e
        inception_4_nums = {
            'conv2d_1'  : [192, 160, 128, 112, 256],
            'conv2d_2_1': [96 , 112, 128, 144, 160],
            'conv2d_2_2': [208, 224, 256, 228, 320],
            'conv2d_3_1': [16 ,  24,  24,  32,  32],
            'conv2d_3_2': [48 ,  64,  64,  64, 128],
            'conv2d_4'  : [64 ,  64,  64,  64, 128]
        }

        with tf.variable_scope('inception_4'):
            for i in range(5):
                conv2d_1_kernels    = inception_4_nums['conv2d_1'][i]
                conv2d_2_1_kernels  = inception_4_nums['conv2d_2_1'][i]
                conv2d_2_2_kernels  = inception_4_nums['conv2d_2_2'][i]
                conv2d_3_1_kernels  = inception_4_nums['conv2d_3_1'][i]
                conv2d_3_2_kernels  = inception_4_nums['conv2d_3_2'][i]
                conv2d_4_kernels    = inception_4_nums['conv2d_4'][i]

                conv2d_1 = conv2d(prev, num_outputs=conv2d_1_kernels,
                                    kernel_size=[1,1], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)

                conv2d_2 = conv2d(prev, num_outputs=conv2d_2_1_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)
                conv2d_2 = conv2d(conv2d_2, num_outputs=conv2d_2_2_kernels,
                            kernel_size=[3,3], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                conv2d_3 = conv2d(prev, num_outputs=conv2d_3_1_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)
                conv2d_3 = conv2d(conv2d_3, num_outputs=conv2d_3_2_kernels,
                            kernel_size=[5,5], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                conv2d_4 = max_pool2d(prev, kernel_size=[3,3], stride=1, padding='SAME')
                conv2d_4 = conv2d(conv2d_4, num_outputs=conv2d_4_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                layers_concat = list()
                layers_concat.append(conv2d_1)
                layers_concat.append(conv2d_2)
                layers_concat.append(conv2d_3)
                layers_concat.append(conv2d_4)
                prev = tf.concat(layers_concat, 3)

                if i is 0:
                    self.inception_4a = prev
                elif i is 1:
                    self.inception_4b = prev
                elif i is 2:
                    self.inception_4c = prev
                elif i is 3:
                    self.inception_4d = prev

            prev = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='SAME')
            self.inception_4e = prev

        # Inception (5)
        # a, b
        inception_5_nums = {
            'conv2d_1'  : [256, 384],
            'conv2d_2_1': [160, 192],
            'conv2d_2_2': [320, 384],
            'conv2d_3_1': [32 ,  48],
            'conv2d_3_2': [128, 128],
            'conv2d_4'  : [128, 128]
        }

        with tf.variable_scope('inception_5'):
            for i in range(2):
                conv2d_1_kernels    = inception_5_nums['conv2d_1'][i]
                conv2d_2_1_kernels  = inception_5_nums['conv2d_2_1'][i]
                conv2d_2_2_kernels  = inception_5_nums['conv2d_2_2'][i]
                conv2d_3_1_kernels  = inception_5_nums['conv2d_3_1'][i]
                conv2d_3_2_kernels  = inception_5_nums['conv2d_3_2'][i]
                conv2d_4_kernels    = inception_5_nums['conv2d_4'][i]

                conv2d_1 = conv2d(prev, num_outputs=conv2d_1_kernels,
                                    kernel_size=[1,1], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)

                conv2d_2 = conv2d(prev, num_outputs=conv2d_2_1_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)
                conv2d_2 = conv2d(conv2d_2, num_outputs=conv2d_2_2_kernels,
                            kernel_size=[3,3], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                conv2d_3 = conv2d(prev, num_outputs=conv2d_3_1_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)
                conv2d_3 = conv2d(conv2d_3, num_outputs=conv2d_3_2_kernels,
                            kernel_size=[5,5], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                conv2d_4 = max_pool2d(prev, kernel_size=[3,3], stride=1, padding='SAME')
                conv2d_4 = conv2d(conv2d_4, num_outputs=conv2d_4_kernels,
                            kernel_size=[1,1], stride=1, padding="SAME",
                            activation_fn=tf.nn.relu)

                layers_concat = list()
                layers_concat.append(conv2d_1)
                layers_concat.append(conv2d_2)
                layers_concat.append(conv2d_3)
                layers_concat.append(conv2d_4)
                prev = tf.concat(layers_concat, 3)

                if i is 0:
                    self.inception_5a = prev

            self.inception_5b = prev

        with tf.variable_scope('final'):
            # Aux #1 output
            aux_avg_pool_1 = avg_pool2d(self.inception_4a, kernel_size=[5,5], stride=3, padding='SAME')
            aux_conv2d_1 = conv2d(aux_avg_pool_1, num_outputs=128,
                                    kernel_size=[1,1], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)
            aux_flat = flatten(aux_conv2d_1)
            aux_fcl_1 = fully_connected(aux_flat, num_outputs=1024, activation_fn=tf.nn.relu)
            aux_droupout_1 = tf.nn.dropout(aux_fcl_1, 0.7)
            self.aux_1_out = fully_connected(aux_droupout_1, num_outputs=1024, activation_fn=tf.nn.relu)

            # Aux #2 output
            aux_avg_pool_1 = avg_pool2d(self.inception_4d, kernel_size=[5,5], stride=3, padding='SAME')
            aux_conv2d_1 = conv2d(aux_avg_pool_1, num_outputs=128,
                                    kernel_size=[1,1], stride=1, padding="SAME",
                                    activation_fn=tf.nn.relu)
            aux_flat = flatten(aux_conv2d_1)
            aux_fcl_1 = fully_connected(aux_flat, num_outputs=1024, activation_fn=tf.nn.relu)
            aux_droupout_1 = tf.nn.dropout(aux_fcl_1, 0.7)
            self.aux_2_out = fully_connected(aux_droupout_1, num_outputs=1024, activation_fn=tf.nn.relu)

            # Final output
            self.final_avg_pool_1 = avg_pool2d(prev, kernel_size=[7,7], stride=1, padding='SAME')
            self.final_dropout = tf.nn.dropout(self.final_avg_pool_1, 0.4)
            self.final_flat = flatten(self.final_dropout)
            self.final_out = fully_connected(self.final_flat, num_outputs=self.num_classes, activation_fn=None)

        return [self.aux_1_out, self.aux_2_out, self.final_out]
