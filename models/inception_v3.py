from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class InceptionV3(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input):
        # STEM Network
        with tf.variable_scope('stem'):
            self.conv2d_1 = conv2d(input, num_outputs=32,
                        kernel_size=[3,3], stride=2, padding='VALID',
                        activation_fn=tf.nn.relu)
            self.conv2d_2 = conv2d(self.conv2d_1, num_outputs=32,
                                    kernel_size=[3,3], stride=1, padding='VALID',
                                    activation_fn=tf.nn.relu)
            self.conv2d_3 = conv2d(self.conv2d_2, num_outputs=64,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
            self.pool_1 = max_pool2d(self.conv2d_3, kernel_size=[3,3], stride=2, padding='VALID')

            self.conv2d_4 = conv2d(self.pool_1, num_outputs=80,
                                    kernel_size=[3,3], stride=1, padding='VALID',
                                    activation_fn=tf.nn.relu)
            self.conv2d_5 = conv2d(self.conv2d_4, num_outputs=192,
                                    kernel_size=[3,3], stride=2, padding='VALID',
                                    activation_fn=tf.nn.relu)
            self.pool_2 = max_pool2d(self.conv2d_5, kernel_size=[3,3], stride=2, padding='VALID')

            prev = self.pool_2

        # Inception (3) 1, 2, 3
        inception3_nums = {
            'branch_a'  : [64, 64, 64],
            'branch_b_1': [48, 48, 48],
            'branch_b_2': [64, 64, 64],
            'branch_c_1': [64, 64, 64],
            'branch_c_2': [96, 96, 96],
            'branch_c_3': [96, 96, 96],
            'branch_d'  : [32, 64, 64]
        }

        with tf.variable_scope('inception_3'):
            for i in range(3):
                branch_a_kernels    = inception3_nums['branch_a'][i]
                branch_b_1_kernels  = inception3_nums['branch_b_1'][i]
                branch_b_2_kernels  = inception3_nums['branch_b_2'][i]
                branch_c_1_kernels  = inception3_nums['branch_c_1'][i]
                branch_c_2_kernels  = inception3_nums['branch_c_2'][i]
                branch_c_3_kernels  = inception3_nums['branch_c_3'][i]
                branch_d_kernels    = inception3_nums['branch_d'][i]

                branch_a = conv2d(prev, num_outputs=branch_a_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')

                branch_b = conv2d(prev, num_outputs=branch_b_1_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')
                branch_b = conv2d(branch_b, num_outputs=branch_b_2_kernels,
                                  kernel_size=[3,3], stride=1, padding='SAME')

                branch_c = conv2d(prev, num_outputs=branch_c_1_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_2_kernels,
                                  kernel_size=[3,3], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_3_kernels,
                                  kernel_size=[3,3], stride=1, padding='SAME')

                branch_d = avg_pool2d(prev, kernel_size=[3,3], stride=1, padding='SAME')
                branch_d = conv2d(branch_d, num_outputs=branch_d_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')

                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                layers_concat.append(branch_c)
                layers_concat.append(branch_d)
                prev = tf.concat(layers_concat, 3)

        with tf.variable_scope('grid_reduction_a'):
            branch_a = conv2d(prev, num_outputs=384,
                              kernel_size=[3,3], stride=2, padding='VALID')

            branch_b = conv2d(prev, num_outputs=64,
                              kernel_size=[1,1], stride=1, padding='SAME')
            branch_b = conv2d(branch_b, num_outputs=96,
                              kernel_size=[3,3], stride=1, padding='SAME')
            branch_b = conv2d(branch_b, num_outputs=96,
                              kernel_size=[3,3], stride=2, padding='VALID')

            branch_c = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='VALID')

            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            layers_concat.append(branch_c)
            prev = tf.concat(layers_concat, 3)

        inception5_nums = {
            'branch_a'  : [192, 192, 192, 192],
            'branch_b_1': [128, 160, 160, 192],
            'branch_b_2': [128, 160, 160, 192],
            'branch_b_3': [192, 192, 192, 192],
            'branch_c_1': [128, 160, 160, 192],
            'branch_c_2': [128, 160, 160, 192],
            'branch_c_3': [128, 160, 160, 192],
            'branch_c_4': [128, 160, 160, 192],
            'branch_c_5': [192, 192, 192, 192],
            'branch_d'  : [192, 192, 192, 192]
        }

        with tf.variable_scope('inception_5'):
            for i in range(4):
                branch_a_kernels    = inception5_nums['branch_a'][i]
                branch_b_1_kernels  = inception5_nums['branch_b_1'][i]
                branch_b_2_kernels  = inception5_nums['branch_b_2'][i]
                branch_b_3_kernels  = inception5_nums['branch_b_3'][i]
                branch_c_1_kernels  = inception5_nums['branch_c_1'][i]
                branch_c_2_kernels  = inception5_nums['branch_c_2'][i]
                branch_c_3_kernels  = inception5_nums['branch_c_3'][i]
                branch_c_4_kernels  = inception5_nums['branch_c_4'][i]
                branch_c_5_kernels  = inception5_nums['branch_c_5'][i]
                branch_d_kernels    = inception5_nums['branch_d'][i]

                branch_a = conv2d(prev, num_outputs=branch_a_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')

                branch_b = conv2d(prev, num_outputs=branch_b_1_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')
                branch_b = conv2d(branch_b, num_outputs=branch_b_2_kernels,
                                  kernel_size=[1,7], stride=1, padding='SAME')
                branch_b = conv2d(branch_b, num_outputs=branch_b_3_kernels,
                                  kernel_size=[7,1], stride=1, padding='SAME')

                branch_c = conv2d(prev, num_outputs=branch_c_1_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_2_kernels,
                                  kernel_size=[7,7], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_3_kernels,
                                  kernel_size=[1,7], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_4_kernels,
                                  kernel_size=[7,1], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_5_kernels,
                                  kernel_size=[1,7], stride=1, padding='SAME')

                branch_d = avg_pool2d(prev, kernel_size=[3,3], stride=1, padding='SAME')
                branch_d = conv2d(branch_d, num_outputs=branch_d_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')

                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                layers_concat.append(branch_c)
                layers_concat.append(branch_d)
                prev = tf.concat(layers_concat, 3)

            self.aux = prev

        with tf.variable_scope('grid_reduction_b'):
            branch_a = conv2d(prev, num_outputs=192,
                              kernel_size=[1,1], stride=1, padding='SAME')
            branch_a = conv2d(branch_a, num_outputs=320,
                              kernel_size=[3,3], stride=2, padding='VALID')

            branch_b = conv2d(prev, num_outputs=192,
                              kernel_size=[1,1], stride=1, padding='SAME')
            branch_b = conv2d(branch_b, num_outputs=192,
                              kernel_size=[1,7], stride=1, padding='SAME')
            branch_b = conv2d(branch_b, num_outputs=192,
                              kernel_size=[7,1], stride=1, padding='SAME')
            branch_b = conv2d(branch_b, num_outputs=192,
                              kernel_size=[3,3], stride=2, padding='VALID')

            branch_c = max_pool2d(prev, kernel_size=[3,3], stride=2, padding='VALID')

            layers_concat = list()
            layers_concat.append(branch_a)
            layers_concat.append(branch_b)
            layers_concat.append(branch_c)
            prev = tf.concat(layers_concat, 3)

        inception2_nums = {
            'branch_a'  : [320, 320],
            'branch_b_1': [384, 384],
            'branch_b_2': [384, 384],
            'branch_b_3': [384, 384],
            'branch_c_1': [448, 448],
            'branch_c_2': [384, 384],
            'branch_c_3': [384, 384],
            'branch_d'  : [192, 192]
        }

        with tf.variable_scope('inception_2'):
            for i in range(2):
                branch_a_kernels    = inception5_nums['branch_a'][i]
                branch_b_1_kernels  = inception5_nums['branch_b_1'][i]
                branch_b_2_kernels  = inception5_nums['branch_b_2'][i]
                branch_b_3_kernels  = inception5_nums['branch_b_3'][i]
                branch_c_1_kernels  = inception5_nums['branch_c_1'][i]
                branch_c_2_kernels  = inception5_nums['branch_c_2'][i]
                branch_c_3_kernels  = inception5_nums['branch_c_3'][i]
                branch_d_kernels    = inception5_nums['branch_d'][i]

                branch_a = conv2d(prev, num_outputs=branch_a_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')

                branch_b = conv2d(prev, num_outputs=branch_b_1_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')
                branch_b = conv2d(branch_b, num_outputs=branch_b_2_kernels,
                                  kernel_size=[1,3], stride=1, padding='SAME')
                branch_b = conv2d(branch_b, num_outputs=branch_b_3_kernels,
                                  kernel_size=[3,1], stride=1, padding='SAME')

                branch_c = conv2d(prev, num_outputs=branch_c_1_kernels,
                                  kernel_size=[1,1], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_2_kernels,
                                  kernel_size=[1,3], stride=1, padding='SAME')
                branch_c = conv2d(branch_c, num_outputs=branch_c_3_kernels,
                                  kernel_size=[3,1], stride=1, padding='SAME')

                branch_d = max_pool2d(prev, kernel_size=[3,3], stride=1, padding='SAME')
                branch_d = conv2d(branch_d, num_outputs=branch_d_kernels,
                                   kernel_size=[1,1], stride=1, padding='SAME')

                layers_concat = list()
                layers_concat.append(branch_a)
                layers_concat.append(branch_b)
                layers_concat.append(branch_c)
                layers_concat.append(branch_d)
                prev = tf.concat(layers_concat, 3)

        with tf.variable_scope('final'):
            self.aux_pool = avg_pool2d(self.aux, kernel_size=[5,5], stride=3, padding='VALID')
            self.aux_conv = conv2d(self.aux_pool, num_outputs=128,
                                   kernel_size=[1,1], stride=1, padding='SAME')
            self.aux_flat = flatten(self.aux_conv)
            self.aux_bn = tf.layers.batch_normalization(self.aux_flat)
            self.aux_out = fully_connected(self.aux_flat, num_outputs=self.num_classes, activation_fn=None)

            self.final_pool =  avg_pool2d(prev, kernel_size=[2,2], stride=1, padding='VALID')
            self.final_dropout = tf.nn.dropout(self.final_pool, 0.8)
            self.final_flat = flatten(self.final_dropout)
            self.final_bn = tf.layers.batch_normalization(self.final_flat)
            self.final_out = fully_connected(self.final_flat, num_outputs=self.num_classes, activation_fn=None)

        return [self.aux_out, self.final_out]
