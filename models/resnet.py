rom models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class ResNet(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input, options=None):
        if options is None:
            raise TypeError        

        # 18, 34, 50, 101, 152
        model_type = options['model_type']
        self.model_type = model_type        

        with tf.variable_scope('conv1'):
            conv1 = conv2d(input, num_outputs=64,
                            kernel_size=[7,7], stride=2, padding='SAME',
                            activation_fn=tf.nn.relu)

        with tf.variable_scope('conv2'):
            conv2 = max_pool2d(conv1, kernel_size=[3,3], stride=2, padding='SAME')
            conv2 = residual_block(conv2, 64)
            conv2 = residual_block(conv2, 64)
            conv2 = residual_block(conv2, 64)

        with tf.variable_scope('conv3'):
            conv3 = residual_block(conv2, 128, pool=True)
            conv3 = residual_block(conv3, 128)
            conv3 = residual_block(conv3, 128)
            conv3 = residual_block(conv3, 128)
        
        with tf.variable_scope('conv4'):
            conv4 = residual_block(conv3, 256, pool=True)
            conv4 = residual_block(conv4, 256)
            conv4 = residual_block(conv4, 256)
            conv4 = residual_block(conv4, 256)
            conv4 = residual_block(conv4, 256)
            conv4 = residual_block(conv4, 256)

        with tf.variable_scope('conv5'):
            conv5 = residual_block(conv4, 512, pool=True)
            conv5 = residual_block(conv5, 512)
            conv5 = residual_block(conv5, 512)

        with tf.variable_scope('before_final'):
            avg_pool = avg_pool2d(conv5, kernel_size=[3,3], stride=2, padding='SAME')
            flat = flatten(avg_pool)

        with tf.variable_scope('final'):
            self.final_out = fully_connected(flat, num_outputs=self.num_classes, activation_fn=None)

        return [self.final_out]

    def residual_block(input, num_outputs, kernel_size=[3,3], stride=2, pool=False):
        res = conv2
        out = input

        if pool:
            out = max_pool2d(conv1, kernel_size=[3,3], stride=2, padding='SAME')
            res = conv2d(out, num_outputs=num_outputs, 
                            kernel_size=[1,1], stride=[2,2], padding='SAME', 
                            activation_fn=tf.nn.relu)

        out = conv2d(out, num_outputs=num_outputs,
                        kernel_size=kernel_size, stride=stride, padding='SAME',
                        activation_fn=tf.nn.relu)
        out = conv2d(out, num_outputs=num_outputs,
                        kernel_size=kernel_size, stride=stride, padding='SAME',
                        activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(res)
        layers_concat.append(out)
        return tf.concat(layers_concat, 3)
