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
        batch_num = tf.shape(input)[0]

        # STEM Network
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

        conv2d_4 = max_pool2d(self.pool_2, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)
        self.inception_3a = tf.concat(layers_concat, 3)

        # inception (3b)
        conv2d_1 = conv2d(self.inception_3a, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_3a, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=192,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_3a, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=96,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_3a, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_3b = tf.concat(layers_concat, 3)
        self.inception_3b_pool_1 = max_pool2d(self.inception_3b, kernel_size=[3,3], stride=2, padding='SAME')

        # inception (4a)
        conv2d_1 = conv2d(self.inception_3b_pool_1, num_outputs=192,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_3b_pool_1, num_outputs=96,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=208,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_3b_pool_1, num_outputs=16,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=48,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_3b_pool_1, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_4a = tf.concat(layers_concat, 3)

        # inception (4b)
        conv2d_1 = conv2d(self.inception_4a, num_outputs=160,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_4a, num_outputs=112,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=224,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_4a, num_outputs=24,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=64,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_4a, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_4b = tf.concat(layers_concat, 3)

        # inception (4c)
        conv2d_1 = conv2d(self.inception_4b, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_4b, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=256,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_4b, num_outputs=24,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=64,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_4b, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_4c = tf.concat(layers_concat, 3)

        # inception (4d)
        conv2d_1 = conv2d(self.inception_4c, num_outputs=112,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_4c, num_outputs=144,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=228,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_4c, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=64,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_4c, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=64,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_4d = tf.concat(layers_concat, 3)

        # inception (4e)
        conv2d_1 = conv2d(self.inception_4d, num_outputs=256,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_4d, num_outputs=160,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=320,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_4d, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=128,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_4d, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_4d = tf.concat(layers_concat, 3)
        self.inception_4d_pool_1 = max_pool2d(self.inception_4d, kernel_size=[3,3], stride=2, padding='SAME')

        # inception (5a)
        conv2d_1 = conv2d(self.inception_4d_pool_1, num_outputs=256,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_4d_pool_1, num_outputs=160,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=320,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_4d_pool_1, num_outputs=32,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=128,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_4d_pool_1, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        layers_concat = list()
        layers_concat.append(conv2d_1)
        layers_concat.append(conv2d_2)
        layers_concat.append(conv2d_3)
        layers_concat.append(conv2d_4)

        self.inception_5a = tf.concat(layers_concat, 3)

        # inception (5b)
        conv2d_1 = conv2d(self.inception_5a, num_outputs=384,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_2 = conv2d(self.inception_5a, num_outputs=192,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_2 = conv2d(conv2d_2, num_outputs=384,
                    kernel_size=[3,3], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_3 = conv2d(self.inception_5a, num_outputs=48,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        conv2d_3 = conv2d(conv2d_3, num_outputs=128,
                    kernel_size=[5,5], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)

        conv2d_4 = max_pool2d(self.inception_5a, kernel_size=[3,3], stride=1, padding='SAME')
        conv2d_4 = conv2d(conv2d_4, num_outputs=128,
                    kernel_size=[1,1], stride=1, padding="SAME",
                    activation_fn=tf.nn.relu)
        self.inception_5b = tf.concat(layers_concat, 3)

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
        self.final_avg_pool_1 = avg_pool2d(self.inception_5b, kernel_size=[7,7], stride=1, padding='SAME')
        self.final_dropout = tf.nn.dropout(self.final_avg_pool_1, 0.4)
        self.final_flat = flatten(self.final_dropout)
        self.final_out = fully_connected(self.final_flat, num_outputs=self.num_classes, activation_fn=None)

        return self.aux_1_out, self.aux_2_out, self.final_out

    def load_pretrained_model(self, save_model_from, options):
        raise NotImplementedError
