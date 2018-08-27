from models.imgclfmodel import ImgClfModel
from dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class AlexNet(ImgClfModel):
    def __init__(self):
        ImgClfModel.__init__(self, scale_to_imagenet=True)

    def create_model(self, input, options=None):
        # 1st
        with tf.name_scope('group1') as group1_scope:
            self.conv1 = conv2d(input, num_outputs=96,
                        kernel_size=[11,11], stride=4, padding="VALID",
                        activation_fn=tf.nn.relu)
            self.lrn1 = tf.nn.local_response_normalization(self.conv1, bias=2, alpha=0.0001,beta=0.75)
            self.pool1 = max_pool2d(self.lrn1, kernel_size=[3,3], stride=2)

            tf.identity(self.conv1, 'conv2d_1'  )
            tf.identity(self.lrn1 , 'lrn_1'     )
            tf.identity(self.pool1, 'max_pool_1')

        # 2nd
        with tf.name_scope('group2') as group2_scope:
            self.conv2 = conv2d(self.pool1, num_outputs=256,
                        kernel_size=[5,5], stride=1, padding="VALID",
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=tf.nn.relu)
            self.lrn2 = tf.nn.local_response_normalization(self.conv2, bias=2, alpha=0.0001, beta=0.75)
            self.pool2 = max_pool2d(self.lrn2, kernel_size=[3,3], stride=2)

            tf.identity(self.conv2, 'conv2d_1'  )
            tf.identity(self.lrn2 , 'lrn_1'     )
            tf.identity(self.pool2, 'max_pool_1')

        #3rd
        with tf.name_scope('group3') as group3_scope:
            self.conv3 = conv2d(self.pool2, num_outputs=384,
                        kernel_size=[3,3], stride=1, padding="VALID",
                        activation_fn=tf.nn.relu)

            tf.identity(self.conv3, 'conv2d_1')

        #4th
        with tf.name_scope('group4') as group4_scope:
            self.conv4 = conv2d(self.conv3, num_outputs=384,
                        kernel_size=[3,3], stride=1, padding="VALID",
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=tf.nn.relu)

            tf.identity(self.conv4, 'conv2d_1')

        #5th
        with tf.name_scope('group5') as group5_scope:
            self.conv5 = conv2d(self.conv4, num_outputs=256,
                        kernel_size=[3,3], stride=1, padding="VALID",
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=tf.nn.relu)
            self.pool5 = max_pool2d(self.conv5, kernel_size=[3,3], stride=2)

            tf.identity(self.conv5, 'conv2d_1'  )
            tf.identity(self.pool5, 'max_pool_1')

        #6th
        with tf.name_scope('fcl') as fcl_scope:
            self.flat = flatten(self.pool5)
            self.fcl1 = fully_connected(self.flat, num_outputs=4096,
                                    biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
            self.dr1 = tf.nn.dropout(self.fcl1, 0.5)

            #7th
            self.fcl2 = fully_connected(self.dr1, num_outputs=4096,
                                    biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
            self.dr2 = tf.nn.dropout(self.fcl2, 0.5)

            tf.identity(self.flat, 'flat' )
            tf.identity(self.fcl1, 'fcl_1')
            tf.identity(self.dr1 , 'dr_1' )
            tf.identity(self.fcl2, 'fcl_2')
            tf.identity(self.dr2 , 'dr_2' )

        #output
        with tf.name_scope('final') as final_scope:
            self.out = fully_connected(self.dr2, num_outputs=self.num_classes, activation_fn=None)
            tf.identity(self.out, 'out')

        return [self.out]

    def load_pretrained_model(self, save_model_from, options=None):
        with tf.Session() as sess:
            loader = tf.train.import_meta_graph(save_model_from + '.meta')

            self.input = tf.get_default_graph().get_tensor_by_name('input:0')
            self.output = tf.get_default_graph().get_tensor_by_name('output:0')

            # 1st
            self.conv1 = tf.get_default_graph().get_tensor_by_name('group1/conv2d_1:0')
            self.lrn1  = tf.get_default_graph().get_tensor_by_name('group1/lrn_1:0')
            self.pool1 = tf.get_default_graph().get_tensor_by_name('group1/max_pool_1:0')

            # 2nd
            self.conv2 = tf.get_default_graph().get_tensor_by_name('group2/conv2d_1:0')
            self.lrn2  = tf.get_default_graph().get_tensor_by_name('group2/lrn_1:0')
            self.pool2 = tf.get_default_graph().get_tensor_by_name('group2/max_pool_1:0')

            #3rd
            self.conv3 = tf.get_default_graph().get_tensor_by_name('group3/conv2d_1:0')

            #4th
            self.conv4 = tf.get_default_graph().get_tensor_by_name('group4/conv2d_1:0')

            #5th
            self.conv5 = tf.get_default_graph().get_tensor_by_name('group5/conv2d_1:0')
            self.pool5 = tf.get_default_graph().get_tensor_by_name('group5/max_pool_1:0')

            #6th
            self.flat = tf.get_default_graph().get_tensor_by_name('fcl/flat:0')
            self.fcl1 = tf.get_default_graph().get_tensor_by_name('fcl/fcl_1:0')
            self.dr1  = tf.get_default_graph().get_tensor_by_name('fcl/dr_1:0')

            #7th
            self.fcl2 = tf.get_default_graph().get_tensor_by_name('fcl/fcl_2:0')
            self.dr2  = tf.get_default_graph().get_tensor_by_name('fcl/dr_2:0')

            #output
            self.out = tf.get_default_graph().get_tensor_by_name('final/out:0')
