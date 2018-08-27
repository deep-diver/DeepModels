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

        with tf.name_scope('group1') as group1_scope:
            # LAYER GROUP #1
            group_1 = conv2d(input, num_outputs=64,
                        kernel_size=[3,3], stride=1, padding='SAME',
                        activation_fn=tf.nn.relu)
            self.group1_conv2d_1 = group_1
            tf.identity(group_1, 'conv2d_1')

            if model_type == 'A-LRN':
                group_1 = tf.nn.local_response_normalization(conv1,
                                                             bias=2, alpha=0.0001, beta=0.75)
                self.group1_lrn_1 = group_1
                tf.identity(group_1, 'lrn_1')

            if model_type != 'A' and model_type == 'A-LRN':
                group_1 = conv2d(group_1, num_outputs=64,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group1_conv2d_2 = group_1
                tf.identity(group_1, 'conv2d_2')

            group_1 = max_pool2d(group_1, kernel_size=[2,2], stride=2)
            self.group1_max_pool_1 = group_1
            tf.identity(group_1, 'max_pool_1')

        with tf.name_scope('group2') as group2_scope:
            # LAYER GROUP #2
            group_2 = conv2d(group_1, num_outputs=128,
                                kernel_size=[3, 3], padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group2_conv2d_1 = group_2
            tf.identity(group_2, 'conv2d_1')

            if model_type != 'A' and model_type == 'A-LRN':
                group_2 = conv2d(group_2, num_outputs=128,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
                self.group2_conv2d_2 = group_2
                tf.identity(group_2, 'conv2d_2')

            group_2 = max_pool2d(group_2, kernel_size=[2,2], stride=2)
            self.group2_max_pool_1 = group_2
            tf.identity(group_2, 'max_pool_1')

        with tf.name_scope('group3') as group3_scope:
            # LAYER GROUP #3
            group_3 = conv2d(group_2, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group3_conv2d_1 = group_3
            tf.identity(group_3, 'conv2d_1')
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group3_conv2d_2 = group_3
            tf.identity(group_3, 'conv2d_2')

            if model_type == 'C':
                group_3 = conv2d(group_3, num_outputs=256,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group3_conv1d_1 = group_3
                tf.identity(group_3, 'conv1d_1')

            if model_type == 'D' or model_type == 'E':
                group_3 = conv2d(group_3, num_outputs=256,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group3_conv2d_3 = group_3
                tf.identity(group_3, 'conv2d_3')

            if model_type == 'E':
                group_3 = conv2d(group_3, num_outputs=256,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group3_conv2d_4 = group_3
                tf.identity(group_3, 'conv2d_4')

            group_3 = max_pool2d(group_3, kernel_size=[2,2], stride=2)
            self.group3_max_pool_1 = group_3
            tf.identity(group_3, 'max_pool_1')

        with tf.name_scope('group4') as group4_scope:
            # LAYER GROUP #4
            group_4 = conv2d(group_3, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group4_conv2d_1 = group_4
            tf.identity(group_4, 'conv2d_1')
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group4_conv2d_2 = group_4
            tf.identity(group_4, 'conv2d_2')

            if model_type == 'C':
                group_4 = conv2d(group_4, num_outputs=512,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group4_conv1d_1 = group_4
                tf.identity(group_4, 'conv1d_1')

            if model_type == 'D' or model_type == 'E':
                group_4 = conv2d(group_4, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group4_conv2d_3 = group_4
                tf.identity(group_4, 'conv2d_3')

            if model_type == 'E':
                group_4 = conv2d(group_4, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group4_conv2d_4 = group_4
                tf.identity(group_4, 'conv2d_4')

            group_4 = max_pool2d(group_4, kernel_size=[2,2], stride=2)
            self.group4_max_pool_1 = group_4
            tf.identity(group_4, 'max_pool_1')

        with tf.name_scope('group5') as group5_scope:
            # LAYER GROUP #5
            group_5 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group5_conv2d_1 = group_5
            tf.identity(group_5, 'conv2d_1')
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)
            self.group5_conv2d_2 = group_5
            tf.identity(group_5, 'conv2d_2')

            if model_type == 'C':
                group_5 = conv2d(group_5, num_outputs=512,
                                    kernel_size=[1,1], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group5_conv1d_1 = group_5
                tf.identity(group_5, 'conv1d_1')

            if model_type == 'D' or model_type == 'E':
                group_5 = conv2d(group_5, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group5_conv2d_3 = group_5
                tf.identity(group_5, 'conv2d_3')

            if model_type == 'E':
                group_5 = conv2d(group_5, num_outputs=512,
                                    kernel_size=[3,3], stride=1, padding='SAME',
                                    activation_fn=tf.nn.relu)
                self.group5_conv2d_4 = group_5
                tf.identity(group_5, 'conv2d_4')

            group_5 = max_pool2d(group_5, kernel_size=[2,2], stride=2)
            self.group5_max_pool_1 = group_5
            tf.identity(group_5, 'max_pool_1')

        with tf.name_scope('fcl') as fcl_scope:
            # 1st FC 4096
            flat = flatten(group_5)
            self.flat = flat
            tf.identity(flat, 'flat')

            fcl1 = fully_connected(flat, num_outputs=4096, activation_fn=tf.nn.relu)
            self.fcl_1 = fcl1
            tf.identity(fcl1, 'fcl_1')

            dr1 = tf.nn.dropout(fcl1, 0.5)
            self.dropout_1 = dr1
            tf.identity(dr1, 'dropout_1')

            # 2nd FC 4096
            fcl2 = fully_connected(dr1, num_outputs=4096, activation_fn=tf.nn.relu)
            self.fcl_2 = fcl2
            tf.identity(fcl2, 'fcl_2')

            dr2 = tf.nn.dropout(fcl2, 0.5)
            self.dropout_2 = dr2
            tf.identity(dr2, 'dropout_2')

        with tf.name_scope('final') as final_scope:
            tf.identity(dr2, 'before_out')

            # 3rd FC 1000
            out = fully_connected(dr2, num_outputs=self.num_classes, activation_fn=None)
            self.out = out
            tf.identity(out, 'out')

        return [out]

    def load_pretrained_model(self, save_model_from, options):
        model_type = options['model_type']
        self.model_type = model_type

        if self.model_type is None:
            print('model type is not set. please run create_model method first')
            raise TypeError

        with tf.Session() as sess:
            loader = tf.train.import_meta_graph(save_model_from + '.meta')

            self.input = tf.get_default_graph().get_tensor_by_name('input:0')
            self.output = tf.get_default_graph().get_tensor_by_name('output:0')

            # GROUP1
            self.group1_conv2d_1 = tf.get_default_graph().get_tensor_by_name('group1/conv2d_1:0')

            if self.model_type == 'A-LRN':
                self.group1_lrn_1 = tf.get_default_graph().get_tensor_by_name('group1/lrn_1:0')

            if self.model_type != 'A' and self.model_type == 'A-LRN':
                self.group1_conv2d_2 = tf.get_default_graph().get_tensor_by_name('group1/conv2d_2:0')

            self.group1_max_pool_1 = tf.get_default_graph().get_tensor_by_name('group1/max_pool_1:0')

            # LAYER GROUP #2
            self.group2_conv2d_1 = tf.get_default_graph().get_tensor_by_name('group2/conv2d_1:0')

            if self.model_type != 'A' and self.model_type == 'A-LRN':
                self.group2_conv2d_2 = tf.get_default_graph().get_tensor_by_name('group2/conv2d_2:0')

            self.group2_max_pool_1 = tf.get_default_graph().get_tensor_by_name('group2/max_pool_1:0')

            # LAYER GROUP #3
            self.group3_conv2d_1 = tf.get_default_graph().get_tensor_by_name('group3/conv2d_1:0')
            self.group3_conv2d_2 = tf.get_default_graph().get_tensor_by_name('group3/conv2d_2:0')

            if self.model_type == 'C':
                self.group3_conv1d_1 = tf.get_default_graph().get_tensor_by_name('group3/conv1d_1:0')

            if self.model_type == 'D' or self.model_type == 'E':
                self.group3_conv2d_3 = tf.get_default_graph().get_tensor_by_name('group3/conv2d_3:0')

            if self.model_type == 'E':
                self.group3_conv2d_4 = tf.get_default_graph().get_tensor_by_name('group3/conv2d_4:0')

            self.group3_max_pool_1 = tf.get_default_graph().get_tensor_by_name('group3/max_pool_1:0')

            # LAYER GROUP #4
            self.group4_conv2d_1 = tf.get_default_graph().get_tensor_by_name('group4/conv2d_1:0')
            self.group4_conv2d_2 = tf.get_default_graph().get_tensor_by_name('group4/conv2d_2:0')

            if self.model_type == 'C':
                self.group4_conv1d_1 = tf.get_default_graph().get_tensor_by_name('group4/conv1d_1:0')

            if self.model_type == 'D' or self.model_type == 'E':
                self.group4_conv2d_3 = tf.get_default_graph().get_tensor_by_name('group4/conv2d_3:0')

            if self.model_type == 'E':
                self.group4_conv2d_4 = tf.get_default_graph().get_tensor_by_name('group4/conv2d_4:0')

            self.group4_max_pool_1 = tf.get_default_graph().get_tensor_by_name('group4/max_pool_1:0')

            # LAYER GROUP #5
            self.group5_conv2d_1 = tf.get_default_graph().get_tensor_by_name('group5/conv2d_1:0')
            self.group5_conv2d_2 = tf.get_default_graph().get_tensor_by_name('group5/conv2d_2:0')

            if self.model_type == 'C':
                self.group5_conv1d_1 = tf.get_default_graph().get_tensor_by_name('group5/conv1d_1:0')

            if self.model_type == 'D' or self.model_type == 'E':
                self.group5_conv2d_3 = tf.get_default_graph().get_tensor_by_name('group5/conv2d_3:0')

            if self.model_type == 'E':
                self.group5_conv2d_4 = tf.get_default_graph().get_tensor_by_name('group5/conv2d_4:0')

            self.group5_max_pool_1 = tf.get_default_graph().get_tensor_by_name('group5/max_pool_1:0')

            # 1st FC 4096
            self.flat = tf.get_default_graph().get_tensor_by_name('fcl/flat:0')
            self.fcl_1 = tf.get_default_graph().get_tensor_by_name('fcl/fcl_1:0')
            self.dropout_1 = tf.get_default_graph().get_tensor_by_name('fcl/dropout_1:0')

            # 2nd FC 4096
            self.fcl_2 = tf.get_default_graph().get_tensor_by_name('fcl/fcl_2:0')
            self.dropout_2 = tf.get_default_graph().get_tensor_by_name('final/before_out:0')

            # 3rd FC 1000
            self.before_out = tf.get_default_graph().get_tensor_by_name('final/before_out:0')
            self.out = tf.get_default_graph().get_tensor_by_name('final/out:0')
