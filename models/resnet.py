from models.imgclfmodel import ImgClfModel
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
                            activation_fn=None)
            conv1 = tf.layers.batch_normalization(conv1)
            conv1 = tf.nn.relu(conv1)
            self.conv1 = conv1

        with tf.variable_scope('conv2'):
            conv2 = max_pool2d(conv1, kernel_size=[3,3], stride=2, padding='SAME')

            if model_type is "18" or model_type is "34":
                conv2 = self.repeat_residual_blocks(repeat=2, 
                                                    x=conv2, 
                                                    block=self.residual_block_a, 
                                                    num_outputs=[64,64], kernel_sizes=[[3,3], [3,3]],
                                                    pool=False)
                if model_type is "34":
                    conv2 = self.repeat_residual_blocks(repeat=2, 
                                                        x=conv2,
                                                        block=self.residual_block_a, 
                                                        num_outputs=[64], kernel_sizes=[[3,3] [3,3]],
                                                        pool=False)

            elif model_type is "50" or model_type is "101" or model_type is "152":
                conv2 = self.repeat_residual_blocks(repeat=3, 
                                                    x=conv2, 
                                                    block=self.residual_block_b, 
                                                    num_outputs=[64,64,256], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                    pool=False)
            self.conv2 = conv2

        with tf.variable_scope('conv3'):
            if model_type is "18" or model_type is "34":
                conv3 = self.repeat_residual_blocks(repeat=2, 
                                                    x=conv2,
                                                    block=self.residual_block_a, 
                                                    num_outputs=[128,128], kernel_sizes=[[3,3], [3,3]],
                                                    pool=True)   
                if model_type is "34":
                    conv3 = self.repeat_residual_blocks(repeat=2, 
                                                        x=conv3,
                                                        block=self.residual_block_a, 
                                                        num_outputs=[128,128], kernel_sizes=[[3,3], [3,3]],
                                                        pool=False)

            elif model_type is "50" or model_type is "101" or model_type is "152":
                conv3 = self.repeat_residual_blocks(repeat=4, 
                                                    x=conv2, 
                                                    block=self.residual_block_b, 
                                                    num_outputs=[128,128,512], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                    pool=True)
                if model_type is "152":
                    conv3 = self.repeat_residual_blocks(repeat=4, 
                                                        x=conv3, 
                                                        block=self.residual_block_b, 
                                                        num_outputs=[128,128,512], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                        pool=False)

            self.conv3 = conv3
        
        with tf.variable_scope('conv4'):
            if model_type is "18" or model_type is "34":
                conv4 = self.repeat_residual_blocks(repeat=2, 
                                                    x=conv3,
                                                    block=self.residual_block_a, 
                                                    num_outputs=[256,256], kernel_sizes=[[3,3], [3,3]],
                                                    pool=True)  
                if model_type is "34":
                    conv4 = self.repeat_residual_blocks(repeat=4, 
                                                        x=conv4,
                                                        block=self.residual_block_a, 
                                                        num_outputs=[256,256], kernel_sizes=[[3,3], [3,3]],
                                                        pool=False)  

            elif model_type is "50" or model_type is "101" or model_type is "152":
                conv4 = self.repeat_residual_blocks(repeat=6, 
                                                    x=conv3,
                                                    block=self.residual_block_b, 
                                                    num_outputs=[256,256,1024], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                    pool=True)

                if model_type is "101" or model_type is "152":
                    conv4 = self.repeat_residual_blocks(repeat=17, 
                                                        x=conv4,
                                                        block=self.residual_block_b, 
                                                        num_outputs=[256,256,1024], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                        pool=False)

                if model_type is "152":
                    conv4 = self.repeat_residual_blocks(repeat=77, 
                                                        x=conv4,
                                                        block=self.residual_block_b, 
                                                        num_outputs=[256,256,1024], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                        pool=False)

            self.conv4 = conv4

        with tf.variable_scope('conv5'):
            if model_type is "18" or model_type is "34":
                conv5 = self.repeat_residual_blocks(repeat=2, 
                                                    x=conv4,
                                                    block=self.residual_block_a, 
                                                    num_outputs=[512,512], kernel_sizes=[[3,3], [3,3]],
                                                    pool=True)
                if model_type is "34":
                    conv5 = self.repeat_residual_blocks(repeat=1,
                                                        x=conv5,
                                                        block=self.residual_block_a, 
                                                        num_outputs=[512,512], kernel_sizes=[[3,3], [3,3]],
                                                        pool=True)

            elif model_type is "50" or model_type is "101" or model_type is "152":
                conv5 = self.repeat_residual_blocks(repeat=3, 
                                                    x=conv4,
                                                    block=self.residual_block_b, 
                                                    num_outputs=[512,512,2048], kernel_sizes=[[1,1], [3,3], [1,1]],
                                                    pool=True)

            self.conv5 = conv5

        with tf.variable_scope('before_final'):
            avg_pool = avg_pool2d(conv5, kernel_size=[3,3], stride=2, padding='SAME')
            flat = flatten(avg_pool)
            self.flat = flat

        with tf.variable_scope('final'):
            self.final_out = fully_connected(flat, num_outputs=self.num_classes, activation_fn=None)

        return [self.final_out]

    def repeat_residual_blocks(self, repeat, x, block, num_outputs, kernel_sizes, pool=True):
        out = x

        # count 1
        if pool:
            out = block(x, num_outputs, kernel_sizes, pool=True)

        for i in range(repeat-1):
            out = block(x, num_outputs, kernel_sizes)

        return out

    # Applicable to 18, 34
    def residual_block_a(self, x, num_output, kernel_size=[[3,3], [3,3]], stride=1, pool=False):
        res = x
        out = x

        if pool:
            out = max_pool2d(out, kernel_size=[3,3], stride=2, padding='SAME')
            res = conv2d(res, num_outputs=num_output, 
                            kernel_size=[1,1], stride=[2,2], padding='SAME', 
                            activation_fn=None)
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)

        for i in range(len(kernel_sizes)):
            num_output = num_outputs[i]
            kernel_size = kernel_sizes[i]

            out = conv2d(out, num_outputs=num_output,
                            kernel_size=kernel_size, stride=stride, padding='SAME',
                            activation_fn=None)
            out = tf.layers.batch_normalization(out)

            if i < len(kernel_size)-1:
                out = tf.nn.relu(out)

        f_x = tf.nn.relu(out + res)
        return f_x

    # Applicable to 50, 101, 152
    def residual_block_b(self, x, num_outputs, kernel_sizes=[[1,1], [3,3], [1,1]], stride=1, pool=False):
        res = x
        out = x

        first_num_output = num_outputs[0]
        last_num_output = num_outputs[len(num_outputs)-1]

        if pool:
            out = max_pool2d(out, kernel_size=[3,3], stride=2, padding='SAME')
            res = conv2d(res, num_outputs=last_num_output, 
                            kernel_size=[1,1], stride=[2,2], padding='SAME', 
                            activation_fn=None)
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)
        else:
            res = conv2d(res, num_outputs=last_num_output, 
                            kernel_size=[1,1], stride=[1,1], padding='SAME', 
                            activation_fn=None)
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)            

        for i in range(len(kernel_sizes)):
            num_output = num_outputs[i]
            kernel_size = kernel_sizes[i]

            out = conv2d(out, num_outputs=num_output,
                            kernel_size=kernel_size, stride=stride, padding='SAME',
                            activation_fn=None)
            out = tf.layers.batch_normalization(out)

            if i < len(kernel_size)-1:
                out = tf.nn.relu(out)

        f_x = tf.nn.relu(out + res)
        return f_x
                    
