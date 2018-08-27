import argparse
import sys

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

from dataset.cifar10_dataset import Cifar10
from dataset.cifar100_dataset import Cifar100

from models.alexnet import AlexNet
from models.vgg import VGG
from trainers.clftrainer import ClfTrainer

learning_rate = 0.0001
epochs = 2
batch_size = 64

def test_alexnet_transfer_from_cifar10_to_cifar100(save_model_from, save_model_to):
    print('Testment #2')
    print('- AlexNet transfer training on CIFAR-100 dataset based on CIFAR-10 parameter')

    print('# Instantiating AlexNet...', end='')
    alexNet = AlexNet()
    print('done...')

    print('# load pre-trained model architecture...', end='')
    alexNet.load_pretrained_model(save_model_from)
    print('done...')

    print('# Preparing CIFAR-100 dataset...', end='')
    cifar100_dataset = Cifar100()
    print('done...')

    print('# Constructing a new fully connected layer for the last part...', end='')
    before_out = alexNet.dr2
    out = fully_connected(before_out, num_outputs=cifar100_dataset.num_classes, activation_fn=None)
    print('done...')

    with tf.name_scope("new_train"):
        output = tf.placeholder(tf.int32, [None, cifar100_dataset.num_classes], name='output_new')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=output), name='cost_new')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_new').minimize(cost)

        correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_new')

    trainer = ClfTrainer(alexNet, cifar100_dataset)
    trainer.transfer_learning(alexNet.input, output,
                              cost, optimizer, accuracy,
                              epochs, batch_size,
                              save_model_from, save_model_to)

def test_alexnet_cifar10_train_resume(save_model_from, save_model_to):
    print('Testment #2')
    print('- AlexNet resuming training on CIFAR-10 dataset from left-off')
    print('- This is not about transfer learning')

    print('# Preparing CIFAR-10 dataset...', end='')
    cifar10_dataset = Cifar10()
    print('done...')

    print('# Instantiating AlexNet...', end='')
    alexNet = AlexNet()
    print('done...')

    print('# Preparing a Trainer...', end='')
    trainer = ClfTrainer(alexNet, cifar10_dataset)
    print('done...')

    print('# Staring the training process...', end='')
    trainer.train_from_ckpt(epochs, batch_size, save_model_from, save_model_to)
    print('done...')

def test_alexnet_cifar10_train(save_model_to):
    print('Testment #1')
    print('- AlexNet training on CIFAR-10 dataset')
    print('- This is not about transfer learning nor resuming the left-off')

    print('# Preparing CIFAR-10 dataset...', end='')
    cifar10_dataset = Cifar10()
    print('done...')

    print('# Instantiating AlexNet...', end='')
    alexNet = AlexNet()
    print('done...')

    print('# Building AlexNet...')
    input, output = alexNet.set_dataset(cifar10_dataset)
    out_layer = alexNet.create_model(input)
    print('done...')

    print('# Defining Loss/Cost function and Optimizer...', end='')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=output))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    print('done...')

    print('# Defining Accuracy metric...', end='')
    correct_pred = tf.equal(tf.argmax(out_layer, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print('done...')

    print('# Preparing a Trainer...', end='')
    trainer = ClfTrainer(alexNet, cifar10_dataset)
    print('done...')

    print('# Staring the training process...', end='')
    trainer.train(input, output,
                  cost, optimizer, accuracy,
                  epochs, batch_size,
                  save_model_to)
    print('done...')

def main():
    # Testment #1
    # test_alexnet_cifar10_train('./alexnet_cifar10.ckpt')

    # Testment #2
    test_alexnet_cifar10_train_resume('./alexnet_cifar10.ckpt-3', './alexnet_cifar10.ckpt')

    # Testment #3
    # test_alexnet_transfer_from_cifar10_to_cifar100('/ckpt/alexnet_cifar10.ckpt-1', './alexnet_cifar100.ckpt')

if __name__ == "__main__":
    main()
