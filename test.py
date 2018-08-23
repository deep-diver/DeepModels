import argparse
import sys

import tensorflow as tf

from dataset.cifar10_dataset import Cifar10
from models.alexnet import AlexNet
from trainers.clftrainer import ClfTrainer

def main():
    learning_rate = 0.0001
    epochs = 1
    batch_size = 64
    save_model_path = './image_classification'

    print('hello world')
    cifar10_dataset = Cifar10()
    alexNet = AlexNet()

    input, output = alexNet.set_dataset(cifar10_dataset)
    model_final_layer = alexNet.create_model(input)
    logits = tf.identity(model_final_layer, name='logits')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=output), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam').minimize(cost)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    trainer = ClfTrainer(alexNet, cifar10_dataset)
    trainer.train(input, output, cost, optimizer, accuracy, epochs, batch_size, save_model_path)

if __name__ == "__main__":
    main()
