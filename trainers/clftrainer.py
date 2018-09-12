import time

import tensorflow as tf

from models.alexnet import AlexNet
from models.vgg import VGG
from models.googlenet import GoogLeNet
from models.inception_v2 import InceptionV2
# from models.inception_v3 import InceptionV3
from trainers.predefined_loss import *

class ClfTrainer:
    def __init__(self, clf_model, clf_dataset):
        self.clf_model = clf_model
        self.clf_dataset = clf_dataset

    def __run_train__(self, sess, input, output,
                        batch_i, batch_size,
                        cost_func, train_op,
                        scale_to_imagenet=False):

        total_loss = 0
        count = 0

        for batch_features, batch_labels in self.clf_dataset.get_training_batches_from_preprocessed(batch_i, batch_size, scale_to_imagenet):
            loss, _ = sess.run([cost_func, train_op],
                                feed_dict={input: batch_features,
                                           output: batch_labels})
            total_loss = total_loss + loss
            count = count + 1

        return total_loss/count

    def __run_accuracy_in_valid_set__(self, sess, input, output, accuracy, batch_size, scale_to_imagenet=False):
        valid_features, valid_labels = self.clf_dataset.get_valid_set(scale_to_imagenet)

        valid_acc = 0
        for batch_valid_features, batch_valid_labels in self.clf_dataset.get_batches_from(valid_features, valid_labels, batch_size):
            valid_acc += sess.run(accuracy,
                                feed_dict={input:batch_valid_features,
                                           output:batch_valid_labels})

        tmp_num = valid_features.shape[0]/batch_size
        return valid_acc/tmp_num

    def __train__(self, input, output,
                    cost_func, train_op, accuracy,
                    epochs, batch_size, save_model_path,
                    save_every_epoch=1):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__run_train__(sess,
                                              input, output,
                                              batch_i, batch_size,
                                              cost_func, train_op,
                                              self.clf_model.scale_to_imagenet)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')
                    print('Avg. Loss: {} '.format(loss), end='')

                    valid_acc = self.__run_accuracy_in_valid_set__(sess,
                                                                   input, output,
                                                                   accuracy, batch_size,
                                                                   self.clf_model.scale_to_imagenet)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

                if epoch % save_every_epoch == 0:
                    print('epoch: {} is saved...'.format(epoch+1))
                    saver = tf.train.Saver()
                    saver.save(sess, save_model_path, global_step=epoch+1, write_meta_graph=False)

    def __get_simple_losses_and_accuracy__(self, out_layers, output, learning_rate, options=None):
        is_loss_weights_considered = False
        if options is not None and \
           'loss_weights' in options and \
           len(options['loss_weights']) is len(out_layers):
            is_loss_weights_considered = True

        aux_cost_sum = 0
        if is_loss_weights_considered:
            for i in range(len(out_layers) - 1):
                aux_out_layer = out_layers[i]
                aux_cost = tf.losses.softmax_cross_entropy(output, aux_out_layer, reduction=tf.losses.Reduction.MEAN) 
                aux_cost_sum += aux_cost * options['loss_weights'][i]

        final_out_layer = out_layers[len(out_layers)-1]
        cost = tf.losses.softmax_cross_entropy(output, final_out_layer, reduction=tf.losses.Reduction.MEAN) 

        if is_loss_weights_considered:
            cost = cost * options['loss_weights'][len(out_layers)-1]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(cost+aux_cost_sum)
        train_op = optimizer.apply_gradients(gradients)

        correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return cost, train_op, accuracy

    def __get_losses_and_accuracy__(self, model, output, out_layers, learning_rate, options=None):
        if isinstance(model, AlexNet):
            return get_alexnet_trainer(output, out_layers, learning_rate)
        elif isinstance(model, VGG):
            return get_vgg_trainer(output, out_layers, learning_rate)
        elif isinstance(model, GoogLeNet):
            return get_googlenet_trainer(output, out_layers, learning_rate)
        elif isinstance(model, ResNet):
            return get_resnet_trainer(output, out_layers, learning_rate)
        elif isinstance(model, InceptionV2):
            return get_inceptionv2_trainer(output, out_layers, learning_rate)
        # elif isinstance(model, inceptionV3):
        #     return get_inceptionv3_trainer(output, out_layers, learning_rate)
        else:
            return self.__get_simple_losses_and_accuracy__(out_layers, output, learning_rate, options)

    # default to use AdamOptimizer w/ softmax_cross_entropy_with_logits_v2
    def run_training(self,
                     epochs, batch_size, learning_rate,
                     save_model_to, save_every_epoch=1,
                     options=None):
        input, output = self.clf_model.set_dataset(self.clf_dataset)
        out_layers = self.clf_model.create_model(input)

        cost, train_op, accuracy = self.__get_losses_and_accuracy__(self.clf_model, output, out_layers, learning_rate)

        self.__train__(input, output,
                       cost, train_op, accuracy,
                       epochs, batch_size,
                       save_model_to, save_every_epoch)

    def resume_training_from_ckpt(self, epochs, batch_size, learning_rate, save_model_from, save_model_to, save_every_epoch=1, options=None):
        graph = tf.Graph()
        with graph.as_default():
            input, output = self.clf_model.set_dataset(self.clf_dataset)
            out_layers = self.clf_model.create_model(input)

            cost, train_op, accuracy = self.__get_losses_and_accuracy__(self.clf_model, output, out_layers, learning_rate)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, save_model_from)

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__run_train__(sess,
                                              input, output,
                                              batch_i, batch_size,
                                              cost, train_op,
                                              self.clf_model.scale_to_imagenet)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')
                    print('Avg. Loss: {} '.format(loss), end='')

                    valid_acc = self.__run_accuracy_in_valid_set__(sess,
                                                                   input, output,
                                                                   accuracy, batch_size,
                                                                   self.clf_model.scale_to_imagenet)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

                if epoch % save_every_epoch == 0:
                    print('epoch: {} is saved...'.format(epoch+1))
                    saver1 = tf.train.Saver()
                    saver1.save(sess, save_model_to, global_step=epoch+1, write_meta_graph=False)

    def run_transfer_learning(self,
                              epochs, batch_size, learning_rate,
                              save_model_from, save_model_to, save_every_epoch=1, options=None):
        graph = tf.Graph()
        with graph.as_default():
            input, output = self.clf_model.set_dataset(self.clf_dataset)
            out_layers = self.clf_model.create_model(input)

            cost, train_op, accuracy = self.__get_losses_and_accuracy__(self.clf_model, output, out_layers, learning_rate)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            var_list = []
            for var in tf.model_variables():
                if 'final' not in var.name:
                    var_list.append(var)

            saver = tf.train.Saver(var_list)
            saver.restore(sess, save_model_from)

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__run_train__(sess,
                                              input, output,
                                              batch_i, batch_size,
                                              cost, train_op,
                                              self.clf_model.scale_to_imagenet)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')
                    print('Avg. Loss: {} '.format(loss), end='')

                    valid_acc = self.__run_accuracy_in_valid_set__(sess,
                                                                   input, output,
                                                                   accuracy, batch_size,
                                                                   self.clf_model.scale_to_imagenet)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

                if epoch % save_every_epoch == 0:
                    print('epoch: {} is saved...'.format(epoch+1))
                    saver2 = tf.train.Saver()
                    saver2.save(sess, save_model_to, global_step=epoch+1, write_meta_graph=False)

    def run_testing(self,
                 data, save_model_from, options=None):
        graph = tf.Graph()
        with graph.as_default():
            input, _ = self.clf_model.set_dataset(self.clf_dataset)
            out_layers = self.clf_model.create_model(input)

            final_out_layer = out_layers[len(out_layers)-1]
            softmax_result = tf.nn.softmax(final_out_layer)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, save_model_from)

            results = sess.run(softmax_result,
                                feed_dict={input:data})

        return results
