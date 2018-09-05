import time

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class ClfTrainer:
    def __init__(self, clf_model, clf_dataset):
        self.clf_model = clf_model
        self.clf_dataset = clf_dataset

    def __run_train__(self, sess, input, output,
                        batch_i, batch_size,
                        cost_func, optimizer,
                        scale_to_imagenet=False):

        total_loss = 0
        count = 0

        for batch_features, batch_labels in self.clf_dataset.get_training_batches_from_preprocessed(batch_i, batch_size, scale_to_imagenet):
            loss, _ = sess.run([cost_func, optimizer],
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
                    cost_func, optimizer, accuracy,
                    epochs, batch_size, save_model_path,
                    save_every_epoch=1):
        loss = tf.identity(cost_func, 'loss')
        accuracy = tf.identity(accuracy, 'accuracy')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            tf.add_to_collection('optimizer', optimizer)

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__run_train__(sess,
                                              input, output,
                                              batch_i, batch_size,
                                              cost_func, optimizer,
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

    # default to use AdamOptimizer w/ softmax_cross_entropy_with_logits_v2
    def run_training(self,
                     epochs, batch_size, learning_rate,
                     save_model_to, save_every_epoch=1,
                     options=None):
        input, output = self.clf_model.set_dataset(self.clf_dataset)
        out_layers = self.clf_model.create_model(input, options)

        # aux_softmax is not implemented yet
        final_out_layer = out_layers[len(out_layers)-1]
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_out_layer, labels=output))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.__train__(input, output,
                       cost, optimizer, accuracy,
                       epochs, batch_size,
                       save_model_to, save_every_epoch)

    def train_from_ckpt(self, epochs, batch_size, learning_rate, save_model_from, save_model_to, save_every_epoch=1, options=None):
        graph = tf.Graph()
        with graph.as_default():
            input, output = self.clf_model.set_dataset(self.clf_dataset)
            out_layers = self.clf_model.create_model(input, options)

            # aux_softmax is not implemented yet
            final_out_layer = out_layers[len(out_layers)-1]
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_out_layer, labels=output))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
                                              cost, optimizer,
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
            out_layers = self.clf_model.create_model(input, options)

            # aux_softmax is not implemented yet
            final_out_layer = out_layers[len(out_layers)-1]
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_out_layer, labels=output))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
                                              cost, optimizer,
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
