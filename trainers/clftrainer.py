import tensorflow as tf

class ClfTrainer:
    def __init__(self, clf_model, clf_dataset):
        self.clf_model = clf_model
        self.clf_dataset = clf_dataset

    def __train__(self, sess, input, output,
                        batch_i, batch_size,
                        cost_func, optimizer,
                        scale_to_imagenet=False):

        total_loss = 0
        count = 0

        for batch_features, batch_labels in self.clf_dataset.load_preprocess_training_batch(batch_i, batch_size, scale_to_imagenet):
            loss, _ = sess.run([cost_func, optimizer],
                                feed_dict={input: batch_features,
                                           output: batch_labels})
            total_loss = total_loss + loss
            count = count + 1

        return total_loss/count

    def __accuracy_in_valid_set__(self, sess, input, output, accuracy, batch_size):
        valid_feature, valid_labels = self.clf_dataset.load_valid_set()

        valid_acc = 0
        for batch_valid_features, batch_valid_labels in self.clf_dataset.batch_features_labels(valid_feature, valid_labels, batch_size):
            valid_acc += sess.run(accuracy,
                                feed_dict={input:batch_valid_features,
                                           output:batch_valid_labels})

        tmp_num = valid_feature.shape[0]/batch_size
        return valid_acc/tmp_num

    def train(self, input, output,
                    cost_func, optimizer, accuracy,
                    epochs, batch_size, save_model_path,
                    save_every_epoch=1):

        saver = tf.train.Saver()

        loss = tf.identity(cost_func, 'loss')
        accuracy = tf.identity(accuracy, 'accuracy')

        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())

            tf.add_to_collection('optimizer', optimizer)

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__train__(sess, input, output, batch_i, batch_size, cost_func, optimizer, True)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')
                    print('Avg. Loss: {} '.format(loss), end='')

                    valid_acc = self.__accuracy_in_valid_set__(sess, input, output, accuracy, batch_size)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

                if epoch % save_every_epoch == 0:
                    print('epoch: {} is saved...'.format(epoch+1))
                    saver.save(sess, save_model_path, global_step=epoch+1)

    def train_from_ckpt(self, epochs, batch_size, save_model_from, save_model_to, save_every_epoch=1):
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_from + '.meta')
            loader.restore(sess, save_model_from)
            saver = tf.train.Saver()

            input = loaded_graph.get_tensor_by_name('input:0')
            output = loaded_graph.get_tensor_by_name('output:0')
            cost_func = loaded_graph.get_tensor_by_name('loss:0')
            optimizer = tf.get_collection('optimizer')[0]
            accuracy = loaded_graph.get_tensor_by_name('accuracy:0')

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__train__(sess, input, output, batch_i, batch_size, cost_func, optimizer, True)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')
                    print('Avg. Loss: {} '.format(loss), end='')

                    valid_acc = self.__accuracy_in_valid_set__(sess, input, output, accuracy, batch_size)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

                if epoch % save_every_epoch == 0:
                    saver.save(sess, save_model_to, global_step=epoch+1)

    def transfer_learning(self, input, output,
                                cost_func, optimizer, accuracy,
                                epochs, batch_size,
                                save_model_from, save_model_to, save_every_epoch=1):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loader = tf.train.import_meta_graph(save_model_from + '.meta')
            loader.restore(sess, save_model_from)

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    loss = self.__train__(sess, input, output, batch_i, batch_size, cost_func, optimizer, True)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')
                    print('Avg. Loss: {} '.format(loss), end='')

                    valid_acc = self.__accuracy_in_valid_set__(sess, input, output, accuracy, batch_size)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

                if epoch % save_every_epoch == 0:
                    saver.save(sess, save_model_to, global_step=epoch+1)
