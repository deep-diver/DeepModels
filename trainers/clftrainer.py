import tensorflow as tf

class ClfTrainer:
    def __init__(self, clf_model, clf_dataset):
        self.clf_model = clf_model
        self.clf_dataset = clf_dataset

    def __train__(self, sess, input, output,
                        batch_i, batch_size,
                        cost_func, optimizer,
                        scale_to_imagenet=False):
        for batch_features, batch_labels in self.clf_dataset.load_preprocess_training_batch(batch_i, batch_size, scale_to_imagenet):
            _ = sess.run(optimizer,
                        feed_dict={input: batch_features,
                                   output: batch_labels})

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
                    epochs, batch_size, save_model_path):
        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = self.clf_dataset.num_batch

                for batch_i in range(1, n_batches + 1):
                    self.__train__(sess, input, output, batch_i, batch_size, cost_func, optimizer, True)
                    print('Epoch {:>2}, {} Batch {}: '.format(epoch + 1, self.clf_dataset.name, batch_i), end='')

                    valid_acc = self.__accuracy_in_valid_set__(sess, input, output, accuracy, batch_size)
                    print('Validation Accuracy {:.6f}'.format(valid_acc))

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)
