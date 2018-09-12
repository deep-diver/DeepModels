import tensorflow as tf

def get_alexnet_trainer(output, out_layers, learning_rate):
    final_out_layer = out_layers[len(out_layers)-1]
    cost = tf.losses.softmax_cross_entropy(output, final_out_layer, reduction=tf.losses.Reduction.MEAN)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(gradients)

    correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return cost, train_op, accuracy

def get_vgg_trainer(output, out_layers, learning_rate):
    final_out_layer = out_layers[len(out_layers)-1]
    cost = tf.losses.softmax_cross_entropy(output, final_out_layer, reduction=tf.losses.Reduction.MEAN)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(gradients)

    correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return cost, train_op, accuracy

def get_googlenet_trainer(output, out_layers, learning_rate):
    aux_cost_sum = 0
    for i in range(len(out_layers) - 1):
        aux_out_layer = out_layers[i]
        aux_cost = tf.losses.softmax_cross_entropy(output, aux_out_layer, reduction=tf.losses.Reduction.MEAN)
        aux_cost_sum += aux_cost * 0.3

    final_out_layer = out_layers[len(out_layers)-1]
    cost = tf.losses.softmax_cross_entropy(output, final_out_layer, reduction=tf.losses.Reduction.MEAN)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(cost+aux_cost_sum)
    train_op = optimizer.apply_gradients(gradients)

    correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return cost, train_op, accuracy

def get_resnet_trainer(output, out_layers, learning_rate):
    final_out_layer = out_layers[len(out_layers)-1]
    cost = tf.losses.softmax_cross_entropy(output, final_out_layer, reduction=tf.losses.Reduction.MEAN)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(gradients)

    correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return cost, train_op, accuracy

def get_inceptionv2_trainer(output, out_layers, learning_rate):
    return get_googlenet_trainer(output, out_layers, learning_rate)

def get_inceptionv3_trainer(output, out_layers, learning_rate):
    aux_cost_sum = 0
    for i in range(len(out_layers) - 1):
        aux_out_layer = out_layers[i]
        aux_cost = tf.losses.softmax_cross_entropy(output, aux_out_layer, reduction=tf.losses.Reduction.MEAN)
        aux_cost_sum += aux_cost * 0.3

    final_out_layer = out_layers[len(out_layers)-1]
    cost = tf.losses.softmax_cross_entropy(output, final_out_layer, reduction=tf.losses.Reduction.MEAN)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(cost+aux_cost_sum)
    train_op = optimizer.apply_gradients(gradients)

    correct_pred = tf.equal(tf.argmax(final_out_layer, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return cost, train_op, accuracy
