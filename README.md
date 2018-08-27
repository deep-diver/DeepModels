# DeepModels

Implementation of state-of-the-art deep learning models since 2012

There are 3 main parts, and each part corresponds to a class.
- Datasets
- Models
- Trainers

## Pre-defined Classes
#### Datasets
- **CIFAR-10**
  - [Information page link](https://www.cs.toronto.edu/~kriz/cifar.html)
  - 10 classes of image in size of 32x32
- **CIFAR-100**
  - [Information page link](https://www.cs.toronto.edu/~kriz/cifar.html)
  - 100 classes of image in size of 32x32  

#### Models
- **[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)**
- **[VGG](https://arxiv.org/pdf/1409.1556.pdf)** (model type = [A | A-LRN | B | C | D | E])
  - **A:** 11 layers
  - **A-LRN:** 11 layers with LRN (Local Response Normalization)
  - **B:** 13 layers
  - **C:** 13 layers with additional convolutional layer whose kernel size is 1x1
  - **D:** 16 layers (known as VGG16)
  - **E:** 19 layers (known as VGG19)
- **[Inception V1 (GoogLeNet)](https://arxiv.org/pdf/1409.4842.pdf)**

#### Trainers
- ClfTrainer: Trainer for image classification like ILSVRC

## Pre-trained accuracy (coming soon)
- AlexNet
- VGG
- Inception V1 (GoogLeNet)

## Example Usage Code Blocks
#### Define hyper-parameters
```python
learning_rate = 0.0001
epochs = 1
batch_size = 64
save_model_path = './current.ckpt'
save_model_to = './new.ckpt'
```

#### Train from nothing
```python
cifar10_dataset = Cifar10() # Choose the dataset to train on
alexNet = AlexNet() # Choose the model to train

# Get input and output
# - set_dataset(): input and output should be dependent upon the target dataset
#                  i.e) CIFAR-10's output is 10, CIFAR-100's output is 100
input, output = alexNet.set_dataset(cifar10_dataset)

# Retrieve the last layer from the model
# - tf.identity for the last layer is used to give an identity
# - this step can be skipped
model_final_layer = alexNet.create_model(input)
logits = tf.identity(model_final_layer, name='logits')

# Define the cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=output))

# Define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define the accuracy metric
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Instantiate a trainer with the chosen model and the dataset
trainer = ClfTrainer(alexnet, cifar10_dataset)

# Start training
trainer.train(input, output,
              cost, optimizer, accuracy,
              epochs, batch_size,
              save_model_path, save_every_epoch=1)
```

#### Train from where left off
```python
trainer.train_from_ckpt(input, output,
                        cost, optimizer, accuracy,
                        epochs, batch_size,
                        save_model_path, save_model_to,
                        save_every_epoch=1)
```

#### Transfer Learning
```python
# Assume the VGGNet below has already been trained on CIFAR-10 dataset
# Then, it will be trained on CIFAR-100 dataset
vgg = VGG()
vgg.load_pretrained_model(save_model_path, options={'model_type': 'A'})

cifar100_dataset = Cifar100()

# Retrieve the layer before the target layer
# Make some changes then
before_out = vgg.dropout_2
out = fully_connected(before_out, num_outputs=cifar100_dataset.num_classes, activation_fn=None)

# It is required to give a new name on each newly defined Tensor/Operation
with tf.name_scope("new_train"):
    output = tf.placeholder(tf.int32, [None, cifar100_dataset.num_classes], name='output_new')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=output), name='cost_new')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_new').minimize(cost)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_new')

trainer = ClfTrainer(vgg, cifar100_dataset)
trainer.transfer_learning(vgg.input, output,
                          cost, optimizer, accuracy,
                          epochs, batch_size,
                          save_model_path_floydhub, save_model_to)
```

## Basic Workflow
1. Define/Instantiate a dataset
2. Define/Instantiate a model
3. Define/Instantiate a trainer with the dataset and the model
4. Begin training

## References
- [CNN Receptive Field Calculator](http://fomoro.com/tools/receptive-fields/index.html)
