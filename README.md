# DeepModels

Implementation of state-of-the-art deep learning models since 2012

There are 3 main parts, and each part corresponds to a class.
- Datasets
- Models
- Trainers

## Pre-defined Classes
#### Datasets
- CIFAR-10
- CIFAR-100

#### Models
- AlexNet
- VGG (model type = [A | A-LRN | B | C | D | E])

#### Trainers
- ClfTrainer: Trainer for image classification like ILSVRC

## Basic Workflow
1. Define/Instantiate a dataset
2. Define/Instantiate a model
3. Define/Instantiate a trainer with the dataset and the model
4. Begin training
