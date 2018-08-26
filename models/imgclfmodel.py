import tensorflow as tf

from dataset.dataset import Dataset

class ImgClfModel:
    def __init__(self, scale_to_imagenet=False):
        self.scale_to_imagenet = scale_to_imagenet

    def set_dataset(self, dataset=None):
        if dataset is not None:
            if isinstance(dataset, Dataset):
                print('Dataset is given as ' + dataset.name)

                width = dataset.width
                height = dataset.height

                if self.scale_to_imagenet:
                    width = 224
                    height = 224

                self.num_classes = dataset.num_classes
                input = tf.placeholder(tf.float32, [None, width, height, 3], name='input')
                output = tf.placeholder(tf.int32, [None, dataset.num_classes], name='output')

                return (input, output)
            else:
                print('dataset is unknown type, please try with Dataset class type')
        else:
            raise TypeError

    def create_model(self, input, options=None):
        raise NotImplementedError

    def load_pretrained_model(self, save_model_from, options=None):
        raise NotImplementedError
