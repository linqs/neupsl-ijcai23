import tensorflow

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model, Sequential


class MNISTAdditionBaseline(Model):
    def __init__(self, class_size=10):
        super(MNISTAdditionBaseline, self).__init__()
        self.class_size = class_size
        self.encoder = Sequential([
            Conv2D(6, 5, input_shape=(28, 28, 1), activation='elu'),
            MaxPool2D(2, 2),  # 6 24 24 -> 6 12 12
            Conv2D(16, 5, activation='elu'),  # 6 12 12 -> 16 8 8
            MaxPool2D(2, 2)  # 16 8 8 -> 16 4 4
        ])
        self.flatten = Flatten()
        self.classifier_1 = Sequential([
            Dense(100,
                  input_shape=(16 * 4 * 4,),
                  activation='elu')
        ])
        self.classifier_2 = Sequential([
            Dense(84,
                  input_shape=(2 * 100,),
                  activation='elu'),
            Dense(19,
                  input_shape=(84,),
                  activation='relu')
        ])

    def call(self, inputs, training=None, mask=None):
        # Digit Classification
        x = self.encoder(tf.gather(inputs, tf.repeat([0], repeats=[tf.shape(inputs)[0]]), batch_dims=1, axis=1))
        x = self.flatten(x)
        x = self.classifier_1(x)

        y = self.encoder(tf.gather(inputs, tf.repeat([1], repeats=[tf.shape(inputs)[0]]), batch_dims=1, axis=1))
        y = self.flatten(y)
        y = self.classifier_1(y)

        # Summation
        x = tf.concat([x, y], axis=-1)
        x = self.classifier_2(x)

        x = softmax(x)

        return x


class MNISTAddition2Baseline(Model):
    def __init__(self):
        super(MNISTAddition2Baseline, self).__init__()
        self.encoder = Sequential([
            Conv2D(6, 5, input_shape=(28, 28, 1), activation='elu'),
            MaxPool2D(2, 2),  # 6 24 24 -> 6 12 12
            Conv2D(16, 5, activation='elu'),  # 6 12 12 -> 16 8 8
            MaxPool2D(2, 2)  # 16 8 8 -> 16 4 4
        ])
        self.flatten = Flatten()
        self.classifier_1 = Sequential([
            Dense(100, input_shape=(16 * 4 * 4,), activation='elu'),
        ])
        self.classifier_2 = Sequential([
            Dense(128,
                  input_shape=(4 * 100,),
                  activation='elu'),
            Dense(199,
                  input_shape=(128,),
                  activation='elu')
        ])

    def call(self, inputs, training=None, mask=None):
        # Number Classification: Specific for MNIST - 2
        x_1 = self.encoder(tf.gather(inputs, tf.repeat([0], repeats=[tf.shape(inputs)[0]]), batch_dims=1, axis=1))
        x_2 = self.encoder(tf.gather(inputs, tf.repeat([1], repeats=[tf.shape(inputs)[0]]), batch_dims=1, axis=1))
        x_1 = self.flatten(x_1)
        x_2 = self.flatten(x_2)
        x_1 = self.classifier_1(x_1)
        x_2 = self.classifier_1(x_2)

        y_1 = self.encoder(tf.gather(inputs, tf.repeat([2], repeats=[tf.shape(inputs)[0]]), batch_dims=1, axis=1))
        y_2 = self.encoder(tf.gather(inputs, tf.repeat([3], repeats=[tf.shape(inputs)[0]]), batch_dims=1, axis=1))
        y_1 = self.flatten(y_1)
        y_2 = self.flatten(y_2)
        y_1 = self.classifier_1(y_1)
        y_2 = self.classifier_1(y_2)

        # Summation
        x = tf.concat([x_1, x_2, y_1, y_2], axis=-1)
        x = self.classifier_2(x)

        x = softmax(x)

        return x
