#!/usr/bin/env python3
import importlib
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy
import tensorflow

import pslpython.deeppsl.model

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

class MNISTAdditionModel(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None
        self._model = None
        self._features = None
        self._labels = None
        self._digit_labels = None
        self._predictions = None
        self._tape = None


    def internal_init_model(self, application, options={}):
        self._application = application
        if self._application  == 'learning':
            self._model = self._create_model(options=options)
        elif self._application  == 'inference':
            self._model = tensorflow.keras.models.load_model(options['save-path'])

        return {}


    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = tensorflow.constant(gradients, dtype=tensorflow.float32)

        gradients = self._tape.gradient(self._predictions, self._model.trainable_weights, output_gradients=structured_gradients)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

        # Compute the metrics scores.
        new_output = self._model(self._features)

        self._model.compiled_metrics.reset_state()
        self._model.compiled_metrics.update_state(self._digit_labels, new_output)

        results = {}

        for metric in self._model.compiled_metrics.metrics:
            results[metric.name] = float(metric.result().numpy())

        return results


    def internal_predict(self, data, options = {}):
        self._prepare_data(data, options=options)

        results = {}

        if self._application == 'inference':
            self._predictions = self._model.predict(self._features, verbose=0)
            results = {'metrics': util.calculate_metrics(self._predictions, self._digit_labels.numpy(), ['categorical_accuracy'])}
        else:
            with tensorflow.GradientTape(persistent=True) as tape:
                self._predictions = self._model(self._features, training=True)
                self._tape = tape

        return self._predictions, results


    def internal_eval(self, data, options = {}):
        self._prepare_data(data, options=options)

        predictions, _ = self.internal_predict(data, options=options)
        results = {'metrics': util.calculate_metrics(predictions, self._digit_labels.numpy(), ['categorical_accuracy'])}

        return results


    def internal_save(self, options = {}):
        self._model.save(options['save-path'], save_format = 'tf')
        return {}


    # See https://github.com/ML-KULeuven/deepproblog/blob/master/src/deepproblog/examples/MNIST/network.py#L44
    # See https://arxiv.org/pdf/1907.08194.pdf#page=30
    def _create_model(self, options={}):
        layers = [
            tensorflow.keras.layers.Input(shape=784),
            tensorflow.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
            tensorflow.keras.layers.Conv2D(filters=6, kernel_size=5, data_format='channels_last'),
            tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_last'),
            tensorflow.keras.layers.Activation('relu'),
            tensorflow.keras.layers.Conv2D(filters=16, kernel_size=5, data_format='channels_last'),
            tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_last'),
            tensorflow.keras.layers.Activation('relu'),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(120, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.L1(0.1)),
            tensorflow.keras.layers.Dense(84, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.L1(0.1)),
            tensorflow.keras.layers.Dense(int(options['class-size']), activation='softmax')
        ]

        model = tensorflow.keras.Sequential(layers)

        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=float(options['learning-rate'])),
            loss='KLDivergence',
            metrics=['categorical_accuracy']
        )

        return model


    def _prepare_data(self, data, options = {}):
        self._features = tensorflow.constant(numpy.asarray(data[:,:-1]), dtype=tensorflow.float32)
        self._digit_labels = tensorflow.constant(numpy.asarray([util.one_hot_encoding(int(label), int(options['class-size'])) for label in data[:,-1]]), dtype=tensorflow.float32)
        self._labels = tensorflow.constant([0] * len(data), dtype=tensorflow.float32)