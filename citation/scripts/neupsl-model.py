#!/usr/bin/env python3
import importlib
import os
import sys

import numpy
import tensorflow

import pslpython.deeppsl.model

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

class CitationModel(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None
        self._model = None
        self._features = None
        self._labels = None
        self._loss = None

    def internal_init_model(self, application, options={}):
        self._application = application

        if application == 'learning':
            self._model = tensorflow.keras.models.load_model(options['load-path'])
        elif application == 'inference':
            self._model = tensorflow.keras.models.load_model(options['save-path'])

        if 'simple' in options['load-path']:
            tensorflow.keras.backend.set_value(self._model.optimizer.learning_rate, float(options['simple-learning-rate']))
        elif 'smoothed' in options['load-path']:
            tensorflow.keras.backend.set_value(self._model.optimizer.learning_rate, float(options['smoothed-learning-rate']))

        return {}

    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = tensorflow.constant(gradients, dtype=tensorflow.float32)

        gradients = self._tape.gradient(self._predictions, self._model.trainable_weights, output_gradients=structured_gradients)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

        # Compute the metrics scores.
        new_output = self._model(self._features)

        self._model.compiled_metrics.reset_state()
        self._model.compiled_metrics.update_state(self._labels, new_output)

        results = {'loss': float(self._loss.numpy())}

        for metric in self._model.compiled_metrics.metrics:
            results[metric.name] = float(metric.result().numpy())

        return results


    def internal_predict(self, data, options = {}):
        self._prepare_data(data, options=options)

        results = {}

        if self._application == 'inference':
            self._predictions = self._model.predict(self._features, verbose=0)
            results = {'loss': float(self._model.compiled_loss(tensorflow.constant(self._predictions, dtype=tensorflow.float32), self._labels).numpy()),
                       'metrics': util.calculate_metrics(self._predictions, self._labels.numpy(), ['categorical_accuracy'])}
        else:
            with tensorflow.GradientTape(persistent=True) as tape:
                self._predictions = self._model(self._features, training=True)
                self._loss = self._model.compiled_loss(tensorflow.constant(self._predictions, dtype=tensorflow.float32), self._labels)
                self._tape = tape

        return self._predictions, results


    def internal_eval(self, data, options = {}):
        self._prepare_data(data, options=options)

        predictions, _ = self.internal_predict(data, options=options)
        results = {'loss': float(self._model.compiled_loss(tensorflow.constant(predictions, dtype=tensorflow.float32), self._labels).numpy()),
                   'metrics': util.calculate_metrics(predictions, self._labels.numpy(), ['categorical_accuracy'])}

        return results


    def internal_save(self, options = {}):
        self._model.save(options['save-path'], save_format = 'tf')
        return {}


    def _prepare_data(self, data, options = {}):
        if self._features is not None and self._labels is not None:
            return

        self._features = tensorflow.constant(numpy.asarray(data[:,:-1]), dtype=tensorflow.float32)
        self._labels = tensorflow.constant(numpy.asarray([util.one_hot_encoding(int(label), int(options['class-size'])) for label in data[:,-1]]), dtype=tensorflow.float32)