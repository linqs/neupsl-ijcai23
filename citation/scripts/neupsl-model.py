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
        self._model = None
        self._features = None
        self._labels = None


    def internal_init_model(self, application, options={}):
        if application == 'learning':
            self._model = tensorflow.keras.models.load_model(options['load-path'])
        elif application == 'inference':
            self._model = tensorflow.keras.models.load_model(options['save-path'])

        # Set the learning rate to a different value then the pretrained model.
        tensorflow.keras.backend.set_value(self._model.optimizer.learning_rate, float(options['learning-rate']))

        return {}


    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = tensorflow.constant(gradients, dtype=tensorflow.float32)

        with tensorflow.GradientTape(persistent=True) as tape:
            output = self._model(self._features, training=True)
            main_loss = tensorflow.reduce_mean(self._model.compiled_loss(self._labels, output))
            total_loss = tensorflow.add_n([main_loss] + self._model.losses)

        neural_gradients = tape.gradient(total_loss, output)
        output_gradients = (1.0 - float(options['alpha'])) * neural_gradients + float(options['alpha']) * structured_gradients

        gradients = tape.gradient(output, self._model.trainable_weights, output_gradients=output_gradients)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

        # Compute the metrics scores.
        new_output = self._model(self._features)

        self._model.compiled_metrics.reset_state()
        self._model.compiled_metrics.update_state(self._labels, new_output)

        results = {'loss': float(total_loss.numpy())}

        for metric in self._model.compiled_metrics.metrics:
            results[metric.name] = float(metric.result().numpy())

        return results


    def internal_predict(self, data, options = {}):
        self._prepare_data(data, options=options)

        predictions = self._model.predict(self._features, verbose=0)
        return predictions, {}


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
