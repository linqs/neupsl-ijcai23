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


    def internal_init_model(self, options={}):
        self._model = tensorflow.keras.models.load_model(options['load_path'])
        return {}


    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data)

        history = []
        for epoch in range(int(options['epochs'])):
            history.append(self._single_epoch_fit(gradients))

        return {
            'history': history
        }

    def _single_epoch_fit(self, structured_gradients, options={}):
        structured_gradients = tensorflow.constant(structured_gradients)

        with tensorflow.GradientTape(persistent=True) as tape:
            output = self._model(self._features, training=True)
            main_loss = tensorflow.reduce_mean(self._model.compiled_loss(self._labels, output))
            total_loss = tensorflow.add_n([main_loss] + self._model.losses)

        neural_gradients = tape.gradient(total_loss, output)
        output_gradients = (1.0 - options['alpha']) * neural_gradients + options['alpha'] * structured_gradients

        gradients = tape.gradient(output, self._model.trainable_weights, output_gradients=output_gradients)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

        # Compute the metrics scores.
        new_output = self._model(self._features)

        self._model.compiled_metrics.reset_state()
        self._model.compiled_metrics.update_state(self._labels, new_output)

        results = {
            'loss': float(total_loss.numpy()),
        }

        for metric in self._model.compiled_metrics.metrics:
            results[metric.name] = float(metric.result().numpy())

        return results


    def internal_predict(self, data, options = {}):
        self._prepare_data(data)

        predictions = self._model.predict(self._features, verbose=0)
        return predictions, {}


    def internal_eval(self, data, options = {}):
        self._prepare_data(data)

        predictions, _ = self.internal_predict(data, options=options)
        results = {'loss': self._model.compiled_loss(predictions, self._features),
                   'metrics': self._model.evaluate(predictions, self._labels)}

        return results


    def internal_save(self, options = {}):
        self._model.save(options['save-path'], save_format = 'tf')
        return {}


    def _prepare_data(self, data):
        self._features = tensorflow.constant(numpy.asarray(data[:,:-1]), dtype=tensorflow.float32)
        self._labels = tensorflow.constant(numpy.asarray([[1, 0] if label == 0 else [0, 1] for label in data[:,-1]]), dtype=tensorflow.float32)
