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
        self._predictions = None
        self._tape = None


    def internal_init_model(self, application, options={}):
        self._application = application
        if application == 'learning':
            self._model = tensorflow.keras.models.load_model(options['load-path'])
        elif application == 'inference':
            self._model = tensorflow.keras.models.load_model(options['save-path'])

        # Set the learning rate to a different value then the pretrained model.
        tensorflow.keras.backend.set_value(self._model.optimizer.learning_rate, float(options['learning-rate']))

        return {}

    def _create_model(self, options={}):
        layers = [
            tensorflow.keras.layers.Input(shape=int(options['input-shape'])),
            tensorflow.keras.layers.Dropout(float(options['dropout'])),
            tensorflow.keras.layers.Dense(int(options['hidden-size']),
                                          kernel_regularizer=tensorflow.keras.regularizers.l2(float(options['weight-regularizer'])),
                                          bias_regularizer=tensorflow.keras.regularizers.l2(float(options['weight-regularizer'])),
                                          activation='relu'),
            tensorflow.keras.layers.Dropout(float(options['dropout'])),
            tensorflow.keras.layers.Dense(int(options['class-size']),
                                          kernel_regularizer=tensorflow.keras.regularizers.l2(float(options['weight-regularizer'])),
                                          bias_regularizer=tensorflow.keras.regularizers.l2(float(options['weight-regularizer'])),
                                          activation='softmax'),
        ]

        model = tensorflow.keras.Sequential(layers=layers)

        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=float(options['learning-rate'])),
            loss='KLDivergence',
            metrics=['categorical_accuracy']
        )

        return model

    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = tensorflow.constant(gradients, dtype=tensorflow.float32)

        gradients = self._tape.gradient(self._predictions, self._model.trainable_weights, output_gradients=structured_gradients)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

        results = {}

        return results

    def internal_predict(self, data, options = {}):
        self._prepare_data(data, options=options)

        results = {}

        if options["learn"]:
            results['mode'] = 'learning'
            with tensorflow.GradientTape(persistent=True) as tape:
                self._predictions = self._model(self._features, training=True)
                self._tape = tape
        else:
            self._predictions = self._model.predict(self._features, verbose=0)
            results = {'mode': 'inference'}

        return self._predictions, results


    def internal_eval(self, data, options = {}):
        self._prepare_data(data, options=options)

        predictions, _ = self.internal_predict(data, options=options)
        results = {}

        return results


    def internal_save(self, options = {}):
        self._model.save(options['save-path'], save_format = 'tf')
        return {}


    def _prepare_data(self, data, options = {}):
        if self._features is not None:
            return

        self._features = tensorflow.constant(numpy.asarray(data[:,:-1]), dtype=tensorflow.float32)
