import numpy as np
from disropt.functions.abstract_function import AbstractFunction
from typing import Callable
import tensorflow as tf


@tf.function
def _tf_subgradient(X, Y, model, loss, vars):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss_val = loss(Y, predictions)
    gradient = tape.gradient(loss_val, vars)
    return loss_val, gradient, predictions

@tf.function
def _tf_eval(X, Y, model, loss, n):
    predictions = model(X, training=True)
    loss_val = loss(Y, predictions)
    norm_loss = tf.reduce_sum(loss_val) / n
    return loss_val, norm_loss, predictions

class TensorflowLoss(AbstractFunction):
    def __init__(self, model, dataset: tf.data.Dataset, loss: Callable):
        self._model = model
        self._loss = loss
        self._dataset = dataset

        shapes = [el.shape for el in self._model.variables]
        self.input_shape = (sum(map(np.prod, shapes)), 1) # number of parameters
        self.output_shape = (1, 1)

        # initialize dataset iterator
        self._dataset_iterator = None
        self._X = None
        self._Y = None
    
    def __check_batch_loaded(self):
        if self._X is None or self._Y is None:
            return RuntimeError("Batch not loaded. Use load_batch() before calling this function")
    
    def __check_epoch_initialized(self):
        if self._dataset_iterator is None:
            return RuntimeError("Epoch not initialized. Use init_epoch() before calling this function")

    def load_batch(self):
        self.__check_epoch_initialized()
        try:
            self._X = self._Y = None
            self._X, self._Y = next(self._dataset_iterator)
            return True
        except StopIteration:
            self._dataset_iterator = None
            return False
    
    def init_epoch(self):
        self._dataset_iterator = None
        self._dataset_iterator = iter(self._dataset)
    
    @property
    def n_batches(self):
        return int(tf.data.experimental.cardinality(self._dataset))
    
    def set_metrics(self, loss_metric=None, acc_metric=None):
        self._loss_metric = loss_metric
        self._acc_metric  = acc_metric
    
    def reset_metrics(self):
        if self._loss_metric is not None:
            self._loss_metric.reset_states()
        if self._acc_metric is not None:
            self._acc_metric.reset_states()
    
    def metrics_result(self):
        loss = acc = 0
        if self._loss_metric is not None:
            loss = self._loss_metric.result()
        if self._acc_metric is not None:
            acc = self._acc_metric.result()
        return loss, acc

    def eval(self, update_metrics=True):
        self.__check_batch_loaded()
        loss, norm_loss, predictions = _tf_eval(self._X, self._Y, self._model, self._loss, self._X.shape[0])
        if update_metrics and self._loss_metric is not None:
            self._loss_metric.update_state(loss)
        if update_metrics and self._acc_metric is not None:
            self._acc_metric.update_state(self._Y, predictions)
        return norm_loss

    def subgradient(self, update_metrics=True):
        self.__check_batch_loaded()
        loss, gradient, predictions = _tf_subgradient(self._X, self._Y, self._model, self._loss, self._model.trainable_variables)
        gradient_np = [x.numpy() for x in gradient]
        if update_metrics and self._loss_metric is not None:
            self._loss_metric.update_state(loss)
        if update_metrics and self._acc_metric is not None:
            self._acc_metric.update_state(self._Y, predictions)
        return gradient_np

    def get_trainable_variables(self):
        return self._model.trainable_variables
