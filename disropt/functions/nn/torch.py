import numpy as np
from disropt.functions.abstract_function import AbstractFunction
import torch
from torch import nn
from torch.utils.data import DataLoader


class TorchLoss(AbstractFunction):
    def __init__(self, model: nn.Module, dataloader: DataLoader, loss, device):
        self._model = model
        self._loss = loss
        self._dataloader = dataloader
        self._device = device

        shapes = [el.shape for el in self._model.parameters() if el.requires_grad]
        self.input_shape = (sum(map(np.prod, shapes)), 1) # number of parameters
        self.output_shape = (1, 1)

        # initialize dataset iterator
        self._dataset_iterator = None
        self._X = None
        self._Y = None

        # metrics
        self.loss_metric = None
        self.acc_metric = None
    
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
            self._X, self._Y = self._X.to(self._device), self._Y.to(self._device)
            return True
        except StopIteration:
            self._dataset_iterator = None
            return False
    
    def init_epoch(self):
        self._dataset_iterator = None
        self._dataset_iterator = iter(self._dataloader)
        self._model.train() # set model in training mode
    
    @property
    def n_batches(self):
        return len(self._dataloader)
    
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
        
        with torch.no_grad():
            # compute prediction error
            pred = self._model(self._X)
            loss = self._loss(pred, self._Y)
        
        # update metrics
        if update_metrics and self._loss_metric is not None:
            self._loss_metric(loss)
        if update_metrics and self._acc_metric is not None:
            self._acc_metric(self._Y, pred)
        
        return loss.item()

    def subgradient(self, update_metrics=True, return_gradients=False):
        self.__check_batch_loaded()

        # compute prediction error
        pred = self._model(self._X)
        loss = self._loss(pred, self._Y)

        # update metrics
        if update_metrics and self._loss_metric is not None:
            self._loss_metric.update_state(loss)
        if update_metrics and self._acc_metric is not None:
            self._acc_metric.update_state(self._Y, pred)

        # backpropagation
        self._model.zero_grad()
        loss.backward()

        # return gradients if requested
        if return_gradients:
            gradients = []
            for param in self._model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad)
            return gradients

    def get_model_parameters(self):
        return self._model.parameters()
