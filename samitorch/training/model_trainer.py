# -*- coding: utf-8 -*-
# Copyright 2019 SAMITorch Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
import torch

from samitorch.utils.model_io import save
from samitorch.metrics.gauges import RunningAverageGauge
from samitorch.inputs.batch import Batch
from samitorch.logger.plots import AccuracyPlot, LossPlot, ParameterPlot


class ModelTrainer(object):

    def __init__(self, config, callbacks, class_name, with_logging=True):
        self._config = config
        self._callbacks = callbacks

        self._visdom = config.visdom

        self.training_metric_gauge = RunningAverageGauge()
        self.training_loss_gauge = RunningAverageGauge()
        self.validation_metric_gauge = RunningAverageGauge()
        self.validation_loss_gauge = RunningAverageGauge()

        if with_logging:
            self._training_metric_plot = AccuracyPlot(self._visdom,
                                                      "{} Training metric lr={} momentum={}".format(
                                                          class_name,
                                                          config.optimizer.param_groups[0]["lr"],
                                                          config.optimizer.param_groups[0]["momentum"]))
            self._validation_metric_plot = AccuracyPlot(self._visdom,
                                                        "{} Validation metric lr={} momentum={}".format(
                                                            class_name,
                                                            config.optimizer.param_groups[0]["lr"],
                                                            config.optimizer.param_groups[0]["momentum"]))
            self._training_loss_plot = LossPlot(self._visdom, "{} Training loss".format(class_name))
            self._validation_loss_plot = LossPlot(self._visdom, "{} Validation loss".format(class_name))
            self._learning_rate_plot = ParameterPlot(self._visdom, "{} Learning rate".format(class_name),
                                                     "learning rate")

        self._setup()

        self._global_step = torch.Tensor().new_zeros((1,), dtype=torch.int64, device='cpu')
        self._epoch = torch.Tensor().new_zeros((1,), dtype=torch.int64, device='cpu')

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def config(self):
        return self._config

    def predict(self, batch: Batch, detach: bool = False):
        transformed_batch = Batch.from_batch(batch).to_device(self._config.running_config.device)
        if detach:
            transformed_batch.x = self._config.model(batch.x).detach()
        else:
            transformed_batch.x = self._config.model(batch.x)
        return transformed_batch

    def compute_metric(self):
        metric = self._config.metric.compute()
        self._config.metric.reset()
        return metric

    def evaluate_loss(self, X, y):
        return self._config.criterion(X, y)

    def update_lr(self, new_lr):
        self._config.optimizer.param_groups["lr"] = new_lr

    def update_metric_gauge(self, metric, n_data, phase="training"):
        if phase == "training":
            self.training_metric_gauge.update(metric, n_data)
        else:
            self.validation_metric_gauge.update(metric, n_data)

    def update_loss_gauge(self, loss, n_data, phase="training"):
        if phase == "training":
            self.training_loss_gauge.update(loss, n_data)
        else:
            self.validation_loss_gauge.update(loss, n_data)

    def update_metric_plot(self, step, value, phase="training"):
        if phase == "training":
            self._training_metric_plot.append(step, value)
        else:
            self._validation_metric_plot.append(step, value)

    def update_loss_plot(self, step, value, phase="training"):
        if phase == "training":
            self._training_loss_plot.append(step, value)
        else:
            self._validation_loss_plot.append(step, value)

    def update_learning_rate_plot(self, epoch, value):
        self._learning_rate_plot.append(epoch, value)

    def update_metric(self, pred, y):
        self._config.metric.update((pred, y))

    def step(self):
        self._config.optimizer.step()

    def at_training_begin(self):
        self._initialize_model(self._config.model)

    def at_training_end(self, epoch_num: int):
        save("model.pickle", self._config.model, epoch_num, self._config.optimizer)

    def at_epoch_begin(self):
        self._config.model.train()

    def at_epoch_end(self):
        self._epoch += 1
        self.training_metric_gauge.reset()
        self.training_loss_gauge.reset()

    def at_iteration_begin(self):
        self._config.optimizer.zero_grad()

    def at_iteration_end(self):
        self._global_step += 1
        self.enable_gradients()

    def at_validation_begin(self):
        self._config.model.eval()

    def at_validation_end(self):
        self.validation_metric_gauge.reset()
        self.validation_loss_gauge.reset()

    @abc.abstractmethod
    def finalize(self):
        raise NotImplementedError()

    def disable_gradients(self):
        for p in self._config.model.parameters(): p.requires_grad = False

    def enable_gradients(self):
        for p in self._config.model.parameters(): p.requires_grad = True

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= int(self._config.running_config.world_size)
        return rt

    def reset_optimizer(self):
        self._config.optimizer.zero_grad()

    @staticmethod
    def _initialize_model(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @abc.abstractmethod
    def _setup(self):
        raise NotImplementedError()
