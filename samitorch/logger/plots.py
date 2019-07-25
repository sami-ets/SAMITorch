#  -*- coding: utf-8 -*-
#  Copyright 2019 SAMITorch Authors. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import numpy as np
import torch
from visdom import Visdom


class LinePlot(object):
    def __init__(self, visdom, x_label, y_label, title, legend):
        self._visdom = visdom
        self._window = self._visdom.line(X=torch.zeros((1,)).cpu(),
                                         Y=torch.zeros(1).cpu(),
                                         opts=dict(xlabel=x_label,
                                                   ylabel=y_label,
                                                   title=title,
                                                   legend=legend))

    def append(self, x, y):
        self._visdom.line(
            X=x,
            Y=y,
            win=self._window,
            update='append')


class ImagesPlot(object):
    def __init__(self, visdom: Visdom, title):
        self._visdom = visdom
        self._title = title
        self._window = self._visdom.images(tensor=torch.Tensor().new_ones((600, 600)), nrow=4, opts=dict(title=title))

    def update(self, tensor):
        tensor = self._normalize(tensor)

        self._visdom.images(tensor=tensor, nrow=4, win=self._window, opts=dict(title=self._title))

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + 1e-6)


class AccuracyPlot(LinePlot):
    def __init__(self, visdom, title):
        super().__init__(visdom, 'step', 'accuracy', title, ['accuracy'])


class LossPlot(LinePlot):
    def __init__(self, visdom, title):
        super().__init__(visdom, 'step', 'loss', title, ['loss'])


class ParameterPlot(LinePlot):
    def __init__(self, visdom, title, parameter_name):
        super().__init__(visdom, 'epoch', parameter_name, title, [parameter_name])
