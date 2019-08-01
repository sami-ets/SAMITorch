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

import torch.nn
import torch.optim
import logging

from typing import Union, List

from collections import OrderedDict

logger = logging.getLogger(__name__)


def save(file_name: str, model: torch.nn.Module = None, epoch_num: int = None,
         optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]] = None,
         **kwargs) -> None:
    """Save a checkpoint.

    Args:
        file_name (str): The file name to save a PyTorch model checkpoint.
        model (:obj:`torch.nn.Module`): A PyTorch model.
        epoch_num (int): Current epoch number.
        optimizer (List of :obj:`torch.optim.Optimizer`):
    """

    if isinstance(model, torch.nn.DataParallel):
        _model = model.module
    else:
        _model = model

    if isinstance(_model, torch.nn.Module):
        model_state = _model.state_dict()
    else:
        model_state = {}
        logger.debug("Saving checkpoint without Model")

    optimizer_state = {}
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_state = optimizer.state_dict()
    elif isinstance(optimizer, List):
        for i, optim in enumerate(optimizer):
            optimizer_state = {}
            if isinstance(optim, torch.optim.Optimizer):
                optimizer_state["optimizer_{}".format(i)] = optim.state_dict()
            else:
                raise TypeError("Optimizer must be instance of torch.optim.Optimizer")
    else:
        logger.debug("Saving checkpoint without Optimizer")

    if epoch_num is None:
        epoch_num = 0

    state = {"optimizer": optimizer_state,
             "model": model_state,
             "epoch": epoch_num}

    torch.save(state, file_name, **kwargs)


def load(file_name: str, **kwargs) -> OrderedDict:
    """Load a checkpoint.

    Args:
        file_name (str): the file name from which to load a PyTorch model checkpoint.

    Returns:
        OrderedDict: checkpoint state_dict
    """
    checkpoint = torch.load(file_name, **kwargs)

    if not all([_key in checkpoint
                for _key in ["model", "optimizer", "epoch"]]):
        return checkpoint['state_dict']
    return checkpoint
