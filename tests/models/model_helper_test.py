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

import torch


class TestModelHelper(object):

    def __init__(self, model, criterion, optimizer):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer

    def _train_step(self, inputs: tuple):
        """
        Run a training step on model for a given batch of data.

        Parameters of the model accumulate gradients and the optimizer performs
        a gradient update on the parameters

        Args:
            inputs (tuple): A 2-element tuple of inputs and labels, to be fed to the model.
        """
        self._model.train()
        self._optimizer.zero_grad()
        inputs, targets = inputs
        preds = self._model.forward(inputs)
        loss = self._criterion(preds, targets)
        loss.backward()
        self._optimizer.step()

    def _forward_step(self, inputs: list):
        """
        Run a forward step on the model.

        Args:
            inputs (list): Model's inputs.

        Returns:
            torch.Tensor: Model's predictions (output).

        """
        self._model.eval()
        with torch.no_grad():
            return self._model(inputs)

    def _var_change_helper(self, vars_change: bool, inputs: tuple, params: list = None):
        """
        Check if given variables (params) change or not during training
        If parameters (params) aren't provided, check all parameters.

        Args:
            vars_change (bool): A flag which controls the check for change or not change.
            inputs (tuple): A 2-element tuple of inputs and labels, to be fed to the model.
            params (list): List of parameters of form (name, variable), optional.

        Raises:
            ValueError: if the model's variables do not behave accordingly to vars_change parameter.
        """
        if params is None:
            # get a list of params that are allowed to change
            params = [np for np in self._model.named_parameters() if np[1].requires_grad]

        # take a copy
        initial_params = [(name, p.clone()) for (name, p) in params]

        # run a training step
        self._train_step(inputs)

        # check if variables have changed
        for (_, p0), (name, p1) in zip(initial_params, params):
            try:
                if vars_change:
                    assert not torch.equal(p0, p1)
                else:
                    assert torch.equal(p0, p1)
            except AssertionError:
                raise ValueError(  # error message
                    "{var_name} {msg}".format(
                        var_name=name,
                        msg='did not change' if vars_change else 'changed'
                    )
                )

    def assert_vars_change(self, inputs: tuple, params: list = None):
        """
        Make sure that the given parameters (params) DO change during training
        If parameters (params) aren't provided, check all parameters.

        Args:
              inputs (tuple): A 2-element tuple of inputs and labels, to be fed to the model.
              params (list): List of parameters of form (name, variable), optional.

        Raises:
              ValueError: if the model's variables do not change.
        """

        self._var_change_helper(True, inputs, params)

    def assert_var_same(self, inputs: tuple, params: list = None):
        """
        Make sure that the given parameters (params) DO NOT change during training
        If parameters (params) aren't provided, check all parameters.

        Args:
              inputs (tuple): A 2-element tuple of inputs and labels, to be fed to the model.
              params (list): List of parameters of form (name, variable), optional.

        Raises:
              ValueError: if the model's variables change.
        """
        self._var_change_helper(False, inputs, params)

    @staticmethod
    def assert_not_nan(tensor):
        """
        Make sure there are no NaN values in the given tensor.

        Args:
            tensor (:obj:`torch.Tensor`): Input tensor.

        Raises:
            ValueError: If one or more NaN values occur in the given tensor.
        """
        try:
            assert not torch.isnan(tensor).byte().any()
        except AssertionError:
            raise ValueError("There was a NaN value in tensor.")

    @staticmethod
    def assert_never_inf(tensor):
        """
        Make sure there are no Inf values in the given tensor.

        Args:
            tensor (:obj:`torch.Tensor`): Input tensor.

        Raises:
            ValueError: If one or more Inf values occur in the given tensor.
        """
        try:
            assert torch.isfinite(tensor).byte().any()
        except AssertionError:
            raise ValueError("There was an Inf value in tensor")
