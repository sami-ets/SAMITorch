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

import unittest
import os
import re

import torch
from hamcrest import *

from samitorch.parsers.parsers import ModelConfigurationParserFactory
from samitorch.optimizers.optimizers import OptimizerFactory, Optimizer
from samitorch.models.resnet3d import ResNet3DModelFactory, ResNetModel
from samitorch.criterions.criterions import Criterion, CriterionFactory
from samitorch.configs.configurations import ModelTrainerConfiguration
from samitorch.training.model_trainer import ModelTrainer
from samitorch.training.trainer import Trainer
from samitorch.training.training_strategies import LossCheckpointStrategy, MetricCheckpointStrategy


class TestModelTrainer(ModelTrainer):
    def _setup(self):
        pass


class LossCheckpointStrategyTest(unittest.TestCase, Trainer):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE_T1 = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    CONFIGURATION_PATH = "samitorch/configs/resnet3d.yaml"
    OUTPUT_DATA_FOLDER_PATH = "tests/data/generated/LossCheckpointStrategy"
    filename_re = re.compile('ResNet3D-.*.tar')

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            for file in os.listdir(cls.OUTPUT_DATA_FOLDER_PATH):
                os.remove(os.path.join(cls.OUTPUT_DATA_FOLDER_PATH, file))
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def setUp(self):
        self.configurationParserFactory = ModelConfigurationParserFactory()
        self.model_factory = ResNet3DModelFactory()
        self.criterion_factory = CriterionFactory()
        self.optimizer_factory = OptimizerFactory()
        self._config = self.configurationParserFactory.parse(self.CONFIGURATION_PATH)
        model = self.model_factory.create_model(ResNetModel.ResNet101, self._config)
        criterion = self.criterion_factory.create(Criterion.MSELoss)
        optimizer = self.optimizer_factory.create(Optimizer.SGD,
                                                  model.parameters(),
                                                  lr=0.1)
        model_trainer_config = ModelTrainerConfiguration(model, optimizer, criterion, None, torch.device('cpu'), None,
                                                         None)

        self.model_trainer = TestModelTrainer(model_trainer_config, None, "TestTrainer", with_logging=False)
        self.strategy = LossCheckpointStrategy(self.model_trainer, "ResNet3D", self.OUTPUT_DATA_FOLDER_PATH)

    def test_model_should_save_according_to_strategy(self):
        file_exists = False
        self.strategy(1.50)
        self.strategy(1.49)

        files = os.listdir(self.OUTPUT_DATA_FOLDER_PATH)
        for file in files:
            if self.filename_re.search(file):
                file_exists = True

        assert_that(file_exists, is_(True))


class MetricCheckpointStrategyTest(unittest.TestCase, Trainer):
    TEST_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../data/test_dataset")
    PATH_TO_SOURCE_T1 = os.path.join(TEST_DATA_FOLDER_PATH, "T1")
    PATH_TO_TARGET = os.path.join(TEST_DATA_FOLDER_PATH, "label")
    CONFIGURATION_PATH = "samitorch/configs/resnet3d.yaml"
    OUTPUT_DATA_FOLDER_PATH = "tests/data/generated/MetricCheckpointStrategy"
    filename_re = re.compile('ResNet3D-.*.tar')

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            for file in os.listdir(cls.OUTPUT_DATA_FOLDER_PATH):
                os.remove(os.path.join(cls.OUTPUT_DATA_FOLDER_PATH, file))
        if not os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH):
            os.makedirs(cls.OUTPUT_DATA_FOLDER_PATH)

        assert_that(len(os.listdir(cls.OUTPUT_DATA_FOLDER_PATH)), is_(0))
        assert_that(os.path.exists(cls.OUTPUT_DATA_FOLDER_PATH), is_(True))

    def setUp(self):
        self.configurationParserFactory = ModelConfigurationParserFactory()
        self.model_factory = ResNet3DModelFactory()
        self.criterion_factory = CriterionFactory()
        self.optimizer_factory = OptimizerFactory()
        self._config = self.configurationParserFactory.parse(self.CONFIGURATION_PATH)
        model = self.model_factory.create_model(ResNetModel.ResNet101, self._config)
        criterion = self.criterion_factory.create(Criterion.MSELoss)
        optimizer = self.optimizer_factory.create(Optimizer.SGD,
                                                  model.parameters(),
                                                  lr=0.1)
        model_trainer_config = ModelTrainerConfiguration(model, optimizer, criterion, None, torch.device('cpu'), None,
                                                         None)
        self.model_trainer = TestModelTrainer(model_trainer_config, None, "TestTrainer", with_logging=False)
        self.strategy = MetricCheckpointStrategy(self.model_trainer, "ResNet3D", self.OUTPUT_DATA_FOLDER_PATH)

    def test_model_should_save_according_to_strategy(self):
        file_exists = False
        self.strategy(0.30)
        self.strategy(0.31)

        files = os.listdir(self.OUTPUT_DATA_FOLDER_PATH)
        for file in files:
            if self.filename_re.search(file):
                file_exists = True

        assert_that(file_exists, is_(True))
