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

import unittest
import torch

from hamcrest import *
from metrics.metrics import Accuracy, MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, MeanPairwiseDistance, \
    TopKCategoricalAccuracy, Precision, Recall
from sklearn.metrics import accuracy_score, precision_score


class AccuracyMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def test_check_shapes():
        # Instanciate an Accuracy metric.
        the_metric = Accuracy()

        predictions = torch.randint(0, 2, size=(10, 1, 28, 28)).type(torch.LongTensor)
        targets = torch.randint(0, 2, size=(10, 28, 28)).type(torch.LongTensor)

        # Sanity check on test's inputs.
        assert_that(predictions.shape, is_((10, 1, 28, 28)))
        assert_that(targets.shape, is_((10, 28, 28)))

        assert_that(the_metric._check_shapes(predictions, targets), not raises(ValueError))

        predictions = torch.randint(0, 2, size=(10, 28, 28)).type(torch.LongTensor)
        targets = torch.randint(0, 2, size=(10, 1, 28, 28)).type(torch.LongTensor)

        # Sanity check on test's inputs.
        assert_that(predictions.shape, is_((10, 28, 28)))
        assert_that(targets.shape, is_((10, 1, 28, 28)))

        assert_that(the_metric._check_shapes(predictions, targets), not raises(ValueError))

    @staticmethod
    def test_binary_wrong_inputs():
        the_metric = Accuracy()

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                                                          torch.arange(0, 10).type(torch.LongTensor)), raises(
            ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.rand(10, 1),
                                                          torch.randint(0, 2, size=(10,)).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                                                          torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor),
                                                          torch.randint(0, 2, size=(10,)).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                                                          torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor)),
                    raises(ValueError))

    @staticmethod
    def test_binary_input_N():
        # Binary accuracy on input of shape (N, 1) or (N, )
        def _test():
            # Test with shape (N, 1)
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with shape (N, )
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(100,)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(100,)).type(torch.LongTensor)
            n_iters = 16
            batch_size = targets.shape[0] // (n_iters + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().ravel()
                assert_that(the_metric._type, is_("binary"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_binary_input_NL():
        # Binary accuracy on input of shape (N, L)
        def _test():
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches.
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(100, 8)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(100, 8)).type(torch.LongTensor)
            n_iters = 16
            batch_size = targets.shape[0] // (n_iters + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().ravel()
                assert_that(the_metric._type, is_("binary"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_binary_input_NHW():
        # Binary accuracy on input of shape (N, H, W, ...)
        def _test():
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(4, 12, 10)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(4, 12, 10)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(4, 1, 12, 10)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(4, 1, 12, 10)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy()
            predictions = torch.randint(0, 2, size=(100, 1, 8, 8)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(100, 8, 8)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().ravel()
                assert_that(the_metric._type, is_("binary"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_multiclass_wrong_inputs():
        # Incompatible shapes
        the_metric = Accuracy()
        assert_that(calling(the_metric.compute).with_args(torch.rand(10, 5, 4),
                                                          torch.randint(0, 2, size=(10,)).type(torch.LongTensor)),
                    raises(ValueError))

        # incompatible shapes
        the_metric = Accuracy()
        assert_that(calling(the_metric.compute).with_args(torch.rand(10, 5, 6),
                                                          torch.randint(0, 5, size=(10, 5)).type(torch.LongTensor)),
                    raises(ValueError))

        # incompatible shapes
        the_metric = Accuracy()
        assert_that(calling(the_metric.compute).with_args(torch.rand(10),
                                                          torch.randint(0, 5, size=(10, 5, 6)).type(torch.LongTensor)),
                    raises(ValueError))

    @staticmethod
    def test_multiclass_input_N():
        # Multiclass input data of shape (N, ) and (N, C)
        def _test():
            the_metric = Accuracy()
            predictions = torch.rand(10, 4)
            targets = torch.randint(0, 4, size=(10,)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy()
            predictions = torch.rand(4, 10)
            targets = torch.randint(0, 10, size=(4, 1)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy()
            predictions = torch.rand(4, 2)
            targets = torch.randint(0, 2, size=(4, 1)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches.
            the_metric = Accuracy()
            predictions = torch.rand(100, 5)
            targets = torch.randint(0, 5, size=(100,)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().argmax(axis=1).ravel()
                assert_that(the_metric._type, is_("multiclass"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_multiclass_input_NL():
        # Multiclass input data of shape (N, L) and (N, C, L)
        def _test():
            # Test with shape (N, C, L)
            the_metric = Accuracy()
            predictions = torch.rand(10, 4, 5)
            targets = torch.randint(0, 4, size=(10, 5)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with shape (C, N, L)
            the_metric = Accuracy()
            predictions = torch.rand(4, 10, 5)
            targets = torch.randint(0, 10, size=(4, 5)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy()
            predictions = torch.rand(100, 9, 7)
            targets = torch.randint(0, 9, size=(100, 7)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().argmax(axis=1).ravel()
                assert_that(the_metric._type, is_("multiclass"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_multiclass_input_NHW():
        # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
        def _test():
            # Test with shape (N, H, W, ...)
            the_metric = Accuracy()
            predictions = torch.rand(4, 5, 12, 10)
            targets = torch.randint(0, 5, size=(4, 12, 10)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with shape (N, C, H, W, ...)
            the_metric = Accuracy()
            predictions = torch.rand(4, 5, 10, 12, 8)
            targets = torch.randint(0, 5, size=(4, 10, 12, 8)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy()
            predictions = torch.rand(100, 3, 8, 8)
            targets = torch.randint(0, 3, size=(100, 8, 8)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().argmax(axis=1).ravel()
                assert_that(the_metric._type, is_("multiclass"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_multiclass_input_NHWD():
        # Multiclass input data of shape (N, H, W, D, ...) and (N, C, H, W, D, ...)
        def _test():
            # Test with shape (N, H, W, D, ...)
            the_metric = Accuracy()
            predictions = torch.rand(4, 5, 12, 10, 14)
            targets = torch.randint(0, 5, size=(4, 12, 10, 14)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with shape (N, C, H, W, D, ...)
            the_metric = Accuracy()
            predictions = torch.rand(4, 5, 10, 12, 8, 14)
            targets = torch.randint(0, 5, size=(4, 10, 12, 8, 14)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy().argmax(axis=1).ravel()
            np_targets = targets.numpy().ravel()
            assert_that(the_metric._type, is_("multiclass"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy()
            predictions = torch.rand(100, 3, 8, 8, 8)
            targets = torch.randint(0, 3, size=(100, 8, 8, 8)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx: idx + batch_size].numpy().argmax(axis=1).ravel()
                assert_that(the_metric._type, is_("multiclass"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def to_numpy_multilabel(y):
        # reshapes input array to (N x ..., C)
        y = y.transpose(1, 0).numpy()
        num_classes = y.shape[0]
        y = y.reshape((num_classes, -1)).transpose(1, 0)
        return y

    @staticmethod
    def test_multilabel_wrong_inputs():
        the_metric = Accuracy(is_multilabel=True)
        assert_that(
            calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)),
                                                  torch.randint(0, 2, size=(10,)).type(torch.LongTensor)),
            raises(ValueError))

        the_metric = Accuracy(is_multilabel=True)
        assert_that(
            calling(the_metric.compute).with_args(torch.rand(10, 5),
                                                  torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)),
            raises(ValueError))

        the_metric = Accuracy(is_multilabel=True)
        assert_that(
            calling(the_metric.compute).with_args(torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)),
            raises(ValueError))

    @staticmethod
    def test_multilabel_input_N():
        # Multilabel input data of shape (N, C, ...) and (N, C, ...)

        def _test():
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(10, 4))
            targets = torch.randint(0, 2, size=(10, 4)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy()
            np_targets = targets.numpy()
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(50, 7)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(50, 7)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = predictions.numpy()
            np_targets = targets.numpy()
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(100, 4))
            targets = torch.randint(0, 2, size=(100, 4)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_targets = targets[idx: idx + batch_size].numpy()
                np_predictions = predictions[idx: idx + batch_size].numpy()
                assert_that(the_metric._type, is_("multilabel"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    def test_multilabel_input_NL(self):
        # Multilabel input data of shape (N, C, L, ...) and (N, C, L, ...)
        def _test():
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(10, 4, 5))
            targets = torch.randint(0, 2, size=(10, 4, 5)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = self.to_numpy_multilabel(predictions)  # (N, C, L, ...) -> (N * L * ..., C)
            np_targets = self.to_numpy_multilabel(targets)  # (N, C, L, ...) -> (N * L ..., C)
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(4, 10, 8)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(4, 10, 8)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = self.to_numpy_multilabel(predictions)  # (N, C, L, ...) -> (N * L * ..., C)
            np_targets = self.to_numpy_multilabel(targets)  # (N, C, L, ...) -> (N * L ..., C)
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # test with batches
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(100, 4, 5))
            targets = torch.randint(0, 2, size=(100, 4, 5)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_predictions = self.to_numpy_multilabel(
                    predictions[idx: idx + batch_size])  # (N, C, L, ...) -> (N * L * ..., C)
                np_targets = self.to_numpy_multilabel(
                    targets[idx: idx + batch_size])  # (N, C, L, ...) -> (N * L ..., C)
                assert_that(the_metric._type, is_("multilabel"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # Check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    def test_multilabel_input_NHW(self):
        # Multilabel input data of shape (N, C, H, W, ...) and (N, C, H, W, ...)

        def _test():
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(4, 5, 12, 10))
            targets = torch.randint(0, 2, size=(4, 5, 12, 10)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = self.to_numpy_multilabel(predictions)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            np_targets = self.to_numpy_multilabel(targets)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(4, 10, 12, 8)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(4, 10, 12, 8)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = self.to_numpy_multilabel(predictions)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            np_targets = self.to_numpy_multilabel(targets)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(100, 5, 12, 10))
            targets = torch.randint(0, 2, size=(100, 5, 12, 10)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_predictions = self.to_numpy_multilabel(
                    predictions[idx: idx + batch_size])  # (N, C, L, ...) -> (N * L * ..., C)
                np_targets = self.to_numpy_multilabel(
                    targets[idx: idx + batch_size])  # (N, C, L, ...) -> (N * L ..., C)
                assert_that(the_metric._type, is_("multilabel"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    def test_multilabel_input_NHWD(self):
        # Multilabel input data of shape (N, C, H, W, D, ...) and (N, C, H, W, D, ...)

        def _test():
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(4, 5, 12, 10, 14))
            targets = torch.randint(0, 2, size=(4, 5, 12, 10, 14)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = self.to_numpy_multilabel(predictions)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            np_targets = self.to_numpy_multilabel(targets)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(4, 10, 12, 8, 14)).type(torch.LongTensor)
            targets = torch.randint(0, 2, size=(4, 10, 12, 8, 14)).type(torch.LongTensor)
            accuracy = the_metric.compute(predictions, targets)
            np_predictions = self.to_numpy_multilabel(predictions)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            np_targets = self.to_numpy_multilabel(targets)  # (N, C, H, W, ...) -> (N * H * W ..., C)
            assert_that(the_metric._type, is_("multilabel"))
            assert_that(accuracy, is_(float))
            assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

            # Test with batches
            the_metric = Accuracy(is_multilabel=True)
            predictions = torch.randint(0, 2, size=(100, 5, 12, 10, 14))
            targets = torch.randint(0, 2, size=(100, 5, 12, 10, 14)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                accuracy = the_metric.compute(predictions[idx: idx + batch_size], targets[idx: idx + batch_size])
                np_predictions = self.to_numpy_multilabel(
                    predictions[idx: idx + batch_size])  # (N, C, L, ...) -> (N * L * ..., C)
                np_targets = self.to_numpy_multilabel(
                    targets[idx: idx + batch_size])  # (N, C, L, ...) -> (N * L ..., C)
                assert_that(the_metric._type, is_("multilabel"))
                assert_that(accuracy, is_(float))
                assert_that(accuracy_score(np_targets, np_predictions), equal_to(accuracy))

        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test()

    @staticmethod
    def test_incorrect_type():
        # Instanciate an Accuracy metric.
        the_metric = Accuracy()

        # Start as binary data
        predictions = torch.randint(0, 2, size=(4,))
        targets = torch.ones(4).type(torch.LongTensor)
        the_metric.compute(predictions, targets)

        # And add a multiclass data
        predictions = torch.rand(4, 4)
        targets = torch.ones(4).type(torch.LongTensor)

        assert_that(calling(the_metric.compute).with_args(predictions, targets), raises(RuntimeError))


class MeanSquareErrorMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def test_compute():
        the_metric = MeanSquaredError()
        predictions = torch.Tensor([[2.0], [-2.0]])
        target = torch.zeros(2)
        mse = the_metric.compute(predictions, target)
        assert_that(mse, is_(float))
        assert_that(mse, equal_to(4.0))

        the_metric = MeanSquaredError()
        predictions = torch.Tensor([[3.0], [-3.0]])
        target = torch.zeros(2)
        mse = the_metric.compute(predictions, target)
        assert_that(mse, is_(float))
        assert_that(mse, equal_to(9.0))


class RootMeanSquareErrorMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def test_compute():
        the_metric = RootMeanSquaredError()
        predictions = torch.Tensor([[2.0], [-2.0]])
        targets = torch.zeros(2)
        rmse = the_metric.compute(predictions, targets)
        assert_that(rmse, is_(float))
        assert_that(rmse, equal_to(2.0))

        the_metric = RootMeanSquaredError()
        predictions = torch.Tensor([[3.0], [-3.0]])
        targets = torch.zeros(2)
        rmse = the_metric.compute(predictions, targets)
        assert_that(rmse, is_(float))
        assert_that(rmse, equal_to(3.0))


class MeanAbsoluteErrorMetricTest(unittest.TestCase):
    def setUp(self):
        pass

    @staticmethod
    def test_compute():
        the_metric = MeanAbsoluteError()
        predictions = torch.Tensor([[2.0], [-2.0]])
        targets = torch.zeros(2)
        mae = the_metric.compute(predictions, targets)
        assert_that(mae, is_(float))
        assert_that(mae, equal_to(2.0))

        the_metric = MeanAbsoluteError()
        predictions = torch.Tensor([[3.0], [-3.0]])
        targets = torch.zeros(2)
        mae = the_metric.compute(predictions, targets)
        assert_that(mae, is_(float))
        assert_that(mae, equal_to(3.0))


class MeanPairwiseDistanceMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def test_compute():
        the_metric = MeanPairwiseDistance()
        predictions = torch.Tensor([[3.0, 4.0], [-3.0, -4.0]])
        targets = torch.zeros(2, 2)
        mpd = the_metric.compute(predictions, targets)
        assert_that(mpd, is_(float))
        assert_that(mpd, equal_to(5.0))

        the_metric = MeanPairwiseDistance()
        predictions = torch.Tensor([[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0]])
        targets = torch.zeros(2, 4)
        mpd = the_metric.compute(predictions, targets)
        assert_that(mpd, is_(float))
        assert_that(mpd, equal_to(8.0))


class TopKCategoritcalAccuracyMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def test_compute():
        the_metric = TopKCategoricalAccuracy(2)
        predictions = torch.FloatTensor([[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2]])
        targets = torch.ones(2).type(torch.LongTensor)
        top_k_accuracy = the_metric.compute(predictions, targets)
        assert_that(top_k_accuracy, is_(float))
        assert_that(top_k_accuracy, equal_to(0.50))

        the_metric = TopKCategoricalAccuracy(2)
        predictions = torch.FloatTensor([[0.4, 0.8, 0.2, 0.6], [0.8, 0.6, 0.4, 0.2]])
        target = torch.ones(2).type(torch.LongTensor)
        top_k_accuracy = the_metric.compute(predictions, target)
        assert_that(top_k_accuracy, is_(float))
        assert_that(top_k_accuracy, equal_to(1.0))


class PrecisionMetricTest(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def test_binary_wrong_inputs():
        the_metric = Precision()

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                                                          torch.arange(0, 10).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.rand(10, 1),
                                                          torch.randint(0, 2, size=(10,)).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                                                          torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor),
                                                          torch.randint(0, 2, size=(10,)).type(torch.LongTensor)),
                    raises(ValueError))

        assert_that(calling(the_metric.compute).with_args(torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                                                          torch.randint(0, 2, size=(10, 5, 6)).type(
                                                              torch.LongTensor)),
                    raises(ValueError))

    @staticmethod
    def test_binary_input_N():
        # Binary accuracy on input of shape (N, 1) or (N, )

        def _test(average):
            the_metric = Precision(average=average)
            predictions = torch.randint(0, 2, size=(10, 1))
            targets = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
            precision = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(precision, is_(float) if average else is_(torch.Tensor))
            pr_compute = the_metric.compute(predictions, targets) if average else the_metric.compute(predictions,
                                                                                                     targets).numpy()
            assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

            the_metric = Precision(average=average)
            predictions = torch.randint(0, 2, size=(10,))
            targets = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
            precision = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(precision, is_(float) if average else is_(torch.Tensor))
            pr_compute = the_metric.compute(predictions, targets) if average else the_metric.compute(predictions,
                                                                                                     targets).numpy()
            assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

            the_metric = Precision(average=average)
            predictions = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])
            predictions = torch.round(predictions)
            targets = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
            precision = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(precision, is_(float) if average else is_(torch.Tensor))
            pr_compute = the_metric.compute(predictions, targets) if average else the_metric.compute(predictions,
                                                                                                     targets).numpy()
            assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

            # Test with batches.
            the_metric = Precision(average=average)
            predictions = torch.randint(0, 2, size=(100,))
            targets = torch.randint(0, 2, size=(100,)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                precision = the_metric.compute(predictions[idx:idx + batch_size], targets[idx:idx + batch_size])
                np_targets = targets[idx:idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx:idx + batch_size].numpy().ravel()
                assert_that(the_metric._type, is_("binary"))
                assert_that(precision, is_(float) if average else is_(torch.Tensor))
                pr_compute = the_metric.compute(predictions[idx:idx + batch_size],
                                                targets[idx:idx + batch_size]) if average else the_metric.compute(
                    predictions[idx:idx + batch_size],
                    targets[idx:idx + batch_size]).numpy()
                assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

        for _ in range(5):
            _test(average=True)
            _test(average=False)

    @staticmethod
    def test_binary_input_NL():
        # Binary accuracy on input of shape (N, L)

        def _test(average):
            the_metric = Precision(average=average)
            predictions = torch.randint(0, 2, size=(10, 5))
            targets = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
            precision = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(precision, is_(float) if average else is_(torch.Tensor))
            pr_compute = the_metric.compute(predictions, targets) if average else the_metric.compute(predictions,
                                                                                                     targets).numpy()
            assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

            the_metric = Precision(average=average)
            predictions = torch.randint(0, 2, size=(10, 1, 5))
            targets = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
            precision = the_metric.compute(predictions, targets)
            np_targets = targets.numpy().ravel()
            np_predictions = predictions.numpy().ravel()
            assert_that(the_metric._type, is_("binary"))
            assert_that(precision, is_(float) if average else is_(torch.Tensor))
            pr_compute = the_metric.compute(predictions, targets) if average else the_metric.compute(predictions,
                                                                                                     targets).numpy()
            assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

            # Test with batches
            the_metric = Precision(average=average)
            predictions = torch.randint(0, 2, size=(100, 5))
            targets = torch.randint(0, 2, size=(100, 1, 5)).type(torch.LongTensor)
            batch_size = 16
            n_iters = targets.shape[0] // (batch_size + 1)

            for i in range(n_iters):
                idx = i * batch_size
                precision = the_metric.compute(predictions[idx:idx + batch_size], targets[idx:idx + batch_size])
                np_targets = targets[idx:idx + batch_size].numpy().ravel()
                np_predictions = predictions[idx:idx + batch_size].numpy().ravel()
                assert_that(the_metric._type, is_("binary"))
                assert_that(precision, is_(float) if average else is_(torch.Tensor))
                pr_compute = the_metric.compute(predictions[idx:idx + batch_size],
                                                targets[idx:idx + batch_size]) if average else the_metric.compute(
                    predictions[idx:idx + batch_size],
                    targets[idx:idx + batch_size]).numpy()
                assert_that(precision_score(np_targets, np_predictions), equal_to(pr_compute))

        for _ in range(5):
            _test(average=True)
            _test(average=False)


#
#
# def test_binary_input_NHW():
#     # Binary accuracy on input of shape (N, H, W)
#
#     def _test(average):
#         pr = Precision(average=average)
#
#         predictions = torch.randint(0, 2, size=(10, 12, 10))
#         y = torch.randint(0, 2, size=(10, 12, 10)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_targets = y.numpy().ravel()
#         np_predictions = predictions.numpy().ravel()
#         assert pr._type == 'binary'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         assert precision_score(np_targets, np_predictions, average='binary') == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(10, 1, 12, 10))
#         y = torch.randint(0, 2, size=(10, 1, 12, 10)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_targets = y.numpy().ravel()
#         np_predictions = predictions.numpy().ravel()
#         assert pr._type == 'binary'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         assert precision_score(np_targets, np_predictions, average='binary') == pytest.approx(pr_compute)
#
#         pr = Precision(average=average)
#         # Batched Updates
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(100, 12, 10))
#         y = torch.randint(0, 2, size=(100, 1, 12, 10)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         np_targets = y.numpy().ravel()
#         np_predictions = predictions.numpy().ravel()
#         assert pr._type == 'binary'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         assert precision_score(np_targets, np_predictions, average='binary') == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#
# def test_multiclass_wrong_inputs():
#     pr = Precision()
#
#     with pytest.raises(ValueError):
#         # incompatible shapes
#         pr.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))
#
#     with pytest.raises(ValueError):
#         # incompatible shapes
#         pr.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).type(torch.LongTensor)))
#
#     with pytest.raises(ValueError):
#         # incompatible shapes
#         pr.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).type(torch.LongTensor)))
#
#     pr = Precision(average=True)
#
#     with pytest.raises(ValueError):
#         # incompatible shapes between two updates
#         pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))
#         pr.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))
#
#     with pytest.raises(ValueError):
#         # incompatible shapes between two updates
#         pr.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))
#         pr.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))
#
#     pr = Precision(average=False)
#
#     with pytest.raises(ValueError):
#         # incompatible shapes between two updates
#         pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))
#         pr.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))
#
#     with pytest.raises(ValueError):
#         # incompatible shapes between two updates
#         pr.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))
#         pr.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))
#
#
# def test_multiclass_input_N():
#     # Multiclass input data of shape (N, ) and (N, C)
#
#     def _test(average):
#         pr = Precision(average=average)
#         predictions = torch.rand(20, 6)
#         y = torch.randint(0, 6, size=(20,)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.rand(10, 4)
#         y = torch.randint(0, 4, size=(10, 1)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         # 2-classes
#         pr.reset()
#         predictions = torch.rand(10, 2)
#         y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         # Batched Updates
#         pr.reset()
#         predictions = torch.rand(100, 3)
#         y = torch.randint(0, 3, size=(100,)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         num_classes = predictions.shape[1]
#         np_targets = y.numpy().ravel()
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#
# def test_multiclass_input_NL():
#     # Multiclass input data of shape (N, L) and (N, C, L)
#
#     def _test(average):
#         pr = Precision(average=average)
#
#         predictions = torch.rand(10, 5, 8)
#         y = torch.randint(0, 5, size=(10, 8)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.rand(15, 10, 8)
#         y = torch.randint(0, 10, size=(15, 8)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         # Batched Updates
#         pr.reset()
#         predictions = torch.rand(100, 8, 12)
#         y = torch.randint(0, 8, size=(100, 12)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         num_classes = predictions.shape[1]
#         np_targets = y.numpy().ravel()
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#
# def test_multiclass_input_NHW():
#     # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
#
#     def _test(average):
#         pr = Precision(average=average)
#
#         predictions = torch.rand(10, 5, 18, 16)
#         y = torch.randint(0, 5, size=(10, 18, 16)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.rand(10, 7, 20, 12)
#         y = torch.randint(0, 7, size=(10, 20, 12)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         num_classes = predictions.shape[1]
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         np_targets = y.numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#         # Batched Updates
#         pr.reset()
#         predictions = torch.rand(100, 8, 12, 14)
#         y = torch.randint(0, 8, size=(100, 12, 14)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         num_classes = predictions.shape[1]
#         np_targets = y.numpy().ravel()
#         np_predictions = predictions.argmax(dim=1).numpy().ravel()
#         assert pr._type == 'multiclass'
#         assert isinstance(pr.compute(), float if average else torch.Tensor)
#         pr_compute = pr.compute() if average else pr.compute().numpy()
#         sk_average_parameter = 'macro' if average else None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             sk_compute = precision_score(np_targets, np_predictions, labels=range(0, num_classes),
#                                          average=sk_average_parameter)
#             assert sk_compute == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#
# def test_multilabel_wrong_inputs():
#     pr = Precision(average=True, is_multilabel=True)
#
#     with pytest.raises(ValueError):
#         # incompatible shapes
#         pr.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))
#
#     with pytest.raises(ValueError):
#         # incompatible predictions
#         pr.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))
#
#     with pytest.raises(ValueError):
#         # incompatible y
#         pr.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))
#
#     with pytest.raises(ValueError):
#         # incompatible shapes between two updates
#         pr.update((torch.randint(0, 2, size=(20, 5)), torch.randint(0, 2, size=(20, 5)).type(torch.LongTensor)))
#         pr.update((torch.randint(0, 2, size=(20, 6)), torch.randint(0, 2, size=(20, 6)).type(torch.LongTensor)))
#
#
# def to_numpy_multilabel(y):
#     # reshapes input array to (N x ..., C)
#     y = y.transpose(1, 0).numpy()
#     num_classes = y.shape[0]
#     y = y.reshape((num_classes, -1)).transpose(1, 0)
#     return y
#
#
# def test_multilabel_input_NC():
#     def _test(average):
#         pr = Precision(average=average, is_multilabel=True)
#
#         predictions = torch.randint(0, 2, size=(20, 5))
#         y = torch.randint(0, 2, size=(20, 5)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_predictions = predictions.numpy()
#         np_targets = y.numpy()
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(10, 4))
#         y = torch.randint(0, 2, size=(10, 4)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_predictions = predictions.numpy()
#         np_targets = y.numpy()
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#         # Batched Updates
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(100, 4))
#         y = torch.randint(0, 2, size=(100, 4)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         np_targets = y.numpy()
#         np_predictions = predictions.numpy()
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#     pr1 = Precision(is_multilabel=True, average=True)
#     pr2 = Precision(is_multilabel=True, average=False)
#     predictions = torch.randint(0, 2, size=(10, 4, 20, 23))
#     y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
#     pr1.update((predictions, y))
#     pr2.update((predictions, y))
#     assert pr1.compute() == pytest.approx(pr2.compute().mean().item())
#
#
# def test_multilabel_input_NCL():
#     def _test(average):
#         pr = Precision(average=average, is_multilabel=True)
#
#         predictions = torch.randint(0, 2, size=(10, 5, 10))
#         y = torch.randint(0, 2, size=(10, 5, 10)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_predictions = to_numpy_multilabel(predictions)
#         np_targets = to_numpy_multilabel(y)
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(15, 4, 10))
#         y = torch.randint(0, 2, size=(15, 4, 10)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_predictions = to_numpy_multilabel(predictions)
#         np_targets = to_numpy_multilabel(y)
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#         # Batched Updates
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(100, 4, 12))
#         y = torch.randint(0, 2, size=(100, 4, 12)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         np_targets = to_numpy_multilabel(y)
#         np_predictions = to_numpy_multilabel(predictions)
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#     pr1 = Precision(is_multilabel=True, average=True)
#     pr2 = Precision(is_multilabel=True, average=False)
#     predictions = torch.randint(0, 2, size=(10, 4, 20, 23))
#     y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
#     pr1.update((predictions, y))
#     pr2.update((predictions, y))
#     assert pr1.compute() == pytest.approx(pr2.compute().mean().item())
#
#
# def test_multilabel_input_NCHW():
#     def _test(average):
#         pr = Precision(average=average, is_multilabel=True)
#
#         predictions = torch.randint(0, 2, size=(10, 5, 18, 16))
#         y = torch.randint(0, 2, size=(10, 5, 18, 16)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_predictions = to_numpy_multilabel(predictions)
#         np_targets = to_numpy_multilabel(y)
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(10, 4, 20, 23))
#         y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
#         pr.update((predictions, y))
#         np_predictions = to_numpy_multilabel(predictions)
#         np_targets = to_numpy_multilabel(y)
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#         # Batched Updates
#         pr.reset()
#         predictions = torch.randint(0, 2, size=(100, 5, 12, 14))
#         y = torch.randint(0, 2, size=(100, 5, 12, 14)).type(torch.LongTensor)
#
#         batch_size = 16
#         n_iters = y.shape[0] // batch_size + 1
#
#         for i in range(n_iters):
#             idx = i * batch_size
#             pr.update((predictions[idx:idx + batch_size], y[idx:idx + batch_size]))
#
#         np_targets = to_numpy_multilabel(y)
#         np_predictions = to_numpy_multilabel(predictions)
#         assert pr._type == 'multilabel'
#         pr_compute = pr.compute() if average else pr.compute().mean().item()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UndefinedMetricWarning)
#             assert precision_score(np_targets, np_predictions, average='samples') == pytest.approx(pr_compute)
#
#     for _ in range(5):
#         _test(average=True)
#         _test(average=False)
#
#     pr1 = Precision(is_multilabel=True, average=True)
#     pr2 = Precision(is_multilabel=True, average=False)
#     predictions = torch.randint(0, 2, size=(10, 4, 20, 23))
#     y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
#     pr1.update((predictions, y))
#     pr2.update((predictions, y))
#     assert pr1.compute() == pytest.approx(pr2.compute().mean().item())
#
#
# def test_incorrect_type():
#     # Tests changing of type during training
#
#     def _test(average):
#         pr = Precision(average=average)
#
#         predictions = torch.softmax(torch.rand(4, 4), dim=1)
#         y = torch.ones(4).type(torch.LongTensor)
#         pr.update((predictions, y))
#
#         predictions = torch.randint(0, 2, size=(4,))
#         y = torch.ones(4).type(torch.LongTensor)
#
#         with pytest.raises(RuntimeError):
#             pr.update((predictions, y))
#
#     _test(average=True)
#     _test(average=False)
#
#     pr1 = Precision(is_multilabel=True, average=True)
#     pr2 = Precision(is_multilabel=True, average=False)
#     predictions = torch.randint(0, 2, size=(10, 4, 20, 23))
#     y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
#     pr1.update((predictions, y))
#     pr2.update((predictions, y))
#     assert pr1.compute() == pytest.approx(pr2.compute().mean().item())
#
#
# def test_incorrect_y_classes():
#     def _test(average):
#         pr = Precision(average=average)
#
#         predictions = torch.randint(0, 2, size=(10, 4)).float()
#         y = torch.randint(4, 5, size=(10,)).long()
#
#         with pytest.raises(ValueError):
#             pr.update((predictions, y))
#
#     _test(average=True)
#     _test(average=False)


if __name__ == 'main':
    unittest.main()
