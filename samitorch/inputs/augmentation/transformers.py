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
import random
import numpy as np


class NoiseAdder(object):

    def __init__(self, exec_probability):
        self._exec_probability = exec_probability

    def __call__(self, ndarray, snr, S0, noise_type='rician'):
        """
        Add noise to a signal.

        Args:
            ndarray  (:obj:`Numpy.ndarray`): 1-d ndarray of the signal in the voxel.
            snr (float) : The desired signal-to-noise ratio. (See notes below.) If `snr` is None, return the signal
                as-is.
            S0 (float): Reference signal for specifying `snr`.
            noise_type (string, optional): The distribution of noise added. Can be either 'gaussian' for Gaussian
                distributed noise, 'rician' for Rice-distributed noise (default) or 'rayleigh' for a Rayleigh
                 distribution.

            Notes:
                SNR is defined here, following [1]_, as ``S0 / sigma``, where ``sigma`` is
                the standard deviation of the two Gaussian distributions forming the real
                and imaginary components of the Rician noise distribution (see [2]_).

            References:
                .. [1] Descoteaux, Angelino, Fitzgibbons and Deriche (2007) Regularized,
                       fast and robust q-ball imaging. MRM, 58: 497-510
                .. [2] Gudbjartson and Patz (2008). The Rician distribution of noisy MRI
                       data. MRM 34: 910-914.

            Source:
                https://github.com/nipy/dipy/blob/108cd1137386462cda08438cddee15285131af08/dipy/sims/voxel.py#L82
        """
        if random.uniform(0, 1) <= self._exec_probability:
            orig_shape = ndarray.shape
            vol_flat = np.reshape(ndarray.copy(), (-1, ndarray.shape[-1]))

            if S0 is None:
                S0 = np.max(ndarray)

            for vox_idx, signal in enumerate(vol_flat):
                vol_flat[vox_idx] = self._apply(signal, snr=snr, S0=S0,
                                                noise_type=noise_type)

            return np.reshape(vol_flat, orig_shape)
        else:
            return ndarray

    def _apply(self, ndarray, snr, S0, noise_type):
        if snr is None:
            return ndarray

        sigma = S0 / snr

        noise_adder = {'gaussian': self._add_gaussian,
                       'rician': self._add_rician,
                       'rayleigh': self._add_rayleigh}

        noise1 = np.random.normal(0, sigma, size=ndarray.shape)

        if noise_type == 'gaussian':
            noise2 = None
        else:
            noise2 = np.random.normal(0, sigma, size=ndarray.shape)

        return noise_adder[noise_type](ndarray, noise1, noise2)

    @staticmethod
    def _add_gaussian(sig, noise1, noise2):
        """
        Adds one of the Gaussians to the sig and ignores the other one.
        """
        return sig + noise1

    @staticmethod
    def _add_rician(sig, noise1, noise2):
        """
        This does the same as abs(sig + complex(noise1, noise2))
        """
        return np.sqrt((sig + noise1) ** 2 + noise2 ** 2)

    @staticmethod
    def _add_rayleigh(sig, noise1, noise2):
        """
        The Rayleigh distribution is $\sqrt\{Gauss_1^2 + Gauss_2^2}$.
        """
        return sig + np.sqrt(noise1 ** 2 + noise2 ** 2)