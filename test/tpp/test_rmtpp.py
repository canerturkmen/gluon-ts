# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Third-party imports
import pytest
import mxnet as mx
import numpy as np
from gluonts.model.predictor import Predictor
from gluonts.tpp import RMTPPEstimator
from gluonts.trainer import Trainer
from mxnet import nd

# First-party imports
from gluonts.tpp._network import RMTPPTrainingNetwork


def _allclose(a: nd.NDArray, b: nd.NDArray):
    return np.allclose(a.asnumpy(), b.asnumpy(), atol=1e-6)


def test_log_likelihood():
    mx.rnd.seed(seed_state=1234)

    smodel = RMTPPTrainingNetwork(
        num_marks=3, prediction_interval_length=2, context_interval_length=1
    )

    smodel.collect_params().initialize()

    past_lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
    past_marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
    past_valid_length = nd.array([3, 4])

    future_lags = nd.array([[0.8, 0.1, 0.1, 0.12], [0.4, 0.25, 0.1, 0.12]])
    future_marks = nd.array([[2, 2, 2, 2], [0, 0, 1, 2]])
    future_valid_length = nd.array([3, 4])

    smodel(
        nd.stack(past_lags, past_marks, axis=-1),
        nd.stack(future_lags, future_marks, axis=-1),
        past_valid_length,
        future_valid_length,
    )

    assert False


def test_rmtpp_disallows_hybrid():
    mx.rnd.seed(seed_state=1234)

    with pytest.raises(NotImplementedError):
        smodel = RMTPPTrainingNetwork(
            num_marks=3,
            prediction_interval_length=2,
            context_interval_length=1,
        )
        smodel.hybridize()


#
# @pytest.mark.parametrize("hybridize", [True, False])
# def test_log_likelihood_max_time(hybridize):
#     mx.rnd.seed(seed_state=1234)
#
#     smodel = RMTPPBlock(num_marks=3, sequence_length=2)
#     if hybridize:
#         smodel.hybridize()
#
#     smodel.collect_params().initialize()
#
#     lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
#     marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
#
#     valid_length = nd.ones(shape=(4,)) * 2
#     max_time = nd.ones(shape=(4,)) * 5
#
#     assert _allclose(
#         smodel(lags, marks, valid_length, max_time),
#         nd.array([-4.2177677, -4.146564, -3.911864, -3.9754848]),
#     )
#
#
# @pytest.mark.parametrize("hybridize", [True, False])
# def test_log_likelihood_valid_length(hybridize):
#     mx.rnd.seed(seed_state=1234)
#
#     smodel = RMTPPBlock(num_marks=3, sequence_length=2)
#     if hybridize:
#         smodel.hybridize()
#
#     smodel.collect_params().initialize()
#
#     lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
#     marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
#
#     valid_length = nd.array([1, 2, 1, 1])
#     max_time = nd.ones(shape=(4,)) * 5
#
#     assert _allclose(
#         smodel(lags, marks, valid_length, max_time),
#         nd.array([-2.6500425, -4.146564, -2.6500664, -2.6819286]),
#     )
#
#
# @pytest.mark.parametrize("hybridize", [True, False])
# def test_sampler_shapes_correct(hybridize):
#     mx.rnd.seed(seed_state=1234)
#
#     smodel = RMTPPBlock(num_marks=3, sequence_length=2)
#     if hybridize:
#         smodel.hybridize()
#
#     smodel.collect_params().initialize()
#
#     lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
#     marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
#     valid_length = nd.array([1, 2, 1, 1])
#     max_time = nd.ones(shape=(4,)) * 5
#
#     smodel(lags, marks, valid_length, max_time)
#
#     sampler = RMTPPSampler(smodel)
#
#     ia_times, marks, valid_length_samp = sampler.ogata_sample(
#         5., batch_size=12
#     )
#
#     assert marks.asnumpy().max() < smodel.num_marks
#     assert marks.shape[1] == 12
#     assert ia_times.shape[1] == 12
#     assert valid_length_samp.shape[0] == 12
