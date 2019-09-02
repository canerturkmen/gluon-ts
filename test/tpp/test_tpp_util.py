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
import numpy as np
import pandas as pd
import mxnet as mx

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.tpp._estimator import RMTPPEstimator
from gluonts.tpp._loader import VariableLengthTrainDataLoader
from gluonts.tpp._transform import (
    ContinuousTimeInstanceSplitter,
    ContinuousTimeUniformSampler,
)
from gluonts.trainer import Trainer


@pytest.fixture()
def lds():

    df = pd.read_pickle("~/code/data/ncedc/ncedc.pkl")

    ia_times = np.diff(df["dt"]).astype(float) / (60 * 60 * 1e9)
    marks = df["Cluster"]

    lds = ListDataset(
        [
            {
                "target": np.c_[ia_times, marks[1:]].T,
                "start": pd.Timestamp("2011-01-01 00:00:00", freq="H"),
                "end": pd.Timestamp("2015-01-01 00:00:00", freq="H"),
            }
        ],
        freq="H",
        one_dim_target=False,
    )

    return lds


def test_instance_splitter(lds):

    splitter = ContinuousTimeInstanceSplitter(
        past_interval_length=72,
        future_interval_length=24,
        train_sampler=ContinuousTimeUniformSampler(num_instances=1),
    )

    np.random.seed(1234)
    # with pytest.raises(ValueError):
    z = splitter(lds, is_train=True)

    next(iter(z))


def test_loader(lds):
    splitter = ContinuousTimeInstanceSplitter(
        past_interval_length=72,
        future_interval_length=24,
        train_sampler=ContinuousTimeUniformSampler(num_instances=1),
    )

    np.random.seed(1234)

    training_data_loader = VariableLengthTrainDataLoader(
        dataset=lds,
        transform=splitter,
        batch_size=10,
        num_batches_per_epoch=5,
        ctx=mx.cpu(),
        float_type=np.float32,
    )

    z = next(iter(training_data_loader))

    return z


def test_train(lds):

    estimator = RMTPPEstimator(
        prediction_interval_length=24,
        context_interval_length=24 * 5,
        num_marks=10,
        embedding_dim=3,
    )

    np.random.seed(1234)

    z = estimator.train(lds)

    assert z is None


def test_predictor(lds):

    estimator = RMTPPEstimator(
        prediction_interval_length=24,
        context_interval_length=24 * 5,
        num_marks=10,
        embedding_dim=3,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10, hybridize=False),
    )

    np.random.seed(1234)

    z = estimator.train(lds)

    assert isinstance(z, Predictor)
