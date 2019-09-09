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
import mxnet as mx
import numpy as np
import pytest

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.loader import DataLoader
from gluonts.tpp._loader import (
    VariableLengthTrainDataLoader,
    VariableLengthInferenceDataLoader,
)
from gluonts.tpp._transform import (
    ContinuousTimeUniformSampler,
    ContinuousTimeInstanceSplitter,
)

# Relative imports
# noinspection PyUnresolvedReferences
from .common import point_process_dataset, point_process_dataset_2


@pytest.fixture
def loader_factory():
    # noinspection PyTypeChecker
    def train_loader(
        dataset: ListDataset,
        prediction_interval_length: float,
        context_interval_length: float,
        is_train: bool = True,
        override_args: dict = None,
    ) -> DataLoader:

        if override_args is None:
            override_args = {}

        splitter = ContinuousTimeInstanceSplitter(
            future_interval_length=prediction_interval_length,
            past_interval_length=context_interval_length,
            train_sampler=ContinuousTimeUniformSampler(num_instances=10),
        )

        kwargs = dict(
            dataset=dataset,
            transform=splitter,
            batch_size=10,
            ctx=mx.cpu(),
            float_type=np.float32,
        )
        kwargs.update(override_args)

        if is_train:
            return VariableLengthTrainDataLoader(
                num_batches_per_epoch=22, **kwargs
            )
        else:
            return VariableLengthInferenceDataLoader(**kwargs)

    return train_loader


def test_train_loader_shapes(loader_factory, point_process_dataset_2):

    loader = loader_factory(point_process_dataset_2, 1.0, 1.5)

    d = next(iter(loader))

    field_names = [
        "past_target",
        "past_valid_length",
        "future_target",
        "future_valid_length",
    ]

    assert all([key in d for key in field_names])

    assert d["past_target"].shape[2] == d["future_target"].shape[2] == 2
    assert d["past_target"].shape[0] == d["future_target"].shape[0] == 10
    assert (
        d["past_valid_length"].shape[0]
        == d["future_valid_length"].shape[0]
        == 10
    )


def test_train_loader_length(loader_factory, point_process_dataset_2):

    loader = loader_factory(point_process_dataset_2, 1.0, 1.5)

    batches = list(iter(loader))

    assert len(batches) == 22


def test_inference_loader_shapes(loader_factory, point_process_dataset_2):

    loader = loader_factory(
        dataset=point_process_dataset_2,
        prediction_interval_length=1.0,
        context_interval_length=1.5,
        is_train=False,
        override_args={"batch_size": 10},
    )

    batches = list(iter(loader))

    assert len(batches) == 1

    d = batches[0]

    assert d["past_target"].shape[2] == 2
    assert d["past_target"].shape[0] == 3
    assert d["past_valid_length"].shape[0] == 3


def test_inference_loader_shapes_small_batch(
    loader_factory, point_process_dataset_2
):

    loader = loader_factory(
        dataset=point_process_dataset_2,
        prediction_interval_length=1.0,
        context_interval_length=1.5,
        is_train=False,
        override_args={"batch_size": 2},
    )

    batches = list(iter(loader))

    assert len(batches) == 2

    d = batches[0]

    assert d["past_target"].shape[2] == 2
    assert d["past_target"].shape[0] == 2
    assert d["past_valid_length"].shape[0] == 2


def test_train_loader_short_intervals(loader_factory, point_process_dataset_2):

    loader = loader_factory(
        dataset=point_process_dataset_2,
        prediction_interval_length=0.001,
        context_interval_length=0.0001,
        is_train=True,
        override_args={"batch_size": 5},
    )

    batches = list(iter(loader))

    d = batches[0]

    assert d["past_target"].shape[1] == d["future_target"].shape[1] == 1


def test_inference_loader_short_intervals(
    loader_factory, point_process_dataset_2
):

    loader = loader_factory(
        dataset=point_process_dataset_2,
        prediction_interval_length=0.001,
        context_interval_length=0.0001,
        is_train=False,
        override_args={"batch_size": 5},
    )

    batches = list(iter(loader))

    d = batches[0]

    assert d["past_target"].shape[1] == 1
