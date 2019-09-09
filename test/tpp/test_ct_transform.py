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
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.tpp._transform import (
    ContinuousTimePointSampler,
    ContinuousTimeUniformSampler,
    ContinuousTimeInstanceSplitter,
)

# Relative imports
from .common import point_process_dataset


class MockCTSampler(ContinuousTimePointSampler):
    # noinspection PyMissingConstructor,PyUnusedLocal
    def __init__(self, ret_values, *args, **kwargs):
        self._ret_values = ret_values

    def __call__(self, *args, **kwargs):
        return np.array(self._ret_values)


def test_ctsplitter_mask_sorted(point_process_dataset):
    d = next(iter(point_process_dataset))

    ia_times = d["target"][0, :]

    ts = np.cumsum(ia_times)

    splitter = ContinuousTimeInstanceSplitter(
        2, 1, train_sampler=ContinuousTimeUniformSampler(num_instances=10)
    )

    # no boundary conditions
    res = splitter._mask_sorted(ts, 1, 2)
    assert all([a == b for a, b in zip([2, 3, 4], res)])

    # lower bound equal, exclusive of upper bound
    res = splitter._mask_sorted(np.array([1, 2, 3, 4, 5, 6]), 1, 2)
    assert all([a == b for a, b in zip([0], res)])


def test_ctsplitter_no_train_last_point(point_process_dataset):
    splitter = ContinuousTimeInstanceSplitter(
        2, 1, train_sampler=ContinuousTimeUniformSampler(num_instances=10)
    )

    iter_de = splitter(point_process_dataset, is_train=False)

    d_out = next(iter(iter_de))

    assert "future_target" not in d_out
    assert "future_valid_length" not in d_out
    assert "past_target" in d_out
    assert "past_valid_length" in d_out

    assert d_out["past_valid_length"] == 6
    assert np.allclose(
        [0.1, 0.5, 0.3, 0.3, 0.2, 0.1], d_out["past_target"][..., 0], atol=0.01
    )


def test_ctsplitter_train_correct(point_process_dataset):
    splitter = ContinuousTimeInstanceSplitter(
        1,
        1,
        train_sampler=MockCTSampler(
            ret_values=[1.01, 1.5, 1.99], num_instances=3
        ),
    )

    iter_de = splitter(point_process_dataset, is_train=True)

    outputs = list(iter_de)

    assert outputs[0]["past_valid_length"] == 2
    assert outputs[0]["future_valid_length"] == 3

    assert np.allclose(
        outputs[0]["past_target"], np.array([[0.19, 0.7], [0, 1]]).T
    )
    assert np.allclose(
        outputs[0]["future_target"], np.array([[0.09, 0.5, 0.3], [2, 0, 1]]).T
    )

    assert outputs[1]["past_valid_length"] == 2
    assert outputs[1]["future_valid_length"] == 4

    assert outputs[2]["past_valid_length"] == 3
    assert outputs[2]["future_valid_length"] == 3


def test_ctsplitter_train_correct_out_count(point_process_dataset):

    # produce new TPP data by shuffling existing TS instance
    def shuffle_iterator(num_duplications=5):
        for entry in point_process_dataset:
            for i in range(num_duplications):
                d = dict.copy(entry)
                d["target"] = np.random.permutation(d["target"].T).T
                yield d

    splitter = ContinuousTimeInstanceSplitter(
        1,
        1,
        train_sampler=MockCTSampler(
            ret_values=[1.01, 1.5, 1.99], num_instances=3
        ),
    )

    iter_de = splitter(shuffle_iterator(), is_train=True)

    outputs = list(iter_de)

    assert len(outputs) == 5 * 3


def test_ctsplitter_train_samples_correct_times(point_process_dataset):

    splitter = ContinuousTimeInstanceSplitter(
        1.25, 1.25, train_sampler=ContinuousTimeUniformSampler(20)
    )

    iter_de = splitter(point_process_dataset, is_train=True)

    assert all(
        [
            (
                pd.Timestamp("2011-01-01 01:15:00")
                <= d["forecast_start"]
                <= pd.Timestamp("2011-01-01 01:45:00")
            )
            for d in iter_de
        ]
    )


def test_ctsplitter_train_short_intervals(point_process_dataset):
    splitter = ContinuousTimeInstanceSplitter(
        0.01,
        0.01,
        train_sampler=MockCTSampler(
            ret_values=[1.01, 1.5, 1.99], num_instances=3
        ),
    )

    iter_de = splitter(point_process_dataset, is_train=True)

    for d in iter_de:
        assert d["future_valid_length"] == d["past_valid_length"] == 0
        assert np.prod(np.shape(d["past_target"])) == 0
        assert np.prod(np.shape(d["future_target"])) == 0
