import pytest
import numpy as np
import pandas as pd
import mxnet as mx

from gluonts.dataset.common import ListDataset
from gluonts.tpp._estimator import RMTPPEstimator
from gluonts.tpp._loader import VariableLengthTrainDataLoader
from gluonts.tpp._transform import (
    ContinuousTimeInstanceSplitter,
    ContinuousTimeUniformSampler,
)


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
