from typing import Dict, Optional, Union, cast

# Third-party imports
import mxnet as mx
import numpy as np
import pandas as pd

# First-party imports
from gluonts.model.forecast import OutputType, Forecast, Config
from pandas import to_timedelta


class PointProcessSampleForecast(Forecast):
    """
    Sample forecast object used for temporal point process inference.
    Differs from standard forecast objects as it does not implement
    fixed length samples. Each sample has a variable length, that is
    kept in a separate :code:`valid_length` attribute.

    Importantly, PointProcessSampleForecast does not implement some
    methods (such as :code:`quantile` or :code:`plot`) that are available
    in discrete time forecasts.

    Parameters
    ----------
    samples
        A multidimensional array of samples, of shape
        (number_of_samples, max_pred_length, target_dim) or
        (number_of_samples, max_pred_length). For marked TPP, the
        target_dim is 2 with the first element keeping interarrival times
        and the latter marks. If samples are two-dimensional, each entry
        stands for the interarrival times in a (unmarked) TPP sample.
    valid_length
        An array of integers denoting the valid lengths of each sample
        in :code:`samples`. That is, :code:`valid_length[0] == 2` implies
        that only the first two entries of :code:`samples[0, ...]` are
        valid "points".
    start_date
        Starting timestamp of the sample
    freq
        The time unit of interarrival times
    prediction_interval_length
        The length of the prediction interval for which samples were drawn.
    end_date
        Optionally, the end timestamp of the sample. If not given, computed
        from start_date, prediction_interval_length and freq.
    item_id
        Item ID, if available.
    info
        Optional dictionary of additional information.
    """

    prediction_length = cast(int, None)  # not used
    prediction_interval_length: float

    # not used
    mean = None
    _index = None

    def __init__(
        self,
        samples,
        valid_length,
        start_date: pd.Timestamp,
        freq: str,
        prediction_interval_length: float,
        end_date: Optional[pd.Timestamp],
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ) -> None:
        assert isinstance(
            samples, (np.ndarray, mx.ndarray.ndarray.NDArray)
        ), "samples should be either a numpy or an mxnet array"
        assert (
            len(np.shape(samples)) == 2 or len(np.shape(samples)) == 3
        ), "samples should be a 2-dimensional or 3-dimensional array. Dimensions found: {}".format(
            len(np.shape(samples))
        )
        self.samples = (
            samples if (isinstance(samples, np.ndarray)) else samples.asnumpy()
        )

        assert isinstance(
            valid_length, (np.ndarray, mx.ndarray.ndarray.NDArray)
        ), "samples should be either a numpy or an mxnet array"
        assert (
            len(valid_length.shape) == 1
        ), "valid_length should be a 1-dimensional array"
        assert (
            valid_length.shape[0] == samples.shape[0]
        ), "valid_length and samples should have compatible dimensions"
        self.valid_length = (
            valid_length
            if (isinstance(valid_length, np.ndarray))
            else valid_length.asnumpy()
        )

        self._dim = len(np.shape(samples))
        self.item_id = item_id
        self.info = info

        assert isinstance(
            start_date, pd.Timestamp
        ), "start_date should be a pandas Timestamp object"
        self.start_date = start_date

        assert isinstance(freq, str), "freq should be a string"
        self.freq = freq

        assert (
            prediction_interval_length > 0
        ), "prediction_interval_length is greater than 0"
        self.prediction_interval_length = prediction_interval_length

        assert (
            isinstance(end_date, pd.Timestamp) or end_date is None
        ), "end_date should be a pandas Timestamp object"

        self.end_date = (
            (
                start_date
                + to_timedelta(1, self.freq) * prediction_interval_length
            )
            if end_date is None
            else end_date
        )

    def dim(self) -> int:
        return self._dim

    @property
    def index(self) -> pd.DatetimeIndex:
        raise AttributeError(
            "Datetime index not defined for point process samples"
        )

    def as_json_dict(self, config: "Config") -> dict:
        result = super().as_json_dict(config)

        if OutputType.samples in config.output_types:
            result["samples"] = self.samples.tolist()
            result["valid_length"] = self.valid_length.tolist()

        return result

    def __repr__(self):
        return ", ".join(
            [
                f"PointProcessSampleForecast({self.samples!r})",
                f"{self.valid_length!r}",
                f"{self.start_date!r}",
                f"{self.end_date!r}",
                f"{self.freq!r}",
                f"item_id={self.item_id!r}",
                f"info={self.info!r})",
            ]
        )

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        raise NotImplementedError()

    def plot(self, **kwargs):
        raise NotImplementedError()

    def copy_dim(self, dim: int):
        raise NotImplementedError()
