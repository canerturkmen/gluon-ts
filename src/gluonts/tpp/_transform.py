# Standard library imports
from typing import Iterator

# Third-party imports
import numpy as np

# First-party imports
from gluonts.dataset.common import DataEntry
from gluonts.transform import FlatMapTransformation


class ContinuousTimePointSampler:
    """
    Abstract class for "continuous time" samplers, which, given a lower bound
    and upper bound, sample points in continuous time from a specified interval.
    """

    def __init__(self, num_instances: int) -> None:
        self.num_instances = num_instances

    def __call__(self, length: float, a: float, b: float) -> np.ndarray:
        """

        Parameters
        ----------
        length
            The length of the interval to sample from. Must be greater than
            b.
        a
            The lower bound (minimum value that the sampled point can take)
        b
            Upper bound. Must be greater than a.
        """
        raise NotImplementedError


class ContinuousTimeUniformSampler(ContinuousTimePointSampler):
    def __call__(self, length: float, a: float, b: float) -> np.ndarray:
        assert a <= b
        return np.random.rand(self.num_instances) * (b - a) + a


class ContinuousTimeInstanceSplitter(FlatMapTransformation):
    """
    Selects training instances by slicing "intervals" from a continous-time
    process instantiation. Concretely, the input data is expected to describe an
    instantiation from a point (or jump) process, with the "target"
    identifying inter-arrival times and other features (marks), as described
    in detail below.

    The transformation is analogous to its discrete counterpart `InstanceSplitter`.

    - Does not allow "incomplete" records. These cause problems in TPP.
    - Works on intervals
    - Outputs a certain layout (NTC)
    - Does not accept time-series fields as these would typically not be
      available in TPP data.

    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.

    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).

    target -> past_target and future_target

    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.

    Convention: time axis is always the last axis.

    Parameters
    ----------

    past_interval_length
        length of the interval seen before making prediction
    future_interval_length
        length of the interval that must be predicted
    train_sampler
        instance sampler that provides sampling indices given a time-series
    target_field
        field containing the target
    start_field
        field containing the start date of the of the point process observation
    end_field
        field containing the end date of the point process observation
    forecast_start_field
        output field that will contain the time point where the forecast starts
    """

    # @validated()
    def __init__(
        self,
        past_interval_length: float,
        future_interval_length: float,
        train_sampler: ContinuousTimePointSampler,
        target_field: str = "target",
        start_field: str = "start",
        end_field: str = "end",
        forecast_start_field: str = "forecast_start",
    ) -> None:

        assert future_interval_length > 0

        self.train_sampler = train_sampler
        self.past_interval_length = past_interval_length
        self.future_interval_length = future_interval_length
        self.target_field = target_field
        self.start_field = start_field
        self.end_field = end_field
        self.forecast_start_field = forecast_start_field

    # noinspection PyMethodMayBeStatic
    def _mask_sorted(self, a: np.ndarray, lb: float, ub: float):
        start = np.searchsorted(a, lb)
        end = np.searchsorted(a, ub)
        return np.arange(start, end)

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:

        assert data[self.start_field].freq == data[self.end_field].freq

        total_interval_length = (
            data[self.end_field] - data[self.start_field]
        ) / data[self.start_field].freq.delta

        # sample forecast start times in continuous time
        if is_train:
            if total_interval_length < (
                self.future_interval_length + self.past_interval_length
            ):
                sampling_times: np.ndarray = np.array([])
            else:
                sampling_times = self.train_sampler(
                    total_interval_length,
                    self.past_interval_length,
                    total_interval_length - self.future_interval_length,
                )
        else:
            sampling_times = np.array([total_interval_length])

        # keep a copy of the target
        d = data.copy()

        ia_times = d[self.target_field][0, :]
        marks = d[self.target_field][1:, :]

        ts = np.cumsum(ia_times)
        assert ts[-1] < total_interval_length, (
            "Target interarrival times provided are inconsistent with "
            "start and end timestamps."
        )

        # select field names that will be included in outputs
        keep_cols = {
            k: v
            for k, v in d.items()
            if k not in [self.target_field, self.start_field, self.end_field]
        }

        for future_start in sampling_times:

            r = dict()

            past_start = future_start - self.past_interval_length
            future_end = future_start + self.future_interval_length

            assert past_start >= 0
            assert future_end <= total_interval_length

            past_mask = self._mask_sorted(ts, past_start, future_start)
            future_mask = self._mask_sorted(ts, future_start, future_end)

            past_ia_times = np.diff(np.r_[0, ts[past_mask] - past_start])[
                np.newaxis
            ]  # expand_dims(axis=0)
            future_ia_times = np.diff(
                np.r_[0, ts[future_mask] - future_start]
            )[np.newaxis]

            r[f"past_{self.target_field}"] = np.concatenate(
                [past_ia_times, marks[:, past_mask]], axis=0
            ).transpose()

            r[f"future_{self.target_field}"] = np.concatenate(
                [future_ia_times, marks[:, future_mask]], axis=0
            ).transpose()

            r["past_valid_length"] = np.array([len(past_mask)])
            r["future_valid_length"] = np.array([len(future_mask)])

            r[self.forecast_start_field] = (
                d[self.start_field]
                + d[self.start_field].freq.delta * future_start
            )

            # include other fields
            r.update(keep_cols.copy())

            yield r
