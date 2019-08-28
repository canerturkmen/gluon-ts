# Standard library imports
from functools import reduce

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.dataset.loader import (
    BatchBuffer,
    TrainDataLoader,
    InferenceDataLoader,
)


class VariableLengthBatchBuffer(BatchBuffer):
    """
    BatchBuffer that pads when stacking

    Parameters
    ----------
    time_axis: int
        The dimension of component tensors that indexes the variable
        length sequences. That is, if working with (T, C) layouts,
        the time axis is 0. Default 0.
    """

    def __init__(self, *args, **kwargs):
        self.time_axis = kwargs.pop("time_axis", 0)
        super().__init__(*args, **kwargs)

    def _is_equal_length(self, xs):
        """
        Check if elements are scalars, have too few dimensions, or their
        time axes have equal length. In this case, fall back to super().stack
        """
        if not isinstance(xs[0], (mx.nd.NDArray, np.ndarray)):
            return True
        if xs[0].ndim <= self.time_axis:
            return True

        s = set([arr.shape[self.time_axis] for arr in xs])
        return len(s) <= 1

    # noinspection PyTypeChecker
    def _get_max_seq_length(self, xs) -> int:
        return reduce(max, (x.shape[self.time_axis] for x in xs))

    def _pad_arrays(self, xs):
        max_len = self._get_max_seq_length(xs)
        arr_lib = np if isinstance(xs[0], np.ndarray) else mx.nd

        xs_padded = []

        for x in xs:
            pad_size = max_len - x.shape[self.time_axis]

            pad_lengths = [(0, 0)] * x.ndim
            pad_lengths[self.time_axis] = (0, pad_size)

            xs_padded.append(
                arr_lib.pad(x, mode="constant", pad_width=pad_lengths)
            )

        return xs_padded

    def stack(self, xs):
        if not self._is_equal_length(xs):
            xs = self._pad_arrays(xs)
        return super().stack(xs)


class VariableLengthTrainDataLoader(TrainDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer = VariableLengthBatchBuffer(
            self.batch_size, self.ctx, self.float_type
        )


class VariableLengthInferenceDataLoader(InferenceDataLoader):
    def _get_buffer(self) -> BatchBuffer:
        return VariableLengthBatchBuffer(
            self.batch_size, self.ctx, self.float_type
        )
