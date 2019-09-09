# Standard library imports
from pathlib import Path
from typing import Dict, Iterator, List, Optional, cast

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import DataBatch
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import GluonPredictor, SymbolBlockPredictor
from gluonts.transform import Transformation, FieldName

# Relative imports
from ._loader import VariableLengthInferenceDataLoader
from ._forecast import PointProcessSampleForecast


class PointProcessGluonPredictor(GluonPredictor):
    """
    Predictor object for marked temporal point process models.

    TPP predictions differ from standard discrete-time models in several
    regards. First, at least for now, only sample forecasts implementing
    PointProcessSampleForecast are available. Similar to TPP Estimator
    objects, the Predictor works with :code:`prediction_interval_length`
    as opposed to :code:`prediction_length`.

    The predictor also accounts for the fact that the prediction network
    outputs a 2-tuple of Tensors, for the samples themselves and their
    `valid_length`.

    Finally, this class uses a VariableLengthInferenceDataLoader as opposed
    to the default InferenceDataLoader.

    Parameters
    ----------
    prediction_interval_length
        The length of the prediction interval
    """

    def __init__(
        self,
        input_names: List[str],
        prediction_net: mx.gluon.Block,
        batch_size: int,
        prediction_interval_length: float,
        freq: str,
        ctx: mx.Context,
        input_transform: Transformation,
        float_type: DType = np.float32,
        forecast_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            input_names,
            prediction_net,
            batch_size,
            np.ceil(prediction_interval_length),  # for validation only
            freq,
            ctx,
            input_transform,
            None,  # no output transform
            float_type,
            "PointProcessSampleForecast",
            forecast_kwargs,
        )

        self.prediction_length = cast(int, None)  # not used by TPP predictor
        self.prediction_interval_length = prediction_interval_length

    def hybridize(self, batch: DataBatch) -> None:
        raise NotImplementedError(
            "Point process models are currently not hybridizable"
        )

    def as_symbol_block_predictor(
        self, batch: DataBatch
    ) -> SymbolBlockPredictor:
        raise NotImplementedError(
            "Point process models are currently not hybridizable"
        )

    def predict(
        self,
        dataset: Dataset,
        num_eval_samples: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Forecast]:

        inference_data_loader = VariableLengthInferenceDataLoader(
            dataset,
            self.input_transform,
            self.batch_size,
            ctx=self.ctx,
            float_type=self.float_type,
        )

        if not num_eval_samples:
            num_eval_samples = self.prediction_net.num_parallel_samples

        for batch in inference_data_loader:
            inputs = [batch[k] for k in self.input_names]

            outputs, valid_length = (
                x.asnumpy() for x in self.prediction_net(*inputs)
            )

            # sample until enough point process trajectories are collected
            num_collected_samples = outputs[0].shape[0]
            collected_samples, collected_vls = [outputs], [valid_length]
            while num_collected_samples < num_eval_samples:
                outputs, valid_length = (
                    x.asnumpy() for x in self.prediction_net(*inputs)
                )

                collected_samples.append(outputs)
                collected_vls.append(valid_length)

                num_collected_samples += outputs[0].shape[0]

            outputs = [
                np.concatenate(s)[:num_eval_samples]
                for s in zip(*collected_samples)
            ]
            valid_length = [
                np.concatenate(s)[:num_eval_samples]
                for s in zip(*collected_vls)
            ]

            assert len(outputs[0]) == num_eval_samples
            assert len(valid_length[0]) == num_eval_samples
            assert len(batch[FieldName.FORECAST_START]) == len(outputs)

            for i, output in enumerate(outputs):
                yield PointProcessSampleForecast(
                    output,
                    valid_length=valid_length[i],
                    start_date=batch[FieldName.FORECAST_START][i],
                    freq=self.freq,
                    prediction_interval_length=self.prediction_interval_length,
                    end_date=None,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )

    def serialize_prediction_net(self, path: Path) -> None:
        raise NotImplementedError()
