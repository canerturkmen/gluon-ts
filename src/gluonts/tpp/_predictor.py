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
from gluonts.tpp import VariableLengthInferenceDataLoader
from gluonts.tpp._forecast import PointProcessSampleForecast
from gluonts.transform import Transformation, FieldName


class PointProcessGluonPredictor(GluonPredictor):
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
            "ContinuousTimeSampleForecast",
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

        for batch in inference_data_loader:
            inputs = [batch[k] for k in self.input_names]

            # todo: tuple output
            outputs = self.prediction_net(*inputs).asnumpy()

            # todo: delete me
            # `outputs` is a numpy array of shape (N, S, T)

            # sample until enough point process trajectories are
            # collected
            num_collected_samples = outputs[0].shape[0]
            collected_samples = [outputs]
            while num_collected_samples < num_eval_samples:
                outputs = self.prediction_net(*inputs).asnumpy()

                collected_samples.append(outputs)
                num_collected_samples += outputs[0].shape[0]

            outputs = [
                np.concatenate(s)[:num_eval_samples]
                for s in zip(*collected_samples)
            ]
            assert len(outputs[0]) == num_eval_samples
            assert len(batch[FieldName.FORECAST_START]) == len(outputs)

            # todo: delete me
            # here, it's expected that dim 0 is the sample axis in each output
            # batch. outputs is a python list of output tuples for each
            # item in the batch. each item contains the first element in the
            # tuple which contains the samples along its first dimension.

            for i, output in enumerate(outputs):
                # FIXME: 2 outputs!

                yield PointProcessSampleForecast(
                    output[0],
                    valid_lengths=output[1],
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
