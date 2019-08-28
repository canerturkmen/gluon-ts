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
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.estimator import GluonEstimator, TrainOutput
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer
from gluonts.transform import Transformation

# Relative imports
from ._network import RMTPPTrainingNetwork
from ._loader import VariableLengthTrainDataLoader
from ._transform import (
    ContinuousTimeUniformSampler,
    ContinuousTimeInstanceSplitter,
)


class RMTPPEstimator(GluonEstimator):
    """
    The "Recurrent Marked Temporal Point Process" is a marked point process model
    where the conditional intensity function and the mark distribution are
    specified by a recurrent neural network, as described in [Duetal2016]_.

    .. [Duetal2016] Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M.,
        & Song, L. (2016, August). Recurrent marked temporal point processes: Embedding
        event history to vector. In Proceedings of the 22nd ACM SIGKDD International
        Conference on Knowledge Discovery and Data Mining (pp. 1555-1564). ACM.
    """

    @validated()
    def __init__(
        self,
        prediction_interval_length: float,
        context_interval_length: float,
        num_marks: int,
        embedding_dim: int = 5,
        trainer: Trainer = Trainer(hybridize=False),
        num_hidden_dimensions: int = 10,
        num_parallel_samples: int = 100,
        num_training_instances: int = 100,
    ) -> None:
        assert (
            not trainer.hybridize
        ), "RMTPP currently only supports the non-hybridized training"

        super().__init__(trainer=trainer)

        assert (
            prediction_interval_length > 0
        ), "The value of `prediction_interval_length` should be > 0"
        assert (
            context_interval_length is None or context_interval_length > 0
        ), "The value of `context_interval_length` should be > 0"
        assert (
            num_hidden_dimensions > 0
        ), "The value of `num_hidden_dimensions` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"
        assert num_marks > 0, "The value of `num_marks` should be > 0"
        assert (
            num_training_instances > 0
        ), "The value of `num_training_instances` should be > 0"

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_interval_length = prediction_interval_length
        self.context_interval_length = (
            context_interval_length
            if context_interval_length is not None
            else prediction_interval_length
        )
        self.num_marks = num_marks
        self.embedding_dim = embedding_dim
        self.num_parallel_samples = num_parallel_samples
        self.num_training_instances = num_training_instances

    def create_training_network(self) -> HybridBlock:
        return RMTPPTrainingNetwork(
            num_marks=self.num_marks,
            prediction_interval_length=self.prediction_interval_length,
            context_interval_length=self.context_interval_length,
            embedding_dim=self.embedding_dim,
            num_hidden_dimensions=self.num_hidden_dimensions,
        )

    def create_transformation(self) -> Transformation:
        return ContinuousTimeInstanceSplitter(
            past_interval_length=self.context_interval_length,
            future_interval_length=self.prediction_interval_length,
            train_sampler=ContinuousTimeUniformSampler(
                num_instances=self.num_training_instances
            ),
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        return Predictor(0, "H")  # todo: update after implementing

    def train_model(self, training_data: Dataset) -> TrainOutput:
        # we have to override the `train_model` method here
        transformation = self.create_transformation()

        transformation.estimate(iter(training_data))

        training_data_loader = VariableLengthTrainDataLoader(
            dataset=training_data,
            transform=transformation,
            batch_size=self.trainer.batch_size,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            ctx=self.trainer.ctx,
            float_type=self.float_type,
        )

        # ensure that the training network is created within the same MXNet
        # context as the one that will be used during training
        with self.trainer.ctx:
            trained_net = self.create_training_network()

        input_names = [
            "past_target",
            "future_target",
            "past_valid_length",
            "future_valid_length",
        ]

        self.trainer(
            net=trained_net,
            input_names=input_names,
            train_iter=training_data_loader,
        )

        with self.trainer.ctx:
            # ensure that the prediction network is created within the same MXNet
            # context as the one that was used during training
            return TrainOutput(
                transformation=transformation,
                trained_net=trained_net,
                predictor=self.create_predictor(transformation, trained_net),
            )
