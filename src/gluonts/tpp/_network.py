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

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


# noinspection PyAbstractClass
class RMTPPNetworkBase(mx.gluon.HybridBlock):
    """
    Implements the graph of a Recurrent Multivariate Temporal Point Process
    """

    def __init__(
        self,
        num_marks: int,
        prediction_interval_length: float,
        context_interval_length: float,
        embedding_dim: int = 5,
        num_hidden_dimensions: int = 10,
        **kwargs,
    ) -> None:
        """
        Initialize an RMTPP Network

        Parameters
        ----------
        num_marks
            Number of discrete marks (correlated processes),
        prediction_interval_length
            Prediction and context lengths take slightly different semantics for
            point process data. Namely, the prediction length is the length of the
            prediction time interval
        context_interval_length
            Length of the time interval on which the network is conditioned
        embedding_dim
            Dimension of the vector embeddings of marks (used only as input)
        num_hidden_dimensions
            Hidden units in the LSTM
        """
        super().__init__(**kwargs)

        self.num_marks = num_marks
        self.num_lstm_hidden = num_hidden_dimensions
        self.prediction_interval_length = prediction_interval_length
        self.context_interval_length = context_interval_length

        with self.name_scope():
            self.decay_bias = self.params.get(
                "decay_bias",
                shape=(1,),
                allow_deferred_init=False,
                wd_mult=0.0,
                init=mx.init.Constant(-10.0),
            )

            self.embedding = mx.gluon.nn.Embedding(
                input_dim=num_marks, output_dim=embedding_dim
            )
            self.lstm = mx.gluon.rnn.LSTMCell(
                num_hidden_dimensions, input_size=embedding_dim + 1
            )

            self.mtpp_ground = mx.gluon.nn.Dense(
                1, in_units=num_hidden_dimensions, flatten=False
            )
            self.mtpp_mark = mx.gluon.nn.Dense(
                num_marks, in_units=num_hidden_dimensions, flatten=False
            )


class RMTPPTrainingNetwork(RMTPPNetworkBase):

    # noinspection PyMethodOverriding,PyPep8Naming,PyIncorrectDocstring
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_valid_length: Tensor,
        future_valid_length: Tensor,
        decay_bias: Tensor = None,
    ) -> Tensor:
        """
        Computes the RMTPP negative log likelihood.

        Parameters
        ----------
        F
            MXNet backend.
        past_target
            Tensor with past observations.
            Shape: (batch_size, past_max_sequence_length, target_dim).
        future_target
            Tensor with future observations.
            Shape: (batch_size, future_max_sequence_length, target_dim).
        past_valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)
        future_valid_length
            The `valid_length` or number of valid entries in the future_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size,).
        """
        if F is mx.sym:
            raise ValueError(
                "The RMTPP model currently doesn't support hybridization."
            )

        past_ia_times, past_marks = F.split(
            past_target, num_outputs=2, axis=-1
        )
        future_ia_times, future_marks = F.split(
            future_target, num_outputs=2, axis=-1
        )

        # fixme: this is a hack to avoid shape and type-related issues from
        # the transformation
        past_valid_length = past_valid_length.reshape(-1).astype(
            past_ia_times.dtype
        )
        future_valid_length = future_valid_length.reshape(-1).astype(
            past_ia_times.dtype
        )

        # compute the LSTM state at the end of the conditioning range
        # and the excess time (the time between the last point and the
        # end of the conditioning interval)
        _, lstm_cond_state = self.lstm.unroll(
            length=past_target.shape[1],
            inputs=F.concat(  # lstm input
                past_ia_times, self.embedding(past_marks.sum(-1)), dim=-1
            ),
            layout="NTC",
            merge_outputs=True,
            valid_length=past_valid_length,
        )

        cond_time_last = F.SequenceMask(
            data=past_ia_times,
            sequence_length=past_valid_length,
            axis=1,
            use_sequence_length=True,
        ).sum(1)

        cond_time_remaining = (
            F.ones_like(cond_time_last) * self.context_interval_length
            - cond_time_last
        ).expand_dims(-1)

        # update the first IA time of the first point in the prediction range
        # to include the remaining time from the end of "context"
        first_ia_time = F.slice_axis(future_ia_times, axis=1, begin=0, end=1)
        first_ia_time = F.broadcast_add(first_ia_time, cond_time_remaining)

        future_ia_times_updated = F.concat(
            first_ia_time,
            F.slice_axis(future_ia_times, axis=1, begin=1, end=None),
        )

        # unroll the LSTM for the prediction range and shift the
        # LSTM outputs by one point

        lstm_out, lstm_out_state = self.lstm.unroll(
            future_ia_times.shape[1],
            F.concat(  # lstm input
                future_ia_times_updated,
                self.embedding(future_marks.sum(-1)),
                dim=-1,
            ),
            layout="NTC",
            begin_state=lstm_cond_state,
            merge_outputs=True,
            valid_length=future_valid_length,
        )

        lstm_out_shifted = F.slice_axis(
            F.concat(lstm_cond_state[0].expand_dims(1), lstm_out, dim=1),
            axis=1,
            begin=0,
            end=-1,
        )

        # map to parameters for conditional distributions of the
        # next mark and point
        mtpp_ia_times = self.mtpp_ground(lstm_out_shifted)  # (N, T, 1)
        mtpp_mark = F.log_softmax(
            self.mtpp_mark(lstm_out_shifted)
        )  # (N, T, K)

        # compute the log intensity and the compensator - (1,)
        beta = -F.Activation(decay_bias, "softrelu")
        beta_ia_times = F.broadcast_mul(
            future_ia_times_updated, beta
        )  # (N, T, 1)

        log_intensity = (
            mtpp_ia_times
            + beta_ia_times
            + F.pick(mtpp_mark, future_marks, keepdims=True)
        )
        compensator = F.broadcast_div(F.exp(mtpp_ia_times), beta) * (
            F.exp(beta_ia_times) - 1
        )

        # we now have to add the extra compensator, the compensator for the last point
        # in the prediction range and the end of the prediction interval
        def _mask(x):
            return F.SequenceMask(
                data=x,
                sequence_length=future_valid_length,
                axis=1,
                use_sequence_length=True,
            )

        pred_time_last = _mask(future_ia_times).sum(1)
        pred_time_remaining = (
            F.ones_like(pred_time_last) * self.prediction_interval_length
            - pred_time_last
        ).expand_dims(
            -1
        )  # (N, 1, 1)

        last_mtpp_out = self.mtpp_ground(lstm_out_state[0].expand_dims(1))

        pred_last_compensator = (
            F.broadcast_div(F.exp(last_mtpp_out), beta)
            * (F.exp(F.broadcast_mul(pred_time_remaining, beta)) - 1)
        ).sum(1)

        log_likelihood = (
            _mask(log_intensity).sum(1)
            - _mask(compensator).sum(1)
            - pred_last_compensator
        ).sum(-1)

        return log_likelihood


class RMTPPPredictionNetwork(RMTPPNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    # noinspection PyMethodOverriding,PyPep8Naming,PyIncorrectDocstring
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_valid_length: Tensor,
        decay_bias: Tensor = None,
    ) -> Tensor:
        """
        Draw forward samples from the RMTPP model via the thinning algorithm,
        a.k.a. Ogata's thinning sampler.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        past_valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        Tensor
            Prediction sample. Shape: (samples, batch_size, prediction_length).
        """
        pass
