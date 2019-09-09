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

# Standard library imports
from typing import Tuple, List

# Third-party imports
import mxnet as mx
from mxnet import nd

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


# noinspection PyAbstractClass
class RMTPPNetworkBase(mx.gluon.HybridBlock):
    """
    Implements the graph of a Recurrent Multivariate Temporal Point Process
    (RMTPP), with a single hidden-layer LSTM recurrent neural network.

    Parameters
    ----------
    num_marks
        Number of discrete marks (correlated processes), that are available
        in the data
    prediction_interval_length
        The length of the total time interval that is in the prediction
        range. Note that in contrast to discrete-time models in the rest
        of GluonTS, the network is trained to predict an interval, in
        continuous time.
    context_interval_length
        Length of the time interval on which the network is conditioned
    embedding_dim
        Dimension of vector embeddings of marks (used only as input)
    num_hidden_dimensions
        Number of hidden units in the LSTM
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

    def hybridize(self, active=True, **kwargs):
        if active:
            raise NotImplementedError(
                "RMTPP blocks do not support hybridization"
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
        Computes the RMTPP negative log likelihood loss.

        As the model is trained on past (resp. future) or context
        (resp. prediction) "intervals" as opposed to fixed-length "sequences",
        the number of data points available varies across observations. To
        account for this, data is made available to the training network as a
        "ragged" tensor. The number of valid entries in each sequence is provided
        in a separate variable, :code:`xxx_valid_length`.

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
    ) -> Tuple[Tensor, Tensor]:
        """
        Draw forward samples from the RMTPP model via the thinning algorithm,
        a.k.a. Ogata's thinning sampler.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim). Has to comply
            with :code:`self.context_interval_length`.
        past_valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        samples: Tensor
            Prediction sample.
            Shape: (samples, batch_size, max_prediction_length, target_dim).
        samples_valid_length: Tensor
            The number of valid entries in the time axis of each sample.
            Shape (samples, batch_size)
        """
        if F is mx.sym:
            raise ValueError(
                "The RMTPP model currently doesn't support hybridization."
            )

        beta = -F.Activation(decay_bias, "softrelu")
        ctx = beta.context

        batch_size = past_target.shape[0]
        assert (
            past_target.shape[-1] == 2
        ), "RMTPP data should have two target_dim, interarrival times and marks"

        sample_dim_size = batch_size * self.num_parallel_samples

        max_time = nd.array([self.prediction_interval_length], ctx=ctx)
        time_samples: List[nd.NDArray] = []
        mark_samples: List[nd.NDArray] = []
        # noinspection PyTypeChecker
        masks: List[nd.NDArray] = []

        # condition the prediction network on the past

        num_observations = past_target.shape[1]

        past_ia_times, past_marks = F.split(
            past_target, num_outputs=2, axis=-1
        )
        past_valid_length = past_valid_length.reshape(-1).astype(
            past_ia_times.dtype
        )

        # condition the model on the past
        _, lstm_cond_state = self.lstm.unroll(
            length=num_observations,
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

        # duplicate the LSTM states and the times by the number of
        # samples required

        sample_dim_size = self.num_parallel_samples * num_observations

        lstm_out, lstm_mem = lstm_cond_state  # (N, T), (N, T)

        time, tau = (
            nd.zeros(sample_dim_size, ctx=ctx),
            nd.zeros(sample_dim_size, ctx=ctx),
        )

        while nd.sum(time < max_time) > 0:

            # propose the next points
            log_int_beginning = self.mtpp_ground(lstm_out)
            lda_proposal = nd.exp(log_int_beginning).reshape(-1)

            tau = nd.random.exponential(
                scale=1.0 / lda_proposal, ctx=ctx
            )  # draw the proposed next point, (N,)

            # compute the intensity in the proposed points, for all marks
            log_mark_prob = nd.log_softmax(self.mtpp_mark(lstm_out))

            all_intensities = nd.sum(
                nd.exp(
                    log_int_beginning
                    + nd.expand_dims(beta * tau, 1).tile(
                        (1, 1, self.num_marks)
                    )
                    + log_mark_prob
                ),
                0,
            )

            p = all_intensities / nd.expand_dims(
                all_intensities.sum(-1), -1
            )  # distributions for next points
            m = nd.random.multinomial(p, dtype="float32")  # sample the marks

            # advance the time
            time += tau
            tau += tau

            # accept-reject
            u = nd.random.uniform(shape=(sample_dim_size,), ctx=ctx)
            accept = (u < all_intensities.sum(-1) / lda_proposal) * (
                time < max_time
            )

            # append samples
            time_samples.append(time * accept)
            mark_samples.append(m * accept)
            # noinspection PyTypeChecker
            masks.append(accept)

            # advance the lstm
            lstm_in = nd.concat(
                tau.reshape((sample_dim_size, 1)), self.embedding(m), dim=1
            )

            _, (tmp_lstm_out, tmp_lstm_mem) = self.lstm(
                lstm_in, (lstm_out, lstm_mem)
            )

            lstm_out = nd.where(accept, tmp_lstm_out, lstm_out)
            lstm_mem = nd.where(accept, tmp_lstm_mem, lstm_mem)

            # set tau to zero for accepted
            tau *= 1 - accept

        # FIXME: temp type checker workaround
        return time, tau
        # ia_times_out, marks_out, valid_length = _realign_masked_samples(
        #     nd.stack(*time_samples, axis=1),
        #     nd.stack(*mark_samples, axis=1),
        #     nd.stack(*masks, axis=1),
        # )
        #
        # return (
        #     nd.stack(
        #         ia_times_out.swapaxes(0, 1),
        #         marks_out.swapaxes(0, 1),
        #         axis=-1
        #     ),
        #     valid_length,
        # )
