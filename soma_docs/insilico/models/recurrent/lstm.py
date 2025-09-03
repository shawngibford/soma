import typing as t

import numpy as np
import torch
from orquestra.qml.api import Batch, TorchGenerativeModel, TrainResult
from orquestra.qml.losses.th import weighted_nll_loss
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn


class _LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        lstm: nn.LSTM,
        output_activation: nn.Module = nn.LogSoftmax(-1),
        padding_token_index: t.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = lstm.input_size
        self.latent_dim = lstm.hidden_size
        self.n_layers = lstm.num_layers
        self.dropout = lstm.dropout
        self._D = (
            2 if lstm.bidirectional else 1
        )  # this is needed for computing output dims
        self.padding_token_index = padding_token_index

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_token_index,
        )

        self.recurrent_net = lstm
        self.latent_linear = nn.Linear(self._D * self.latent_dim, self.vocab_size)
        self.output_activation = output_activation

    def forward(
        self,
        inputs: torch.Tensor,
        initial_hidden_state: t.Optional[t.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> t.Tuple[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        # inputs -> (batch_size, sequence_length)
        # embedded_inputs -> (batch_size, sequence_length, embedding_dim)
        embedded_inputs = self.embedding_layer(inputs)

        # raw_sequences -> (batch_size, sequence_length, hidden_dim)
        raw_sequences, hidden_state = self.recurrent_net(
            embedded_inputs, initial_hidden_state
        )

        # raw_logits -> (batch_size, sequence_length, vocab_size)
        raw_logits = self.latent_linear(raw_sequences)

        return self.output_activation(raw_logits), hidden_state

    @property
    def hidden_size(self) -> int:
        return self.latent_dim


class LSTM(TorchGenerativeModel):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        sos_token_index: int,
        padding_token_index: t.Optional[int] = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "loss",
        model_identifier: str = "lstm_mle",
    ) -> None:
        super().__init__()

        recurrent_net = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=False,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.sos_token_index = sos_token_index
        self.padding_token_index = padding_token_index
        self._input_size = (seq_len,)

        self._model: _LSTM = _LSTM(
            vocab_size, recurrent_net, padding_token_index=padding_token_index
        )

        self.optimizer = optimizer_config.optimizer(self._model.parameters())
        self.loss_fn = weighted_nll_loss
        self.loss_key = loss_key
        self.model_identifier = model_identifier

    def __call__(
        self,
        inputs: torch.Tensor,
        initial_hidden_state: t.Optional[t.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> t.Tuple[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        # outputs[0] -> (batch_size, sequence_length, vocab_size)
        return self._model(inputs, initial_hidden_state)

    def _make_xo(self, n_samples: int) -> torch.Tensor:
        return torch.full((n_samples, 1), self.sos_token_index)

    def get_token_embeddings(self, token_indices: t.Sequence[int]) -> torch.Tensor:
        input_tensor = torch.tensor(token_indices).view((-1, 1))
        with torch.no_grad():
            embeddings = self._model.embedding_layer(input_tensor)
        return embeddings

    def _train_on_batch(self, batch: Batch) -> TrainResult:
        data = batch.data
        probs = batch.probs

        self.set_train_state()
        self.optimizer.zero_grad()

        if len(data.size()) != 2:
            raise ValueError(
                f"Expected 2D tensor as input, but got {len(data.size())}D tensor."
            )

        batch_size, seq_len = data.shape

        # first element will be the <START> token
        # append to that everything but last element of sequence
        x0 = self._make_xo(batch_size)
        model_inputs = torch.concat((x0, data[:, : seq_len - 1]), dim=1).long()

        # output -> (batch_size, sequence_length, vocab_size)
        generated_sequence, *_ = self(model_inputs, None)

        # permute to fit shape needed by NLL loss function
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        generated_sequence_t = generated_sequence.permute(0, 2, 1)
        loss = self.loss_fn(generated_sequence_t, data.long(), probs)  # (B, L)

        loss.backward()
        self.optimizer.step()

        return {self.loss_key: loss.detach()}

    def _generate_w_probs(
        self, n_samples: int, random_seed: t.Optional[int] = None
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.
        Args:
            n_samples (int): then number of samples to generate.
            random_seed (Optional[int], optional): an optional random seed for reproducibility. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the raw generated sequences and the associated probabilities.
        """
        random_generator = None
        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        inputs = self._make_xo(n_samples)  # (batch_size, 1)
        hidden_state: t.Optional[t.Tuple[torch.Tensor, torch.Tensor]] = None
        outputs = torch.zeros((n_samples, self.seq_len))
        seq_probabilities = torch.ones((n_samples, 1))  # TODO: update probs

        with torch.no_grad():
            for index in range(0, self.seq_len):
                # log_prob_dist -> (batch_size, 1, vocab_size)
                log_prob_dist, hidden_state = self(inputs, hidden_state)

                # output is passed through LogSoftmax so take exponent
                # prob_dist -> (batch_size, vocab_size)
                prob_dist = torch.exp(log_prob_dist).squeeze(1)

                # sampled_token_indices -> (batch_size, 1)
                sampled_token_indices = torch.multinomial(prob_dist, 1)
                outputs[:, index] = sampled_token_indices.squeeze(1)

                # inputs -> (batch_size, 1)
                inputs = sampled_token_indices

        return outputs, seq_probabilities

    @property
    def _models(self) -> t.List[nn.Module]:
        return [self._model]

    @property
    def sample_size(self) -> t.Tuple[int, ...]:
        return tuple(self.generate(1).shape[1:])

    @property
    def input_size(self) -> t.Tuple[int, ...]:
        return self._input_size

    @property
    def config(self) -> t.Dict:
        """Returns model configuration."""
        d = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self._model.embedding_dim,
            "latent_dim": self._model.latent_dim,
            "n_layers": self._model.n_layers,
            "dropout": self._model.dropout,
            "bidirectional": self._model._D,
            "padding_token_index": self.padding_token_index,
        }

        return d

    def generate(
        self, n_samples: int, random_seed: t.Optional[int] = None
    ) -> torch.Tensor:
        generated_sequences, probs = self._generate_w_probs(n_samples, random_seed)
        return generated_sequences
