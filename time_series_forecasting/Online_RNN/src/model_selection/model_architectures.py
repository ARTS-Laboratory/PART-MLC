import torch

from custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch


class TorchRNNExperiment(RecurrentNeuralNetworkTorch):
    def __init__(self, history_length, loss_fn, num_layers=1, data_type=torch.float32):
        """

        :param int history_length: How many features the model looks at at one time.
        :param loss_fn: Loss function to apply to model
        :type loss_fn:
        :param int num_layers: Number of layers model has. Defaults to 1.
        :param data_type: Type of data to feed into model
        :type data_type: torch.dtype
        """
        super(TorchRNNExperiment, self).__init__()
        self.dtype = data_type
        self.input_length = history_length
        self.hidden_features = 1
        self.num_layers = num_layers
        self.rec_1 = torch.nn.RNN(
            history_length, self.hidden_features, num_layers=self.num_layers,
            dtype=self.dtype, batch_first=True)
        self.relu_1 = torch.nn.ReLU()
        self.loss_fn = loss_fn
        self.hidden_state = self.random_hidden()

    def forward(self, input_val, hidden):
        """ Return prediction from model.

            :param input_val: Input values to feed into model.
            :param hidden: Hidden state of the model.
            :type input_val: torch.Tensor. Should have size
             (sequence length, input size) or (batch, sequence length, input size)
            :returns: Tuple of prediction output and state.
            :rtype: tuple[torch.Tensor, torch.Tensor]

            """
        # input_val.dtype(self.dtype)
        out, hidden = self.rec_1(input_val, hidden)
        return out, hidden

    def predict(self, input_val, hidden):
        """ Return call to forward.

            Wrapper for forward method.
        """
        return self.forward(input_val, hidden)

    def loss(self, prediction, actual):
        return self.loss_fn(prediction, actual)

    def random_hidden(self, batched=0, bidirect=False):
        if batched != 0:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                batched, self.hidden_features)
        else:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                self.hidden_features)


class TorchLSTMExperiment(RecurrentNeuralNetworkTorch):
    def __init__(self, history_length, loss_fn, num_layers=1, data_type=torch.float32):
        super(TorchLSTMExperiment, self).__init__()
        self.dtype = data_type
        self.input_length = history_length
        self.hidden_features = 1
        self.num_layers = 2
        self.rec_1 = torch.nn.LSTM(
            history_length, self.hidden_features, num_layers=self.num_layers,
            batch_first=True, dtype=self.dtype)
        self.relu_1 = torch.nn.ReLU()
        self.loss_fn = loss_fn

    def forward(self, input_val, hidden_state):
        """ Return prediction from model.

            :param input_val: Input values to feed into model. Expected size is
             (sequence length, input size) or (batch, sequence length, input size)
            :type input_val: torch.Tensor
            :param hidden_state: Hidden state of the model.
            :type hidden_state: tuple[torch.Tensor, torch.Tensor]

            """
        hidden, cell = hidden_state
        # input_val.dtype(self.dtype)
        out, (hidden, cell) = self.rec_1(input_val, (hidden, cell))
        return out, (hidden, cell)

    def predict(self, input_val, hidden_state):
        """ Return call to forward.

            Wrapper for forward method.
        """
        return self.forward(input_val, hidden_state)

    def loss(self, prediction, actual):
        return self.loss_fn(prediction, actual)

    def random_hidden_single(self, batched=0, bidirect=False):
        if batched != 0:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                batched, self.hidden_features)
        else:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                self.hidden_features)

    def random_hidden(self, batched=0, bidirect=False):
        """ Generate random values for initial hidden state and initial cell state.

            :returns: Tuple of random values. Does not currently account for projections.
        """
        return (
            self.random_hidden_single(batched=batched, bidirect=bidirect),
            self.random_hidden_single(batched=batched, bidirect=bidirect))