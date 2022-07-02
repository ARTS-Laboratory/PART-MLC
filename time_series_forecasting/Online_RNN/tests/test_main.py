from unittest import TestCase

import numpy as np
import torch

from forecaster.main import TorchRNN, copy_weights


class Test(TestCase):
    def test_load_data_numpy(self):
        self.fail()

    def test_load_data_pandas(self):
        self.fail()

    def test_parse_data(self):
        self.fail()

    def test_copy_weights(self):
        first = TorchRNN(20, loss_fn=None)
        second = TorchRNN(20, loss_fn=None)
        print(str(first.state_dict()) == str(second.state_dict()))
        # self.assertFalse(first.parameters() == second.parameters())
        copy_weights(first, second)
        for param_1, param_2 in zip(first.parameters(), second.parameters()):
            # print(param_1.data)
            # print(param_2.data)
            self.assertTrue(torch.equal(param_1.data, param_2.data))
        # self.assertTrue(first.parameters() == second.parameters())


    def test_main(self):
        self.fail()


class TestTorchRNN(TestCase):
    def setUp(self) -> None:
        self.history_length = 4
        self.dtype = torch.float32
        self.dtypes = [torch.float, torch.float16, torch.float32, torch.float64]
        self.loss_fn = torch.nn.MSELoss
        self.predictor = TorchRNN(self.history_length, loss_fn=None)
        self.learner = TorchRNN(self.history_length, loss_fn=self.loss_fn)
        self.input_shape = (1, self.history_length)
        self.single_dummy_input_torch = torch.ones(self.input_shape)
        self.single_dummy_input_numpy = np.ones(self.input_shape)
        # for name, value in self.predictor.named_parameters():
        #     print(f'Name: {name}, Value: {value}')
        # for name, value in self.learner.named_parameters():
        #     print(f'Name: {name}, Value: {value}')

        self.params = self.learner.parameters()
        self.state = self.learner.state_dict()
        # print(self.params is self.learner.parameters())
        # print(self.params == self.learner.parameters())
        # for param in self.params:
        #     print(param)
        # for k, v in self.state.items():
        #     print(f'Key: {k} , Value: {v}')

    def test_forward_torch_single_data(self):
        a, b = self.predictor.forward(self.single_dummy_input_torch)
        print(a.data)
        print(b.data)

    def test_forward_numpy_single_data(self):
        a = self.predictor.forward(self.single_dummy_input_numpy)

    def test_predict(self):
        self.fail()

    def test_loss(self):
        self.fail()

    # def test_standalone_optimizer(self):
    #     optim = torch.optim.Adam(self.params)
    #     print(optim.state_dict())
    #     # self.assertTrue(True)
