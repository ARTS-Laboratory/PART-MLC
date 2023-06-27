import unittest
from unittest.mock import mock_open, patch, Mock, MagicMock
from unittest import TestCase
from multiprocessing import Lock, Queue, Pipe, Process

import torch
import time

# todo test this import in rest of code
try:
    from main import TorchRNN
    from custom_models.forecaster_model import Forecaster, make_predictions, make_improvements_torch_pipe, \
        make_predictions_torch_pipe, run_forecaster
    from custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch
except ModuleNotFoundError:
    from time_series_forecasting.Online_RNN.src.main import TorchRNN
    from time_series_forecasting.Online_RNN.src.custom_models.forecaster_model import (
        Forecaster, make_predictions, make_improvements_torch_pipe,
        make_predictions_torch_pipe, run_forecaster)
    from time_series_forecasting.Online_RNN.src.custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch


class TestForecaster(TestCase):
    def setUp(self) -> None:
        self.forecaster = Forecaster()

    def test_predict(self):
        self.fail()

    def test_learn(self):
        self.fail()

    def test_update_weights(self):
        self.fail()


class TestPredictor(TestCase):
    def setUp(self) -> None:
        self.seq_length = 3
        self.type = torch.float32
        self.input_data = [
            torch.tensor([[0]], dtype=self.type),
            torch.tensor([[1]], dtype=self.type),
            torch.tensor([[2]], dtype=self.type),
            torch.tensor([[3]], dtype=self.type),
            torch.tensor([[4]], dtype=self.type),
            torch.tensor([[5]], dtype=self.type), None]
        self.hidden = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]
        self.expected = [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]]]
        self.mock_model = TorchRNN(1, torch.nn.MSELoss())
        self.mock_model_1 = TorchRNN(1, torch.nn.MSELoss())
        self.lr = 0.05  # Learning rate for optimizer
        self.mock_optimizer = torch.optim.SGD(self.mock_model.parameters(), self.lr)
        # self.mock_model = RecurrentNeuralNetworkTorch()
        # self.mock_model_1 = RecurrentNeuralNetworkTorch()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.parent, self.child = Pipe()
        # self.lock = Lock()

    def test_make_predictions_torch_pipe_time(self):
        # Input instantiation
        for data in self.input_data[0: len(self.input_data) - 2]*1_00:
            self.input_queue.put(data)
        for data in self.input_data:
            self.input_queue.put(data)
        # Function call
        # Patching close because close works differently on same process
        with patch.object(self.output_queue, 'close') as mock_output_close,\
                patch.object(self.child, 'close') as mock_pipe_close:
            start = time.perf_counter()
            make_predictions_torch_pipe(self.mock_model, self.input_queue, self.output_queue, self.child)
            end = time.perf_counter()
        # Results and cleanup
        print(f'Total Time: {end - start}, Average time: {(end - start) / ((5*1_00)+6)}')
        self.assertTrue(self.input_queue.empty())
        self.input_queue.close()
        mock_output_close.assert_called_once()
        mock_pipe_close.assert_called_once()
        results = queue_to_list(self.output_queue)

    # TODO test queue failures, timeouts, check for leaks
    def test_make_predictions_mock_sub(self):
        # Input instantiation
        mock_lock = MagicMock(Lock())
        for data in self.input_data:
            self.input_queue.put(data)
        # Function call
        with patch.object(self.output_queue, 'close') as mock_output_close:
            make_predictions(
                self.mock_model, self.input_queue, self.output_queue, mock_lock)
        self.assertTrue(self.input_queue.empty())
        self.input_queue.close()
        # Results and cleanup
        mock_output_close.assert_called_once()
        results = list()
        item = self.output_queue.get()
        while item is not None:
            results.append(item)
            del item
            item = self.output_queue.get()
        # del self.output_queue
        # self.assertEqual(self.expected, results)

    # TODO test queue failures, timeouts, check for leaks
    def test_make_predictions_pipe_torch_mock_sub_succeeds(self):
        # Input instantiation
        for data in self.input_data:
            self.input_queue.put(data)
        # Function call
        # Patching close because close works differently on same process
        with patch.object(self.output_queue, 'close') as mock_output_close,\
                patch.object(self.child, 'close') as mock_pipe_close:
            make_predictions_torch_pipe(self.mock_model, self.input_queue, self.output_queue, self.child)
        # Results and cleanup
        self.assertTrue(self.input_queue.empty())
        self.input_queue.close()
        mock_output_close.assert_called_once()
        mock_pipe_close.assert_called_once()
        results = queue_to_list(self.output_queue)

    def test_make_prediction_pipe_torch_mock_sub_model_fails_all_input_sent(self):
        # Input instantiation
        self.mock_model.predict = MagicMock(side_effect=RuntimeError('Something went wrong with prediction'))
        for data in self.input_data:
            self.input_queue.put(data)
        # Function call
        with self.assertRaises(RuntimeError):
            with patch.object(self.output_queue, 'close') as mock_output_close,\
                    patch.object(self.child, 'close') as mock_pipe_close:
                make_predictions_torch_pipe(self.mock_model, self.input_queue, self.output_queue, self.child)
        # Results and cleanup
        self.input_queue.close()
        mock_output_close.assert_called_once()
        mock_pipe_close.assert_called_once()

    def test_make_prediction_pipe_torch_mock_sub_model_fails_some_input_sent(self):
        # Input instantiation
        self.mock_model.predict = MagicMock(side_effect=RuntimeError('Something went wrong with prediction'))
        input_queue, output_queue = Queue(), Queue()
        parent, child = Pipe()
        half = len(self.input_data)//2
        for data in self.input_data[half:]:
            input_queue.put(data)
        # Function call
        with self.assertRaises(RuntimeError):
            with patch.object(output_queue, 'close') as mock_output_close, patch.object(child, 'close') as mock_pipe_close:
                make_predictions_torch_pipe(self.mock_model, input_queue, output_queue, child)
            for data in self.input_data[:half]:
                input_queue.put(data)
            # Results and cleanup
            input_queue.close()
            mock_output_close.assert_called_once()
            mock_pipe_close.assert_called_once()

    def test_make_prediction_torch_pipe_other_pipe_closed_all_input_sent(self):
        # Input instantiation
        self.parent.close()
        for data in self.input_data:
            self.input_queue.put(data)
        # Function call, no error should occur
        with patch.object(self.output_queue, 'close') as mock_output_close,\
                patch.object(self.child, 'close') as mock_pipe_close:
            make_predictions_torch_pipe(
                self.mock_model, self.input_queue, self.output_queue, self.child)
        # Results and cleanup
        self.input_queue.close()
        mock_output_close.assert_called_once()
        mock_pipe_close.assert_called_once()

    # def test_make_predictions_torch_pipe(self):
    #     # Input instantiation
    #     for data in self.input_data:
    #         self.input_queue.put(data)
    #     # Function call
    #     # Patching close because close works differently on same process
    #     with patch.object(self.output_queue, 'close') as mock_output_close,\
    #             patch.object(self.child, 'close') as mock_pipe_close:
    #         make_predictions_torch_pipe(self.mock_model, self.input_queue, self.output_queue, self.child)
    #     # Results and cleanup
    #     self.assertTrue(self.input_queue.empty())
    #     self.input_queue.close()
    #     mock_output_close.assert_called_once()
    #     mock_pipe_close.assert_called_once()
    #     results = list()
    #     item = self.output_queue.get()
    #     while item is not None:
    #         results.append(item)
    #         del item
    #         item = self.output_queue.get()
    #     del item


class TestLearner(unittest.TestCase):
    def setUp(self) -> None:
        self.seq_length = 3
        self.type = torch.float32
        self.input_data = [
            torch.tensor([[0]], dtype=self.type),
            torch.tensor([[1]], dtype=self.type),
            torch.tensor([[2]], dtype=self.type),
            torch.tensor([[3]], dtype=self.type),
            torch.tensor([[4]], dtype=self.type),
            torch.tensor([[5]], dtype=self.type), None]
        self.hidden = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]
        self.expected = [
            torch.tensor([[1]], dtype=self.type), torch.tensor([[2]], dtype=self.type),
            torch.tensor([[3]], dtype=self.type), torch.tensor([[4]], dtype=self.type),
            torch.tensor([[5]], dtype=self.type), torch.tensor([[6]], dtype=self.type)]
        self.mock_model = TorchRNN(1, torch.nn.MSELoss())
        self.mock_model_1 = TorchRNN(1, torch.nn.MSELoss())
        self.lr = 0.05  # Learning rate for optimizer
        self.mock_optimizer = torch.optim.SGD(self.mock_model.parameters(), self.lr)
        self.mock_loss_fn = torch.nn.MSELoss()
        # self.mock_model = RecurrentNeuralNetworkTorch()
        # self.mock_model_1 = RecurrentNeuralNetworkTorch()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.parent, self.child = Pipe()
        self.lock = Lock()

    def test_make_improvements_torch_pipe_succeeds(self):
        expected_call_count = len(self.expected)
        for data, expect in zip(self.input_data, self.expected + [None]):
            self.input_queue.put((data, expect))
        with patch.object(self.child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, self.input_queue, self.mock_loss_fn, self.seq_length, self.mock_optimizer,
                self.child)
        mock_child_close.assert_called_once()
        # self.child.recv.assert_not_called()
        # self.parent.send.assert_not_called()
        # self.assertEqual(expected_call_count, mock_loss_fn.backward.call_count)
        # self.assertEqual(expected_call_count, mock_optimizer.step.call_count)

    def test_make_improvements_torch_pipe_optimizer_fails(self):
        expected_call_count = len(self.expected)
        self.mock_optimizer.step = MagicMock(side_effect=RuntimeError('Optimizer step failed.'))
        for data, expect in zip(self.input_data, self.expected + [None]):
            self.input_queue.put((data, expect))
        with self.assertRaises(RuntimeError), patch.object(self.child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, self.input_queue, self.mock_loss_fn, self.seq_length, self.mock_optimizer,
                self.child)
            mock_child_close.assert_called_once()

    def test_make_improvements_torch_pipe_parent_pipe_closed(self):
        expected_call_count = len(self.expected)
        self.parent.close()
        for data, expect in zip(self.input_data, self.expected + [None]):
            self.input_queue.put((data, expect))
        with self.assertRaises(ChildProcessError),\
                patch.object(self.child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, self.input_queue, self.mock_loss_fn, self.seq_length, self.mock_optimizer,
                self.child)
            mock_child_close.assert_called_once()

    def test_make_improvements_torch_pipe_pipe_full(self):
        for data, expect in zip(self.input_data, self.expected + [None]):
            self.input_queue.put((data, expect))
        with patch.object(self.child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, self.input_queue, self.mock_loss_fn, self.seq_length, self.mock_optimizer,
                self.child)


    def test_update_weights(self):
        self.fail()

    def test_update_weights_multi_p(self):
        self.fail()

    def test_run_forecaster(self):
        run_forecaster()

    def test_run_forecaster_predictor_fails(self):
        self.fail()

    def test_run_forecaster_learner_fails(self):
        self.fail()

    def tearDown(self) -> None:
        self.input_queue.close()
        self.output_queue.close()
        self.parent.close()
        self.child.close()
        del self.input_queue, self.output_queue
        del self.parent, self.child
        self.input_data.clear()
        self.expected.clear()
        del self.mock_model
        del self.mock_model_1
        del self.input_data, self.expected


def queue_to_list(queue):
    results = list()
    item = queue.get()
    while item is not None:
        results.append(item)
        del item
        item = queue.get()
    del item
    return results


if __name__ == '__main__':
    unittest.main()
