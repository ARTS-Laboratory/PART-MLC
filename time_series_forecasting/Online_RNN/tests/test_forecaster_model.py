import unittest
from unittest.mock import mock_open, patch, Mock, MagicMock
from unittest import TestCase
from multiprocessing import Lock, Queue, Pipe, Process

import torch

from forecaster.custom_models.forecaster_model import Forecaster, make_predictions, make_improvements_torch_pipe, \
    make_predictions_torch_pipe, run_forecaster
from forecaster.custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch


class TestForecaster(TestCase):
    def setUp(self) -> None:
        self.forecaster = Forecaster()

    def test_predict(self):
        self.fail()

    def test_learn(self):
        self.fail()

    def test_update_weights(self):
        self.fail()


class Test(TestCase):
    def setUp(self) -> None:
        self.input_data = [[0], [1], [2], [3], [4], [5], None]
        self.hidden = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]
        self.expected = [[1], [2], [3], [4], [5], [6]]
        self.mock_model = RecurrentNeuralNetworkTorch()
        self.mock_model_1 = RecurrentNeuralNetworkTorch()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.parent, self.child = Pipe()
        self.lock = Lock()

    # TODO test queue failures, timeouts, check for leaks
    def test_make_predictions_mock_sub(self):
        # Input instantiation
        # mock_model = RecurrentNeuralNetworkTorch()
        self.mock_model.predict = MagicMock(side_effect=iter(zip(self.hidden, self.expected)))
        input_queue = Queue()
        mock_output_queue = Queue()
        mock_lock = MagicMock(Lock())
        for data in self.input_data:
            input_queue.put(data)
        # Function call
        with patch.object(mock_output_queue, 'close') as mock_output_close:
            make_predictions(self.mock_model, input_queue, mock_output_queue, mock_lock)
        input_queue.close()
        # Results and cleanup
        mock_output_close.assert_called_once()
        results = list()
        item = mock_output_queue.get()
        while item is not None:
            results.append(item)
            del item
            item = mock_output_queue.get()
        del mock_output_queue
        self.assertEqual(self.expected, results)

    # TODO test queue failures, timeouts, check for leaks
    def test_make_predictions_pipe_torch_mock_sub_succeeds(self):
        # Input instantiation
        self.mock_model.predict = MagicMock(side_effect=iter(zip(self.hidden, self.expected)))
        input_queue, output_queue = Queue(), Queue()
        parent, child = Pipe()
        for data in self.input_data:
            input_queue.put(data)
        # Function call
        with patch.object(output_queue, 'close') as mock_output_close, patch.object(child, 'close') as mock_pipe_close:
            make_predictions_torch_pipe(self.mock_model, input_queue, output_queue, child)
        # Results and cleanup
        self.assertTrue(input_queue.empty())
        input_queue.close()
        mock_output_close.assert_called_once()
        mock_pipe_close.assert_called_once()
        results = list()
        item = output_queue.get()
        while item is not None:
            results.append(item)
            del item
            item = output_queue.get()
        del output_queue, item
        self.assertEqual(self.expected, results)

    def test_make_prediction_pipe_torch_mock_sub_model_fails_all_input_sent(self):
        # Input instantiation
        self.mock_model.predict = MagicMock(side_effect=RuntimeError('Something went wrong with prediction'))
        input_queue, output_queue = Queue(), Queue()
        parent, child = Pipe()
        for data in self.input_data:
            input_queue.put(data)
        # Function call
        with self.assertRaises(RuntimeError):
            with patch.object(output_queue, 'close') as mock_output_close, patch.object(child, 'close') as mock_pipe_close:
                make_predictions_torch_pipe(self.mock_model, input_queue, output_queue, child)
        # Results and cleanup
            input_queue.close()
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


    def test_make_predictions_torch_pipe(self):
        # Input instantiation
        # mock_model = RecurrentNeuralNetworkTorch()
        self.mock_model.predict = MagicMock(side_effect=iter(zip(self.hidden, self.expected)))
        mock_input_queue = Queue()
        mock_output_queue = Queue()
        for data in self.input_data:
            mock_input_queue.put(data)
        parent, child = Pipe()
        # Function call
        p = Process(
            target=make_predictions_torch_pipe, args=(self.mock_model, mock_input_queue, mock_output_queue, child))
        p.start()
        mock_input_queue.close()
        p.join()
        # Results and cleanup
        results = list()
        item = mock_output_queue.get()
        while item is not None:
            results.append(item)
            del item
            item = mock_output_queue.get()
        del mock_output_queue
        self.assertEqual(self.expected, results)

    def test_make_improvements_torch_pipe_succeeds(self):
        # mock_model = RecurrentNeuralNetworkTorch()
        expected_call_count = len(self.expected)
        self.mock_model.predict = MagicMock(side_effect=iter(self.expected))
        input_queue = Queue()
        mock_loss_fn = MagicMock(torch.nn.MSELoss)
        # mock_loss_fn.backward = MagicMock()
        mock_optimizer = MagicMock(torch.optim.Adam)
        mock_optimizer.step = MagicMock()
        parent, child = Pipe()
        # Expected to be used
        child.send = MagicMock()
        parent.recv = MagicMock()
        # Not expected to be called
        child.recv = MagicMock()
        parent.send = MagicMock()
        for data, expect in zip(self.input_data, self.expected + [None]):
            input_queue.put((data, expect))
        with patch.object(child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, input_queue, mock_loss_fn, mock_optimizer,
                child)
        mock_child_close.assert_called_once()
        child.recv.assert_not_called()
        parent.send.assert_not_called()
        # self.assertEqual(expected_call_count, mock_loss_fn.backward.call_count)
        self.assertEqual(expected_call_count, mock_optimizer.step.call_count)

    def test_make_improvements_torch_pipe_optimizer_fails(self):
        # mock_model = RecurrentNeuralNetworkTorch()
        expected_call_count = len(self.expected)
        self.mock_model.predict = MagicMock(side_effect=iter(self.expected))
        input_queue = Queue()
        mock_loss_fn = MagicMock(torch.nn.MSELoss)
        # mock_loss_fn.backward = MagicMock()
        mock_optimizer = MagicMock(torch.optim.Adam)
        mock_optimizer.step = MagicMock(side_effect=RuntimeError)
        parent, child = Pipe()
        # Expected to be used
        child.send = MagicMock()
        parent.recv = MagicMock()
        # Not expected to be called
        child.recv = MagicMock()
        parent.send = MagicMock()
        for data, expect in zip(self.input_data, self.expected + [None]):
            input_queue.put((data, expect))
        with self.assertRaises(RuntimeError), patch.object(child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, input_queue, mock_loss_fn, mock_optimizer,
                child)
            mock_child_close.assert_called_once()

    def test_make_improvements_torch_pipe_optimizer_fails(self):
        # mock_model = RecurrentNeuralNetworkTorch()
        expected_call_count = len(self.expected)
        self.mock_model.predict = MagicMock(side_effect=iter(self.expected))
        input_queue = Queue()
        mock_loss_fn = MagicMock(torch.nn.MSELoss, side_effect=RuntimeError)
        # mock_loss_fn.backward = MagicMock()
        mock_optimizer = MagicMock(torch.optim.Adam)
        mock_optimizer.step = MagicMock()
        parent, child = Pipe()
        # Expected to be used
        child.send = MagicMock()
        parent.recv = MagicMock()
        # Not expected to be called
        child.recv = MagicMock()
        parent.send = MagicMock()
        for data, expect in zip(self.input_data, self.expected + [None]):
            input_queue.put((data, expect))
        with self.assertRaises(RuntimeError), patch.object(child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, input_queue, mock_loss_fn, mock_optimizer,
                child)
            mock_child_close.assert_called_once()

    def test_make_improvements_torch_pipe_pipe_closes(self):
        # mock_model = RecurrentNeuralNetworkTorch()
        expected_call_count = len(self.expected)
        self.mock_model.predict = MagicMock(side_effect=iter(self.expected))
        input_queue = Queue()
        mock_loss_fn = MagicMock(torch.nn.MSELoss)
        # mock_loss_fn.backward = MagicMock()
        mock_optimizer = MagicMock(torch.optim.Adam)
        mock_optimizer.step = MagicMock()
        parent, child = Pipe()
        # Expected to be used
        child.send = MagicMock()
        parent.recv = MagicMock()
        # Not expected to be called
        child.recv = MagicMock()
        parent.send = MagicMock()
        # Close receiving pipe
        parent.close()
        for data, expect in zip(self.input_data, self.expected + [None]):
            input_queue.put((data, expect))
        with self.assertRaises(BrokenPipeError), patch.object(child, 'close') as mock_child_close:
            make_improvements_torch_pipe(
                self.mock_model, input_queue, mock_loss_fn, mock_optimizer,
                child)
        mock_child_close.assert_called_once()
        parent.send.assert_called_once()
        child.recv.assert_not_called()
        parent.send.assert_not_called()
        # self.assertEqual(expected_call_count, mock_loss_fn.backward.call_count)
        # self.assertEqual(expected_call_count, mock_optimizer.step.call_count)

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


if __name__ == '__main__':
    unittest.main()
