""" """

import time
import torch
from multiprocessing import Process, Lock, Pipe, Queue
import numpy as np

from constants import PIPE_SENTINEL, PIPE_CONFIRMATION, QUEUE_SENTINEL

# Types
from multiprocessing.queues import Queue as QueueType
from multiprocessing.connection import PipeConnection as PipeType
from multiprocessing.synchronize import Lock as LockType
from typing import Optional
# Exceptions
from queue import Empty, Full

from custom_losses import MSETRACLoss
# Class implementation
from custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch


class Forecaster:
    # TODO add functionality to include non scratch implementations
    def __init__(self, predictor, learner, loss_fn, gradient_fn):
        # Assume models are already built
        self.predictor = predictor
        self.learner = learner
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn

    def predict(self, data):
        """ """
        return self.predictor.predict(data)

    def learn(self, data, expected, weights, learning_rate):
        """ """
        # Guess on past data
        past_prediction = self.learner.predict(data)
        # Compute loss
        loss = self.loss_fn(data, expected)
        # New gradients
        gradients = self.gradient_fn(weights, loss, learning_rate)
        # Idk bro

    def update_predictor_weights(self, new_weights, lock=None):
        """ """
        if lock is not None:
            with lock:
                update_weights(self.predictor, new_weights)
        else:
            update_weights(self.predictor, new_weights)


# TODO fix data processing, time_skip is not clear enough
# Function implementation
def run_forecaster(pair: Forecaster, data, input_size, time_skip: int, use_torch=True, **kwargs):
    """
    # TODO pair will be changed to be two models in this function. predictor and learner
    :param pair:
    :param data: Data to send to predictor and learner
    :type data: np.ndarray or torch.Tensor
    :type data: torch.Tensor or np.ndarray
    :param int input_size: Number of elements model looks at at any one time
    :param int time_skip: Number of elements to skip between current and past prediction
    :param bool use_torch: Whether the model to use is a torch model
    """
    # Regular variables
    # todo Get rid of magic numbers
    seq_length = 16
    start = 0
    end = time_skip + (input_size << 1)  # inclusive end index
    if use_torch and not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data).detach()  # Might need to add if clause if not ndarray
    # TODO make typing run better here
    if isinstance(data, np.ndarray):
        end_index = np.size(data, 1) - end
    elif isinstance(data, torch.Tensor):
        end_index = data.size(1) - end
    else:
        raise NotImplementedError
    # print(end_index)
    # TODO function call needs parameters predictor and learner instead of Forecaster
    # Other Process Variables
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(pair.learner.parameters())
    # Shared Process Variables
    # curr_q: QueueType = Queue()
    # curr_out: QueueType = Queue()
    # hist_q: QueueType = Queue()
    # hist_out: QueueType = Queue()
    # update_lock: LockType = Lock()
    predict_parent, predict_child = Pipe()
    learn_parent, learn_child = Pipe()
    parent_param, child_param = Pipe()
    ## All pipes version
    # Start processes
    predict: Process = Process(
        target=make_predictions_torch_all_pipes,
        args=(pair.predictor, predict_child, parent_param)
    )
    learn: Process = Process(
        target=make_improvements_torch_all_pipes,
        args=(pair.learner, learn_child, loss_fn, seq_length, optimizer, child_param, True)
    )
    try:
        predict.start()
        learn.start()
        results, results_2 = push_data_to_pipes(
            data, start, end_index, time_skip, input_size, predict_parent, learn_parent)
        predict.join()
        learn.join()
    finally:
        if predict.is_alive():
            predict.terminate()
        if learn.is_alive():
            learn.terminate()
        time.sleep(0.5)
        predict.close()
        learn.close()
    return results, results_2
    ## Old version
    # p: Process = Process(
    #     target=make_predictions_torch_pipe,
    #     args=(pair.predictor, curr_q, curr_out, parent_param))
    # q: Process = Process(
    #     target=make_improvements_torch_pipe,
    #     args=(pair.learner, hist_q, loss_fn, seq_length, optimizer, child_param, hist_out))
    # try:
    #     p.start()
    #     q.start()
    #     # Push items to queues, then close
    #     print('Starting loop')
    #     print(f'Iterating over array of length {len(data)} from 0 to {end_index}')
    #     push_data_to_queues(
    #         data, start, end_index, time_skip, input_size, curr_q, hist_q)
    #     # # Close queues
    #     # curr_q.close()
    #     # hist_q.close()
    #     print('Threads closed, joining threads')
    #     curr_q.join_thread()
    #     hist_q.join_thread()
    #     print('Threads joined, compile results')
    #     results = queue_to_list(curr_out)
    #     results_2 = queue_to_list(hist_out)
    #     print(f'Number of results {len(results)}')
    #     # Close processes
    #     p.join()
    #     print(f'Process predictor exit code: {p.exitcode}')
    #     q.join()
    #     print(f'Process learner exit code: {q.exitcode}')
    # finally:
    #     # Close processes
    #     p.join()
    #     q.join()
    #     print(f'Process predictor exit code: {p.exitcode} (final)')
    #     print(f'Process learner exit code: {q.exitcode} (final)')
    #     curr_q.close()
    #     curr_out.close()
    #     hist_out.close()
    #     hist_q.close()
    #     parent_param.close()
    #     child_param.close()
    #     del curr_q, curr_out, hist_q, parent_param, child_param
    #     del update_lock
    # return results, results_2  # list of tensors, fix


def push_data_to_pipes(
        data, start, end_index, time_skip, input_size,
        curr_pipe: PipeType, hist_pipe: PipeType):
    """ Send data from parent process to child processes through pipes."""
    curr_buff_empty, hist_buff_empty = True, True
    curr_buff = list()
    hist_x_buff, hist_y_buff = list(), list()
    curr_out, hist_out = list(), list()
    curr_ready, hist_ready = True, True
    if isinstance(data, torch.Tensor):
        with curr_pipe, hist_pipe:
            for idx in range(start, end_index):
                # end = idx + input_size + (time_skip << 1)  # inclusive end index
                curr_buff.append(data[:, (idx + time_skip): idx + input_size + time_skip].detach())
                hist_x_buff.append(data[:, idx: idx + input_size].detach())
                hist_y_buff.append(data[:, idx + input_size + time_skip].unsqueeze(1).detach())
                if curr_ready:
                    curr_pipe.send(torch.cat(curr_buff))
                    curr_buff.clear()
                    curr_ready = False
                if hist_ready:
                    hist_pipe.send([torch.cat(hist_x_buff), torch.cat(hist_y_buff)])
                    hist_x_buff.clear()
                    hist_y_buff.clear()
                    hist_ready = False
                if curr_pipe.poll():
                    curr_out.append(curr_pipe.recv())
                    curr_ready = True
                if hist_pipe.poll():
                    hist_out.append(hist_pipe.recv())
                    hist_ready = True
                time.sleep(0.05)
            if len(curr_buff) > 0:  # If buffer is not empty yet
                if not curr_ready:  # If predictor not ready for new data, wait
                    curr_out.append(curr_pipe.recv())
                curr_pipe.send(torch.cat(curr_buff))
                curr_buff_empty = False
            if len(hist_x_buff) > 0:  # If buffer is not empty yet
                if not hist_ready:  # If predictor not ready for new data, wait
                    hist_out.append(hist_pipe.recv())
                hist_pipe.send([torch.cat(hist_x_buff), torch.cat(hist_y_buff)])
                hist_buff_empty = False
            if not curr_buff_empty:
                curr_out.append(curr_pipe.recv())
            if not hist_buff_empty:
                hist_out.append(hist_pipe.recv())
            curr_pipe.send(PIPE_SENTINEL)
            hist_pipe.send([PIPE_SENTINEL, PIPE_SENTINEL])
        return torch.cat(curr_out), torch.cat(hist_out)
    elif isinstance(data, np.ndarray):
        raise NotImplementedError('Numpy arrays not implemented yet.')
    else:
        raise NotImplementedError(f'Array of type {type(data)} not supported.')


def push_data_to_queues(
        data, start, end_index, time_skip, input_size, curr_q, hist_q):
    """ Take data and push to child process queues."""
    try:
        if isinstance(data, torch.Tensor):
            for idx in range(start, end_index):
                end = idx + time_skip + (input_size << 1)  # inclusive end index
                curr_q.put(data[:, (idx + input_size + time_skip): end].detach())
                hist_q.put([data[:, idx: idx + input_size].detach(), data[:, end].unsqueeze(1).detach()])
                time.sleep(0.05)  # TODO this is a workaround to handle when learner finishes first, please fix
                # print(f'Current Queue size: {curr_q.qsize()}, Historical Queue size: {hist_q.qsize()}')
                # Add sleep here for time delay?
        elif isinstance(data, np.ndarray):
            for idx in range(start, end_index):
                end = idx + time_skip + (input_size << 1)  # inclusive end index
                curr_q.put(data[:, (idx + input_size + time_skip): end])
                hist_q.put([data[:, idx: idx + input_size], data[:, end]])
                # time.sleep(5)
                # Add sleep here for time delay?
        else:
            raise NotImplementedError(f'Array of type {type(data)} not supported.')
            print('End of loop')
        curr_q.put(QUEUE_SENTINEL)
        hist_q.put((QUEUE_SENTINEL, QUEUE_SENTINEL))
        print('Inserted end stuff')
    finally:
        # Close queues
        curr_q.close()
        hist_q.close()


def make_predictions(model, queue_or_data: QueueType, output_queue: QueueType = None, lock: LockType = None):
    """ Send predictions on input data asynchronously.

        :param model: Model used to make predictions on sequences of data
        # TODO type for model must be changed later to account for different classes, but this will work for now
        :type model: RecurrentNeuralNetworkTorch
        :param QueueType queue_or_data: Input queue for data
        :param QueueType output_queue: Queue for sending prediction data out
        :param LockType lock: Semaphore lock for model
    """
    # TODO maybe add check for if queue, assume queue for now
    try:
        data = queue_or_data.get(timeout=20)  # Change this later
        hidden = model.make_hidden_state()  # Add batch size if batched
        while data is not None:  # TODO this check will only break when get times out
            if isinstance(model, RecurrentNeuralNetworkTorch):
                data = torch.as_tensor(data)
            if lock is not None:
                with lock:
                    prediction, hidden = model.predict(data, hidden)
            else:
                prediction, hidden = model.predict(data, hidden)
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach()
            output_queue.put(prediction)
            del data
            data = queue_or_data.get(timeout=20)  # Change this later
        del data
        output_queue.put(QUEUE_SENTINEL)
    except Empty:
        raise TimeoutError('Queue timed out waiting for data.')
    finally:
        output_queue.close()
    # No way to send data back without this, assume no return
    # if output_queue is None:
    #     return data


def queue_to_list(queue):
    results = list()
    item = queue.get()
    while item is not None:
        results.append(item)
        del item
        item = queue.get()
    del item
    return results


def make_predictions_torch_pipe(
        model, input_queue: QueueType, output_queue: QueueType,
        pipe: PipeType):
    """Make predictions on input data, update weights with data from pipe, and send predictions through queue.

        Model makes predictions on input data from input_queue. It updates
        whenever there is new data in the pipe to use. It sends all output
        data through

        :param model: Model used to make predictions on sequences of data
        # TODO type for model must be changed later to account for different classes, but this will work for now
        :type model: RecurrentNeuralNetworkTorch
        :param QueueType input_queue: Input queue for data
        :param QueueType output_queue: Queue for sending prediction data out
        :param PipeType pipe: Pipe connection to receive model updates through
        """
    count = 0
    # TODO make hidden only if RNN type
    try:
        data = input_queue.get(timeout=20)  # Change this later
        hidden = model.make_hidden_state()  # Add batch size if batched
        while data is not QUEUE_SENTINEL:
            count += 1
            # print(f'Predictor looking at data slice #{count}', flush=True)
            with torch.no_grad():
                if isinstance(model, RecurrentNeuralNetworkTorch):
                    data = torch.as_tensor(data)
                prediction, hidden = model.predict(data, hidden)
            output_queue.put(prediction)
            del data
            if pipe.poll():  # Assumes this pipe will never get None, may change
                new_state = pipe.recv()
                torch_update_state(model, new_state)
                del new_state
                pipe.send(PIPE_CONFIRMATION)
            data = input_queue.get(timeout=5)  # Change this later
        del data
        output_queue.put(QUEUE_SENTINEL)
    except Empty:
        raise TimeoutError('Queue timed out waiting for data.')
    except BrokenPipeError:
        raise ChildProcessError('A pipe error occurred.')
    finally:
        pipe.send(PIPE_SENTINEL)
        output_queue.close()
        pipe.close()
        # input_queue.close()
        print('Predictor pipe closed.', flush=True)


def make_predictions_torch_all_pipes(
        model, data_pipe: PipeType, pipe: PipeType):
    """Make predictions on input data, update weights with data from pipe, and send predictions through queue.

        Model makes predictions on input data from input_queue. It updates
        whenever there is new data in the pipe to use. It sends all output
        data through

        :param model: Model used to make predictions on sequences of data
        # TODO type for model must be changed later to account for different classes, but this will work for now
        :type model: RecurrentNeuralNetworkTorch
        :param PipeType data_pipe: Pipe for receiving and sending data from and to parent process.
        :param PipeType pipe: Pipe connection to receive model updates through
        """
    count = 0
    # TODO make hidden only if RNN type
    try:
        with data_pipe, pipe:
            data = data_pipe.recv()
            hidden = model.make_hidden_state()  # Add batch size if batched
            while data is not PIPE_SENTINEL:
                count += 1
                print(f'Predictor looking at data slice #{count}', flush=True)
                with torch.no_grad():
                    if isinstance(model, RecurrentNeuralNetworkTorch):
                        data = torch.as_tensor(data)
                    prediction, hidden = model.predict(data, hidden)
                data_pipe.send(prediction)
                del data
                if pipe.poll():  # Assumes this pipe will never get None, may change
                    new_state = pipe.recv()
                    torch_update_state(model, new_state)
                    del new_state
                    pipe.send(PIPE_CONFIRMATION)
                data = data_pipe.recv()
            del data
            # data_pipe.send(PIPE_SENTINEL)
            pipe.send(PIPE_SENTINEL)
    except Empty:
        raise TimeoutError('Queue timed out waiting for data.')
    except BrokenPipeError:
        raise ChildProcessError('A pipe error occurred.')
    finally:
        print('Predictor pipe closed.', flush=True)


def make_improvements_torch_pipe(
        learner: torch.nn.Module, in_queue: QueueType, loss_fn, seq_length: int,
        optimizer: torch.optim.Optimizer, out_pipe: PipeType, out_queue=None):
    """ Update weights of learner and send to optimizer via pipe.

    :param out_queue:
    :param seq_length:
    :param learner: Model for learning from predictions
    :type learner: torch.Module
    :param in_queue: Multiprocessing queue to receive input data
    :type in_queue: QueueType
    :param loss_fn: Function to compute loss between learner predictions and real values
    :param torch.optim.Optimizer optimizer: Optimizer to update learner
    :param PipeType out_pipe: Connection to send updates to predictor
    :rtype: None
    """
    # Defaults for booleans allow model to send weights once
    predictor_done, update_predictor, new_weights = False, True, True
    out_exists = out_queue is not None
    count = 0
    try:
        (data, actual) = in_queue.get(timeout=20)  # Change this later
        hidden = learner.make_hidden_state()
        while data is not None and not predictor_done:  # while learner still needed
            count += 1
            print(f'Learner looking at data slice #{count}', flush=True)
            if isinstance(learner, RecurrentNeuralNetworkTorch):
                data = torch.as_tensor(data)
            prediction, hidden = learner.predict(data, hidden)
            if out_exists:
                out_queue.put(prediction.detach())
            if count % seq_length == 0:  # if enough forward passes have passed
                loss = loss_fn(prediction, actual)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hidden = learner.make_hidden_state()  # Add batch size if batched
                new_weights = True
            del data, actual
            if out_pipe.poll():
                out_data = out_pipe.recv()
                if out_data == PIPE_CONFIRMATION:
                    update_predictor = True
                elif out_data == PIPE_SENTINEL:  # if sentinel, end
                    print('Sentinel received, ending', flush=True)
                    predictor_done = True
                else:
                    raise NotImplementedError('Unexpected pipe message.')
            if update_predictor and new_weights:
                out_pipe.send(learner.state_dict())
                update_predictor, new_weights = False, False  # Data sent, reset
            (data, actual) = in_queue.get(timeout=5)
        del data, actual
        if out_exists:
            out_queue.put(QUEUE_SENTINEL)  # Signal end of messages
            print('Learner sentinel sent', flush=True)
    except TimeoutError:
        raise ChildProcessError('Process timed out waiting for data.')
    except BrokenPipeError:
        print('Caught Broken pipe error', flush=True)
        raise ChildProcessError('A pipe error occurred.')
    finally:
        out_pipe.close()
        if out_exists:
            out_queue.close()
        # in_queue.close()
        print('Learner done', flush=True)


def make_improvements_torch_all_pipes(
        learner: torch.nn.Module, data_pipe: PipeType, loss_fn, seq_length: int,
        optimizer: torch.optim.Optimizer, out_pipe: PipeType, out_queue=False):
    """ Update weights of learner and send to optimizer via pipe.

    :param out_queue:
    :param seq_length:
    :param learner: Model for learning from predictions
    :type learner: torch.Module
    :param data_pipe: Multiprocessing pipe to receive input data and send output data
    :type data_pipe: PipeType
    :param loss_fn: Function to compute loss between learner predictions and real values
    :param torch.optim.Optimizer optimizer: Optimizer to update learner
    :param PipeType out_pipe: Connection to send updates to predictor
    :rtype: None
    """
    # Defaults for booleans allow model to send weights once
    predictor_done, update_predictor, new_weights = False, True, True
    out_exists = out_queue is True
    count, total_count = 0, 0
    try:
        with data_pipe, out_pipe:
            (data, actual) = data_pipe.recv()
            hidden = learner.make_hidden_state()
            while data is not None and not predictor_done:  # while learner still needed
                count += 1
                total_count += 1
                print(f'Learner looking at data slice #{total_count}', flush=True)
                if isinstance(learner, RecurrentNeuralNetworkTorch):
                    data = torch.as_tensor(data)
                prediction, hidden = learner.predict(data, hidden)
                if out_exists:
                    data_pipe.send(prediction.detach())
                if count > seq_length:  # if enough forward passes have passed
                    loss = loss_fn(prediction, actual)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    hidden = learner.make_hidden_state()  # Add batch size if batched
                    new_weights = True
                    count = 0
                del data, actual
                if out_pipe.poll():
                    out_data = out_pipe.recv()
                    if out_data == PIPE_CONFIRMATION:
                        update_predictor = True
                    elif out_data == PIPE_SENTINEL:  # if sentinel, end
                        print('Sentinel received, ending', flush=True)
                        predictor_done = True
                    else:
                        raise NotImplementedError('Unexpected pipe message.')
                if update_predictor and new_weights:
                    out_pipe.send(learner.state_dict())
                    update_predictor, new_weights = False, False  # Data sent, reset
                (data, actual) = data_pipe.recv()
            del data, actual
            # if out_exists:
            #     data_pipe.send(PIPE_SENTINEL)  # Signal end of messages
            out_pipe.send(PIPE_SENTINEL)
            print('Learner sentinel sent', flush=True)
    except BrokenPipeError:
        print('Caught Broken pipe error', flush=True)
        raise ChildProcessError('A pipe error occurred.')
    finally:
        print('Learner done', flush=True)


def update_weights(model: torch.nn.Module, new_weights):
    """ Update weights of model.

        :param model: PyTorch Model
        :type model: torch.nn.Module
        :param new_weights: Iterator of parameters"""
    for (name_1, param_1), (name_2, param_2) in zip(model.named_parameters(), new_weights):
        if name_1 == name_2:  # this should always be true, faster to check all at once
            param_1.data = param_2.data


def torch_update_state(model: torch.nn.Module, new_state_dict: dict) -> None:
    """ Update state dict of model.

        :param model: PyTorch Model
        :type model: torch.nn.Module
        :param dict new_state_dict: State dictionary for PyTorch model
        :rtype: None
        """
    model.load_state_dict(new_state_dict)


def update_weights_lock(model: torch.nn.Module, new_weights, lock: LockType):
    """ Update weights of model when lock is available.

        :param model: PyTorch Model
        :type model: torch.nn.Module
        :param new_weights: Iterator of parameters
        :param LockType lock: Semaphore lock
        :rtype: None
        """
    with lock:
        update_weights(model, new_weights)
