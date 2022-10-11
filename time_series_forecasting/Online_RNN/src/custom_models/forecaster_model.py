""" """
# TODO Make sequential version, convert to concurrent
import time
import torch
from multiprocessing import Process, Lock, Pipe, Queue
import numpy as np

from constants import PIPE_SENTINEL, PIPE_CONFIRMATION

# Types
from multiprocessing.queues import Queue as QueueType
from multiprocessing.connection import PipeConnection as PipeType
from multiprocessing.synchronize import Lock as LockType
from typing import Optional
# Exceptions
from queue import Empty, Full


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

# def get_indices(seq_len: int, time_skip: int):
#     """ """
#     start = 0
#     end = time_skip + (seq_len << 1)  # inclusive end index
#     return start, end

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
    seq_length = 10
    start = 0
    end = time_skip + (input_size << 1)  # inclusive end index
    if use_torch and not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)  # Might need to add if clause if not ndarray
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
    curr_q: QueueType = Queue()
    curr_out: QueueType = Queue()
    hist_q: QueueType = Queue()
    hist_out: QueueType = Queue()
    update_lock: LockType = Lock()
    parent_param, child_param = Pipe()
    # Start processes
    p: Process = Process(
        target=make_predictions_torch_pipe,
        args=(pair.predictor, curr_q, curr_out, parent_param))
    q: Process = Process(
        target=make_improvements_torch_pipe,
        args=(pair.learner, hist_q, loss_fn, 10, optimizer, child_param))
    try:
        p.start()
        q.start()
        # Push items to queues, then close
        print('Starting loop')
        print(f'Iterating over array of length {len(data)} from 0 to {end_index}')
        if isinstance(data, torch.Tensor):
            for idx in range(start, end_index):
                end = idx + time_skip + (input_size << 1)  # inclusive end index
                curr_q.put(data[:, (idx + input_size + time_skip): end].detach())
                hist_q.put([data[:, idx: idx + input_size].detach(), data[:, end].unsqueeze(1).detach()])
                time.sleep(0.075)  # TODO this is a workaround to handle when learner finishes first, please fix
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
        curr_q.put(None)
        hist_q.put((None, None))
        print('Inserted end stuff')
        # Close queues
        curr_q.close()
        hist_q.close()
        curr_q.join_thread()
        hist_q.join_thread()
        results = list()
        item = curr_out.get()
        while item is not None:
            results.append(item)
            del item
            item = curr_out.get()
        print(f'Number of results {len(results)}')
        # Close processes
        p.join()
        print(f'Process predictor exit code: {p.exitcode}')
        q.join()
        print(f'Process learner exit code: {q.exitcode}')
    finally:
        # Close processes
        p.join()
        q.join()
        print(f'Process predictor exit code: {p.exitcode} (final)')
        print(f'Process learner exit code: {q.exitcode} (final)')
        curr_q.close()
        curr_out.close()
        hist_q.close()
        parent_param.close()
        child_param.close()
        del curr_q, curr_out, hist_q, parent_param, child_param
        del update_lock
    return results  # list of tensors, fix


# TODO make special Queue class to iterate until None
# def get_data(data, seq_len, time_skip: int):
#     for

def make_predictions(model, queue_or_data: QueueType, output_queue: QueueType = None, lock: LockType = None):
    """ Send predictions on input data asynchronously.

        :param model: Model used to make predictions on sequences of data
        # TODO type for model must be changed later to account for different classes, but this will work for now
        :type model: RecurrentNeuralNetworkTorch
        :param QueueType queue_or_data: Input queue for data
        :param QueueType output_queue: Queue for sending prediction data out
        :param LockType lock: Semaphore lock for model
    """
    # output = output_queue if output_queue is not None else SimpleQueue()
    # if isinstance(queue_or_data, (np.ndarray, torch.Tensor)): # if it's just data
    #     if lock is None:
    #         model.predict(queue_or_data)
    #     else:
    #         try:
    #             lock.acquire()
    #             output = model.predict(queue_or_data)
    #             if output_queue
    #         finally:
    #             lock.release()
    #             if output_queue is not None:
    #                 output_queue.close()
    # TODO maybe add check for if queue, assume queue for now
    try:
        data = queue_or_data.get(timeout=20)  # Change this later
        if isinstance(model, RecurrentNeuralNetworkTorch):
            data = torch.as_tensor(data)
        hidden = model.make_hidden_state()  # Add batch size if batched
        while data is not None:  # TODO this check will only break when get times out
            if lock is not None:
                with lock:
                    prediction, hidden = model.predict(data, hidden)
                # lock.acquire()
                # prediction = model.predict(data)
                # lock.release()
            else:
                prediction, hidden = model.predict(data, hidden)
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach()
            output_queue.put(prediction)
            del data
            data = queue_or_data.get(timeout=20)  # Change this later
            if isinstance(model, RecurrentNeuralNetworkTorch):
                data = torch.as_tensor(data)
        del data
        output_queue.put(None)
    except Empty:
        raise TimeoutError('Queue timed out waiting for data.')
    # except Exception:
    #     raise ChildProcessError('Something unexpected went wrong in predictor.')
    finally:
        output_queue.close()
    # No way to send data back without this, assume no return
    # if output_queue is None:
    #     return data


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
        while data is not None:
            count += 1
            print(f'Predictor looking at data slice #{count}', flush=True)
            if isinstance(model, RecurrentNeuralNetworkTorch):
                data = torch.as_tensor(data)
            prediction, hidden = model.predict(data, hidden)
            if isinstance(model, RecurrentNeuralNetworkTorch):
                prediction = prediction.detach()
            output_queue.put(prediction)
            del data
            if pipe.poll():  # Assumes this pipe will never get None, may change
                new_state = pipe.recv()
                torch_update_state(model, new_state)
                del new_state
                pipe.send(PIPE_CONFIRMATION)
            data = input_queue.get(timeout=5)  # Change this later
        del data
        output_queue.put(None)
    except Empty:
        raise TimeoutError('Queue timed out waiting for data.')
    except BrokenPipeError:
        raise ChildProcessError('A pipe error occurred.')
    # except Exception:
    #     raise ChildProcessError('Something unexpected went wrong in predictor.')
    # else:
    #     pipe.send(PIPE_SENTINEL)
    finally:
        pipe.send(PIPE_SENTINEL)
        output_queue.close()
        pipe.close()
        print('Predictor pipe closed.', flush=True)


# def make_predictions(model, queue_or_data, output_queue, lock):
#     """ """
#     try:
#         lock.acquire()
#         make_predictions(model, queue_or_data, output_queue)
#     finally:
#         lock.release()


# def make_improvements_torch(
#         learner, predictor, data, actual, loss_fn,
#         optimizer: torch.optim.Optimizer):
#     """ """
#     prediction = learner.predict(data)
#     loss = loss_fn(prediction, actual)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     update_weights(predictor, learner.named_parameters())


def make_improvements_torch_pipe(
        learner: torch.nn.Module, in_queue: QueueType, loss_fn, seq_length,
        optimizer: torch.optim.Optimizer, out_pipe: PipeType):
    """ Update weights of learner and send to optimizer via pipe.

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
    # TODO Returning the loss in another queue would be nice
    predictor_done, predictor_updated = False, True
    count = 0
    # This may be worse than a giant try/catch block, will look into
    try:
        (data, actual) = in_queue.get(timeout=20)  # Change this later
        hidden = learner.make_hidden_state()
        while data is not None and not predictor_done:
            count += 1
            print(f'Learner looking at data slice #{count}', flush=True)
            if isinstance(learner, RecurrentNeuralNetworkTorch):
                data = torch.as_tensor(data)
            prediction, hidden = learner.predict(data, hidden)
            if count % seq_length == 0:  # if enough forward passes have passed
                loss = loss_fn(prediction, actual)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hidden = learner.make_hidden_state()  # Add batch size if batched
            if out_pipe.poll(timeout=0.05):
                out_data = out_pipe.recv()
                if out_data == PIPE_CONFIRMATION:  # if confirmed, send next
                    print('Confirmation received, sending next', flush=True)
                    out_pipe.send(learner.state_dict())
                    print('Learner sent data', flush=True)
                elif out_data == PIPE_SENTINEL:  # if sentinel, end
                    print('Sentinel received, ending', flush=True)
                    predictor_done = True
                else:
                    raise NotImplementedError('Unexpected pipe message.')

            # if not out_pipe.poll(timeout=0.05):  # If nothing to receive
            #     print('Sending data.', flush=True)
            #     out_pipe.send(learner.state_dict())
            #     print('Data sent.', flush=True)
            # else:
            #     print('Data received.', flush=True)
            #     out_data = out_pipe.recv()
            #     if out_data is PIPE_SENTINEL:
            #         print('Pipe sentinel detected.', flush=True)
            #         predictor_done = True
            #         break
            del data, actual
            (data, actual) = in_queue.get(timeout=5)
        del data, actual
        print('out of while loop', flush=True)
    except TimeoutError:
        raise ChildProcessError('Process timed out waiting for data.')
    except BrokenPipeError:
        print('Broken pipe error', flush=True)
        raise ChildProcessError('A pipe error occurred.')
    # except Exception:
    #     raise ChildProcessError('Something unexpected went wrong in learner.')
    finally:
        print('Learner pipe closing.', flush=True)
        out_pipe.close()
        print('Learner pipe closed.', flush=True)


def make_improvements_torch_queue(
        learner: torch.nn.Module, in_queue: QueueType, loss_fn, seq_length,
        optimizer: torch.optim.Optimizer, update_queue: QueueType,
        is_rnn: bool = True, out_pred_queue=None, out_loss_queue=None, timeout=None):
    """ Update weights of learner and send to optimizer via pipe.
    # TODO update docstring for new signature
    :param seq_length:
    :param learner: Model for learning from predictions
    :type learner: torch.Module
    :param in_queue: Multiprocessing queue to receive input data
    :type in_queue: QueueType
    :param loss_fn: Function to compute loss between learner predictions and real values
    :param torch.optim.Optimizer optimizer: Optimizer to update learner
    :param PipeType update_queue: Connection to send updates to predictor
    :rtype: None
    """
    # TODO Returning the loss in another queue would be nice
    # TODO Detect when other process is closed and stop sending data
    # TODO Continue to update optimizer until predictor gets next from model
    predictor_done = False
    count = 0
    # This may be worse than a giant try/catch block, will look into
    try:
        (data, actual) = in_queue.get(timeout=timeout)
        hidden = learner.make_hidden_state()
        while data is not None and not predictor_done:
            count += 1
            print(f'Learner looking at data slice #{count}', flush=True)
            if isinstance(learner, RecurrentNeuralNetworkTorch):
                data = torch.as_tensor(data)
            prediction, hidden = learner.predict(data, hidden)
            if count % seq_length == 0:  # if enough forward passes have passed
                loss = loss_fn(prediction, actual)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hidden = learner.make_hidden_state()  # Add batch size if batched
            out_data = update_queue.get()  # TODO change to block and wait
            if out_data is PIPE_SENTINEL:
                predictor_done = True
                break
            else:
                update_queue.put(learner.state_dict())  # TODO change to block and wait
            del data, actual
            (data, actual) = in_queue.get(timeout=timeout)
        del data, actual
    except TimeoutError:
        raise ChildProcessError('Process timed out waiting for data.')
    finally:
        update_queue.close()


# def make_improvements_torch_sm(
#         learner, predictor, data, actual, loss_fn,
#         optimizer: torch.optim.Optimizer, lock: Lock):
#     """ """
#     # TODO Update and add shared memory
#     prediction = learner.predict(data)
#     loss = loss_fn(prediction, actual)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     try:
#         lock.acquire()
#         update_weights(predictor, learner.named_parameters())
#     finally:
#         lock.release()


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
        :param lock: Semaphore lock
        :type lock: LockType
        :rtype: None
        """
    with lock:
        update_weights(model, new_weights)
