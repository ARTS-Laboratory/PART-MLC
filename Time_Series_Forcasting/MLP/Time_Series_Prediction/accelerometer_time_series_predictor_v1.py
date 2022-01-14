import numpy as np
import sklearn.neural_network
from matplotlib import pyplot as plt
from sklearn import preprocessing


def load_data(filename, separate_time=True):
    fn_data = np.loadtxt(filename, delimiter=',', skiprows=2)
    if separate_time:
        return fn_data[:, 0], fn_data[:, 1:]
    else:
        return fn_data


def load_data_files(filenames, separate_time=True):
    fn_data = np.array([load_data(name, separate_time=False) for name in filenames])
    if separate_time:
        return fn_data[:, :, 0], fn_data[:, :, 1:]
    else:
        return fn_data


def prepare_data(complete, scaling=None, return_labels=False):
    returned_values = dict()
    print('Complete shape: {}'.format(complete.shape))
    full_time = complete[:, :, 0]
    full_data, labels = [], []
    print('Full time shape: {}'.format(full_time.shape))
    for example_set in complete.transpose((0, 2, 1)):
        for test_num, test in enumerate(example_set[1:5]):
            if np.sum(np.square(test)) > 1e-8: # numbers aren't all zero
                # print('Test #{} shape: {}'.format(test_num + 1, test.shape))
                labels.append(test_num + 1)
                full_data.append(test)
    full_data, labels = np.array(full_data), np.array(labels)
    returned_values['times'] = full_time
    returned_values['data'] = full_data
    if return_labels:
        returned_values['labels'] = labels
    if scaling is not None and scaling == 'normalize':
        returned_values['normalized'], returned_values['scalars'] = normalize_data(full_data)
    return returned_values



def normalize_data(data):
    norm_data = None
    if np.ndim(data) < 2 or np.ndim(data) > 3:
        return None
    elif np.ndim(data) == 3:
        norm_data = data.reshape((data.shape[1], -1))
    elif np.ndim(data) == 2:
        norm_data = data.reshape((data.shape[0], -1))
    norm_data, norm = preprocessing.normalize(norm_data, norm='l2', axis=1, return_norm=True)
    norm_data = norm_data.reshape(data.shape)
    return norm_data, norm


def denormalize_data(data, norms):
    norm_data = None
    if np.ndim(data) < 2 or np.ndim(data) > 3:
        return None
    elif np.ndim(data) == 3:
        norm_data = data.reshape((data.shape[1], -1))
    elif np.ndim(data) == 2:
        norm_data = data.reshape((data.shape[0], -1))
    resized = np.array([norm_data[idx, :] * norm_val for idx, norm_val in enumerate(norms)])
    resized = resized.reshape(data.shape)
    return resized

def sliding_window_over_data(data, start_idx, window_length, prediction_offset, prediction_length):
    x_values, y_values = [], []
    curr_idx = start_idx
    while curr_idx + window_length + prediction_length + prediction_offset < len(data):
        x_values.append(data[curr_idx: curr_idx + window_length])
        y_values.append(data[curr_idx + window_length + prediction_offset: curr_idx + window_length + prediction_length + prediction_offset])
        curr_idx += 1
    return x_values, y_values

if __name__=='__main__':
    start_idx, window_length, prediction_offset, prediction_length = 0, 100, 100, 10
    full_time, full_data = load_data('data/accel_4.csv', separate_time=True)
    print('Data shape: {}, Time shape: {}'.format(full_data.shape, full_time.shape))
    windowed_x, windowed_y = sliding_window_over_data(full_data.transpose()[0], start_idx=start_idx,
                                                      window_length=window_length, prediction_offset=prediction_offset,
                                                      prediction_length=prediction_length)
    print('Windowed x data shape: {}, Windowed y data shape: {}'.format(len(windowed_x), len(windowed_y)))
    model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(20, 20, 10), max_iter=4000, early_stopping=True,
                                                n_iter_no_change=100, batch_size=16, solver='adam').fit(windowed_x, windowed_y)
    print('Model score: {} after {} epoch{}'.format(model.score(windowed_x, windowed_y),
                                                    model.n_iter_, 's' if model.n_iter_ > 1 else ''))
    print('Current loss: {}'.format(model.loss_))
    
    #%% Plot the data
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(model.loss_curve_)
    plt.title('Loss curve')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.xlim([0, 20])
    plt.savefig('MLP_Loss_Curve')
    
    
    plt.figure(figsize=(6.4, 4.8))
    print(len(windowed_x))
    print(len(windowed_x[0]))
    for idx, window in enumerate(windowed_x):
        prediction = model.predict([window])[0]
        plt.scatter(full_time[idx + window_length + prediction_offset:
                              idx + window_length + prediction_offset + len(prediction)], prediction, 1)
    # prediction = model.predict(windowed_x)
    # plt.scatter(full_time[0: len(prediction)], prediction, 1, label='Predicted points', color='orange')
    plt.plot(full_time, full_data.transpose()[0], label='Actual points')
    plt.title('Real vs Predicted plots')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Future_predictions_plot')
    plt.show()
    coefs, intercepts = model.coefs_, model.intercepts_
