__author__ = 'miljan'


import csv
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pylab as pl
from pprint import pprint
import preprocessing_utils


def process_breath_data(filename, is_pickled=False):
    """ Takes the raw breath data and returns a numpy matrix with each row representing a second
        with sampling rate of 100 Hz

    :param filename: raw data file name (without extension)
    :param is_pickled: flag indicating if data should be pickled or not
    :return: numpy 2D matrix
    """

    # holds the data from the breath meter
    breath_data = defaultdict(list)

    with open('../data/breath/' + filename + '.txt') as file1:
        file_reader = csv.reader(file1, delimiter=' ')
        # read first measurement to get
        for row_id, row in enumerate(file_reader):
            # skip first 2 lines
            if row_id == 0 or row_id == 1:
                continue
            # get the first second
            elif row_id == 2:
                first_second = int(row[1].split('.')[0])
            # get the current second and measurement
            second = int(row[1].split('.')[0]) - first_second
            measurement = row[2]
            breath_data[second].append(measurement)
        pass

    # matrix to hold the breath data with frequency = 100 Hz
    breath_data_np = np.zeros((len(breath_data), 100), dtype=float)
    # convert the breath data into a numpy array, with frequency of 100 Hz
    for key_id, key in enumerate(breath_data.keys()):
        # all the values for the given second (key)
        values = breath_data[key]
        # get the last value of the list to use as a potential padding if the frequency is not 100 Hz
        pad_value = values[-1]
        if len(values) < 100:
            values += [pad_value] * (breath_data_np.shape[1] - len(values))
        elif len(values) > 100:
            values = values[0:100]
        # use row_id since there are couple of missing seconds in dict data
        breath_data_np[key_id] = np.array(values, dtype=float)

    if is_pickled:
        pickle.dump(breath_data_np, open('../data/pickles/' + filename + '.pickle', 'wb'))

    return breath_data_np


def _slice_local_maximums(data_slice):

    pl.plot(data_slice)
    smoothed_data = preprocessing_utils.smooth_signal(data_slice, window_len=15, window='bartlett')
    pl.plot(smoothed_data[7:-7])
    pl.show()
    local_mins = preprocessing_utils.detect_local_minima(smoothed_data)
    pprint(local_mins)



def _slice_first_abs_difference_signals(data_slice):
    return np.sum(np.abs(data_slice[1:] - data_slice[:-1])) / (len(data_slice) - 1)


def _slice_second_abs_difference_signals(data_slice):
    return np.sum(np.abs(data_slice[2:] - data_slice[:-2])) / (len(data_slice) - 2)


def _slice_amplitude(data_slice):
    return np.std(data_slice)


def _slice_mean(data_slice):
    return np.mean(data_slice)


def calculate_daily_user_mean(username, is_pickled=False):
    calm_data = process_breath_data('_Respiration_Data_Calm_' + username)
    excited_data = process_breath_data('_Respiration_Data_Excited_' + username)
    mean = (np.average(calm_data) + np.average(excited_data)) / 2.0
    if is_pickled:
        pickle.dump(mean, open('../data/pickles/_Respiration_Data_Average_' + username, 'wb'))
    return mean


def calculate_window_features(data, username, features, window_size=30):
    data = np.reshape(data, (data.shape[0] * 4, 25))
    # calculate average measurement for each second
    data = np.average(data, 1)
    # subtract the mean of the day to account for variations in the tightness of the breath sensor
    data = np.subtract(data, calculate_daily_user_mean(username))
    all_features = []
    # go over data ranges, in window_size range * 4 Hz steps
    for i in range(0, len(data), window_size*4):
        try:
            data_slice = data[i:i+window_size*4]
        except:
            data_slice = data[i:]

        feature_vector = []
        # call the feature functions given in the list of feature functions to be used
        for feature in features:
            feature_vector.append(feature(data_slice))
        all_features.append(feature_vector)

    return np.array(all_features)


if __name__ == '__main__':
    username = 'Gaziz'
    features = [
        _slice_local_maximums,
        _slice_first_abs_difference_signals,
        _slice_second_abs_difference_signals,
        _slice_amplitude,
        _slice_mean,
    ]
    processed_data = process_breath_data('_Respiration_Data_Calm_' + username)
    calculate_window_features(processed_data, username, features)