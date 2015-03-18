__author__ = 'miljan'


import csv
import pickle
from collections import defaultdict
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from pprint import pprint


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


def detect_local_minima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    return np.where(detected_minima)


def feature_difference_of_maxes(data_slice):
    local_minima = detect_local_minima(data_slice)
    pprint(local_minima)
    pprint( data_slice[local_minima])
    pprint(data_slice)


def calculate_window_features(data, window_size=30):
    data = np.reshape(data,(data.shape[0] * 4, 25))
    # calculate average measurement for each second
    data = np.average(data, 1)
    for i in range(0, 120, 120):
        try:
            data_slice = data[i:i+120]
        except:
            data_slice = data[i:]
        feature_difference_of_maxes(data_slice)


if __name__ == '__main__':
    processed_data = process_breath_data('_Respiration_Data_Calm_Gaziz')
    calculate_window_features(processed_data)