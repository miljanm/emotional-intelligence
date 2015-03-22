
###########
# Imports #
###########
import csv
import numpy as np
import math


################################
# Bracelet measurement session #
################################
class bracelet_measurement_session:

    # Constructor, initialize attributes
    def __init__(self, features, my_patient, my_emotion, slot_time, sampling_frequency, expected_frequency, debug = False):

        self.debug_mode = debug
        self.patient = my_patient
        self.emotion = my_emotion

        self.slot_time_span = slot_time
        self.exp_sample_per_sec = expected_frequency
        self.exp_sample_per_slot = self.slot_time_span*self.exp_sample_per_sec

        self.sampling = {'ACC':sampling_frequency[0], 'EDA':sampling_frequency[1], 'BVP':sampling_frequency[2]}
        self.data = {'ACCx': [], 'ACCy': [], 'ACCz': [], 'EDA': [], 'BVP': []}

        self.num_feat = sum([len(x) for x in features.values()])
        self.engineered_data = []

    # TODO subtract day average for patient
    def normalize_signals(self):
        pass

    # compute slot mean
    def mean_feature(self, slot):
        return float(sum(slot))/len(slot)

    # compute slot variance
    def stddev_feature(self, slot, mean):
        return np.std(np.array(slot))

    # compute slot mean approx absolute value of gradient
    def approx_gradient_feature(self, slot):
        abs_diff_vect = np.absolute(np.diff(np.array(slot)))
        return np.mean(abs_diff_vect)

    # compute slot mean approx absolute value of second order gradient
    def approx_secondorder_feature(self, slot):
        abs_second_diff = np.absolute(np.diff(np.diff(np.array(slot))))
        return np.mean(abs_second_diff)

    # TODO compute mean frequency
    def frequency_feature(self, slot):
        pass

    # compute features
    def compute_features(self):
        # compute number of slots
        num_slots = int(math.floor(float(len(self.data['EDA']))/self.exp_sample_per_slot))
        signals = ['EDA', 'BVP', 'ACCx', 'ACCy', 'ACCz']
        # iterate over 30 second slots
        for i in xrange(num_slots):
            # TODO: compute slot fetures
            slots = map(lambda s: self.data[s][int(i*self.exp_sample_per_slot) : int((i+1)*self.exp_sample_per_slot)], signals)

    # ACC: Read data and interpolate/average to get the required sampling frequency
    def read_ACC(self):

        # initialize
        axes = ['x', 'y', 'z']
        raw_data = {'x':self.data['ACCx'], 'y':self.data['ACCy'], 'z':self.data['ACCz']}
        slot_accumulator = {'x':0.0, 'y':0.0, 'z':0.0}

        with open("../data/bracelet/%s_%s/ACC.csv"%(self.patient, self.emotion), "rb") as f1:
            file_reader = csv.reader(f1, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue
                # average samples in order to have one aggregated measurement per each time unit
                if (row_id-2)%round((self.sampling['ACC']/self.exp_sample_per_sec))==0:
                    i=0
                    for axis in axes:
                        unit_avg = slot_accumulator[axis]/round((self.sampling['ACC']/self.exp_sample_per_sec))
                        raw_data[axis].append(unit_avg)
                        slot_accumulator[axis] = float(row[0].split(',')[i])
                        i += 1
                else:
                    i=0
                    for axis in axes:
                        slot_accumulator[axis] += float(row[0].split(',')[i])
                        i += 1
                # early break and prints
                if self.debug_mode and row_id>50:
                        break

        # print
        if self.debug_mode:
            print self.data['ACCx'], len(self.data['ACCx'])
            print self.data['ACCy'], len(self.data['ACCy'])
            print self.data['ACCz'], len(self.data['ACCz'])

    # EDA: Read data and interpolate/average to get the required sampling frequency
    def read_EDA(self):

        # initialize
        slot_accumulator = 0.0

        # open approapriate file
        with open("../data/bracelet/%s_%s/EDA.csv"%(self.patient, self.emotion), "rb") as f1:

            file_reader = csv.reader(f1, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue

                # store all values in time unit
                if (row_id-2)%round((self.sampling['EDA']/self.exp_sample_per_sec))==0:
                    unit_avg = slot_accumulator/round((self.sampling['EDA']/self.exp_sample_per_sec))
                    self.data['EDA'].append(unit_avg)
                    slot_accumulator = float(row[0])
                else:
                    slot_accumulator += float(row[0])

                # average samples in order to have one aggregated measurement per each 0.25s time unit
                if self.debug_mode and row_id>20:
                    break

        # print
        if self.debug_mode:
            print self.data['EDA'], len(self.data['EDA'])

    # BVP: Read data and interpolate/average to get the required sampling frequency
    def read_BVP(self):

        # initialize
        slot_accumulator = 0.0

        # open approapriate file
        with open("../data/bracelet/%s_%s/BVP.csv"%(self.patient, self.emotion), "rb") as f1:

            file_reader = csv.reader(f1, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue

                # average samples in order to have one aggregated measurement per each 0.25s time unit
                if (row_id-2)%round((self.sampling['BVP']/self.exp_sample_per_sec))==0:
                    unit_avg = slot_accumulator/round((self.sampling['BVP']/self.exp_sample_per_sec))
                    self.data['BVP'].append(unit_avg)
                    slot_accumulator = float(row[0])
                else:
                    slot_accumulator += float(row[0])

                # early break and prints
                if self.debug_mode and row_id>100:
                        break

        # print
        if self.debug_mode:
            print self.data['BVP'], len(self.data['BVP'])


###########
# Execute #
###########

features = {'EDA': ['mean', 'grad'], 'BVP': ['mean', 'std_dev'], 'ACC': ['mean', 'std_dev']}
session = ["Gaziz", "calm", 30.0, [32.0, 4.0, 64.0], 4.0,]

data = bracelet_measurement_session(features, *session, debug = True)

data.read_EDA()
data.read_BVP()
data.read_ACC()