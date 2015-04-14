
import pickle
import csv
import numpy as np
import math


"""
Data structure to store and process the data corresponding to one measurement session:
- features is a dictionary with the list of features to be computer for each type of data
- feat_func is a dictionary that correlates the name of the feature to the function that computes it
"""
class bracelet_measurement_session:

    # Constructor, initialize attributes
    def __init__(self, features, my_patient, my_emotion, slot_time, sampling_frequency, expected_frequency, debug=False):

        self.debug_mode = debug
        self.patient = my_patient
        self.emotion = my_emotion

        self.slot_time_span = slot_time
        self.exp_sample_per_sec = expected_frequency
        self.exp_sample_per_slot = self.slot_time_span*self.exp_sample_per_sec

        self.sampling = {'ACC':sampling_frequency[0], 'EDA':sampling_frequency[1], 'BVP':sampling_frequency[2]}
        self.data = {'ACCx': [], 'ACCy': [], 'ACCz': [], 'EDA': [], 'BVP': []}

        self.feat = features
        self.feat_func = {
            'mean': self.mean_feature,
            'grad': self.approx_gradient_feature,
            'std_dev': self.stddev_feature,
        }
        self.num_feat = sum([len(x) for x in features.values()])

        self.engineered_data = []

    # TODO subtract day average for patient
    def normalize_signals(self, avg_list):
        pass

    # compute slot mean
    def mean_feature(self, slot):
        return float(sum(slot))/len(slot)

    # compute slot variance
    def stddev_feature(self, slot):
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
        self.engineered_data = np.zeros((num_slots, self.num_feat+1))

        # iterate over 30 second slots
        for i in xrange(num_slots):
            # collect data slots
            my_slots = map(lambda s: self.data[s][int(i*self.exp_sample_per_slot) : int((i+1)*self.exp_sample_per_slot)], signals)
            # iterate over sensors and compute features
            feat_counter = 0
            for s in my_slots:
                sig = signals[my_slots.index(s)]
                sig_features = map(lambda ff: self.feat_func[ff](s), self.feat[sig])
                for w in xrange(len(sig_features)):
                    self.engineered_data[i,feat_counter+w] = sig_features[w]
                feat_counter += len(self.feat[sig])

                if self.emotion.lower()=='calm':
                    self.engineered_data[i,-1] = 0
                elif self.emotion.lower()=='excited':
                    self.engineered_data[i,-1] = 1
                else:
                    self.engineered_data[i,-1] = 2

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
                if self.debug_mode and row_id>500:
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
                if self.debug_mode and row_id>200:
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
                if self.debug_mode and row_id>400:
                        break

        # print
        if self.debug_mode:
            print self.data['BVP'], len(self.data['BVP'])


"""
Utility functions:
- derive global, inter session data
- separate neutral slots
"""
def get_patient_averages(sessions, patient):
    pass

def get_neutral_slots(sessions, patient):
    pass



'''
Main function:
- declare features to be computed
- read data and divide in slots
- process slots and derive feature
- pickle data
'''
if __name__=='__main__':

    features = {
        'EDA': ['mean', 'grad'],
        'BVP': ['mean', 'std_dev'],
        'ACCx': ['mean', 'std_dev'],
        'ACCy': ['mean', 'std_dev'],
        'ACCz': ['mean', 'std_dev']
    }

    paramsGC = ["Gaziz", "calm", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsGE = ["Gaziz", "excited", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsMC = ["Matteo", "calm", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsME = ["Matteo", "excited", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsG2C = ["Gaziz2", "Calm", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsG2E = ["Gaziz2", "Excited", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsG2N = ["Gaziz2", "Neutral", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsJC = ["James", "Calm", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsJE = ["James", "Excited", 30.0, [32.0, 4.0, 64.0], 4.0,]
    paramsJN = ["James", "Neutral", 30.0, [32.0, 4.0, 64.0], 4.0,]
    params = [paramsGC, paramsGE, paramsG2C, paramsG2E, paramsG2N, paramsMC, paramsME, paramsJC, paramsJE, paramsJN]

    boundaries = [range(0,48),range(0,39),range(0,30),range(0,30),range(0,22),range(0,32),range(0,27),range(0,29),range(0,31), range(0,19)]

    sessions = []
    for p in params:
        session = bracelet_measurement_session(features, *p, debug=False)
        session.read_EDA()
        session.read_BVP()
        session.read_ACC()
        session.normalize_signals(get_patient_averages(sessions, p[0]))
        session.compute_features()
        sessions.append(session)
        print session.engineered_data.shape

    i = 0
    data_set = sessions[0].engineered_data[boundaries[0],:]
    for s in sessions:
        if i==0:
            i+=1
            continue
        data_set = np.concatenate((data_set, s.engineered_data[boundaries[i],:]), axis=0)
        i+=1

    print sessions[0].engineered_data.shape, sessions[1].engineered_data.shape, sessions[2].engineered_data[:-1,:].shape, sessions[3].engineered_data.shape
    print data_set.shape

    pickle.dump(data_set, open("../data/bracelet.pkl", "wb"))
