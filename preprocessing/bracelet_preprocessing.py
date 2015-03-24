

import csv
import numpy as np
import math
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


"""
Data structure to store and process the data corresponding to one measurement session:
- features is a dictionary with the list of features to be computer for each type of data
- feat_func is a ictionary that correlates the name of the feature to the function that computes it
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
                self.engineered_data[i,-1] = int(self.emotion=='calm')


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
- train logistic regression
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
    params = [paramsGC, paramsGE, paramsMC, paramsME]

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

    data_set = np.concatenate((sessions[0].engineered_data,sessions[1].engineered_data,sessions[2].engineered_data, sessions[3].engineered_data), axis=0)
    X = data_set[:,0:-1]
    np.random.shuffle(X)
    y = data_set[:,-1].astype('int')-1

    # Experiment with different classifiers
    n_folds = 5

    clf = LogisticRegression(dual=False, penalty='l1')
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("LogReg l1 regularized - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = LogisticRegression(dual=False, penalty='l2')
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("LogReg l2 regularized - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = RandomForestClassifier(n_estimators=180,min_samples_split=4)
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("Random Forest - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = GradientBoostingClassifier(n_estimators=180,min_samples_split=4)
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("Gradient Boosting - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("Linear SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = svm.SVC(kernel='poly', degree=2, C=1)
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("Polynomial (d=2) kernel SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = svm.SVC(kernel='poly', degree=3, C=1)
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("Polynomial (d=3) kernel SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = svm.SVC(kernel='sigmoid', C=1)
    scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
    print("Sigmoid kernel SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





"""
    X = data_set[:,0:-1]
    np.random.shuffle(X)
    y = data_set[:,-1].astype('int')-1
    print (X.shape)[0]
    train_idx = int(0.7*(X.shape)[0])
    X_train = X[0:train_idx,:]
    X_test = X[train_idx:,:]
    y_train = y[0:train_idx]
    y_test = y[train_idx:]

    print X_test.shape
    print X_train.shape
    print y_test.shape
    print y_train.shape

    m = LogisticRegression(dual=False, penalty='l2')
    m.fit(X_train, y_train)
    pred = m.predict(X_test)

    print "\n Error:"
    pred = abs(pred-1)
    print float(sum(abs(y_test-pred)))/len(y_test)

    m = svm.SVC(kernel='poly')
    m.fit(X_train, y_train)
    pred = m.predict(X_test)

    print "\n Error:"
    pred = abs(pred-1)
    print float(sum(abs(y_test-pred)))/len(y_test)
"""