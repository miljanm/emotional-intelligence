
###########
# Imports #
###########
import csv
import numpy as np


#################
# Bracelet data #
#################
class bracelet_measurement_session:

    # Constructor, initialize attributes
    def __init__(self, my_patient, my_emotion, slot_time, expected_frequency, debug = False):

        self.debug_mode = debug

        self.patient = my_patient
        self.emotion = my_emotion

        self.slot_time_span = slot_time
        self.exp_samplepersec = expected_frequency

        self.ACC_hertz = 32.0
        self.ACCsamples_slot = self.ACC_hertz*self.slot_time_span

        self.BVP_hertz = 64.0
        self.BVPsamples_slot = self.BVP_hertz*self.slot_time_span

        self.EDA_hertz = 4.0
        self.EDAsamples_slot = self.EDA_hertz*self.slot_time_span

        self.raw_EDA = []
        self.raw_BVP = []
        self.raw_ACCx = []
        self.raw_ACCy = []
        self.raw_ACCz = []

        self.features_EDA = []
        self.features_BVP = []
        self.features_ACCx = []
        self.features_ACCy = []
        self.features_ACCz = []


    # subtract day average for patient
    def normalize_signals(self):
        pass


    # compute slot mean
    def mean_feature(self, slot):

        my_sum = sum(slot)
        my_avg = float(my_sum)/len(slot)
        return my_avg


    # compute slot variance
    def stddev_feature(self, slot, mean):

        numpy_slot = np.array(slot)
        std_dev = np.std(numpy_slot)
        return std_dev


    # compute slot mean approx absolute value of gradient
    def approx_gradient_feature(self, slot):

        numpy_slot = np.array(slot)
        diff_vect = np.diff(numpy_slot)
        abs_diff_vect = np.absolute(diff_vect)
        avg_grad = np.mean(abs_diff_vect)
        return avg_grad


    # compute slot mean approx absolute value of second order gradient
    def approx_secondorder_feature(self):
        pass


    # compute mean amplitude
    def amplitude_feature(self):
        pass


    # compute mean frequency
    def frequency_feature(self):
        pass


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
                if (row_id-2)%round((self.EDA_hertz/self.exp_samplepersec))==0:
                    unit_avg = slot_accumulator/round((self.EDA_hertz/self.exp_samplepersec))
                    self.raw_EDA.append(unit_avg)
                    slot_accumulator = float(row[0])
                else:
                    slot_accumulator += float(row[0])

                # average samples in order to have one aggregated measurement per each 0.25s time unit
                if self.debug_mode and row_id>20:
                    break

        # print
        if self.debug_mode:
            print self.raw_EDA, len(self.raw_EDA)


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
                if (row_id-2)%round((self.BVP_hertz/self.exp_samplepersec))==0:
                    unit_avg = slot_accumulator/round((self.BVP_hertz/self.exp_samplepersec))
                    self.raw_BVP.append(unit_avg)
                    slot_accumulator = float(row[0])
                else:
                    slot_accumulator += float(row[0])

                # early break and prints
                if self.debug_mode and row_id>100:
                        break

        # print
        if self.debug_mode:
            print self.raw_BVP, len(self.raw_BVP)


    # ACC: Read data and interpolate/average to get the required sampling frequency
    def read_ACC(self):

        # initialize
        slot_accumulator_x = 0.0
        slot_accumulator_y = 0.0
        slot_accumulator_z = 0.0

        with open("../data/bracelet/%s_%s/ACC.csv"%(self.patient, self.emotion), "rb") as f1:

            file_reader = csv.reader(f1, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue

                # average samples in order to have one aggregated measurement per each time unit
                if (row_id-2)%round((self.ACC_hertz/self.exp_samplepersec))==0:
                    unit_avg = slot_accumulator_x/round((self.ACC_hertz/self.exp_samplepersec))
                    self.raw_ACCx.append(unit_avg)
                    slot_accumulator_x = float(row[0].split(',')[0])
                    unit_avg = slot_accumulator_y/round((self.ACC_hertz/self.exp_samplepersec))
                    self.raw_ACCy.append(unit_avg)
                    slot_accumulator_y = float(row[0].split(',')[1])
                    unit_avg = slot_accumulator_z/round((self.ACC_hertz/self.exp_samplepersec))
                    self.raw_ACCz.append(unit_avg)
                    slot_accumulator_z = float(row[0].split(',')[2])
                else:
                    slot_accumulator_x += float(row[0].split(',')[0])
                    slot_accumulator_y += float(row[0].split(',')[1])
                    slot_accumulator_z += float(row[0].split(',')[2])

                # early break and prints
                if self.debug_mode and row_id>50:
                        break

        # print
        if self.debug_mode:
            print self.raw_ACCx, len(self.raw_BVP)
            print self.raw_ACCy, len(self.raw_BVP)
            print self.raw_ACCz, len(self.raw_BVP)


########
# Main #
########

data = bracelet_measurement_session("Gaziz", "calm", 30.0, 4.0, debug = True)
data.read_EDA()
data.read_BVP()
data.read_ACC()