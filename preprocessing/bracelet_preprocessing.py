
###########
# Imports #
###########
import csv


#################
# Bracelet data #
#################
class bracelet_measurement_session:

    # Constructor, initialize attributes
    def __init__(self, my_patient, my_emotion):

        self.slot_time_span = 30

        self.patient = my_patient
        self.emotion = my_emotion

        self.ACC_hertz = 32
        self.ACCsamples_slot = self.ACC_hertz*time_slot

        self.BVP_hertz = 64
        self.BVPsamples_slot = self.BVP_hertz*time_slot

        self.EDA_hertz = 4
        self.EDAsamples_slot = self.EDA_hertz*time_slot


    # Preprocess data
    def preprocess_EDA(self):

        with open("../data/bracelet/%s_%s/EDA.csv"%(self.patient, self.emotion), "rb") as f1:

            file_reader = csv.reader(f1, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue

                # read data
                # average samples in order to have one aggregated measurement per each 0.25s time unit


    # Preprocess data
    def preprocess_BVP(self):

        with open("../data/bracelet/%s_%s/BVP.csv"%(self.patient, self.emotion), "rb") as f2:

            file_reader = csv.reader(f2, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue

                # read data
                # average samples in order to have one aggregated measurement per each 0.25s time unit


    # Preprocess data
    def preprocess_ACC(self):

        with open("../data/bracelet/%s_%s/ACC.csv"%(self.patient, self.emotion), "rb") as f3:

            file_reader = csv.reader(f3, delimiter=' ')
            for row_id, row in enumerate(file_reader):

                # skip first 2 lines
                if row_id == 0 or row_id == 1:
                    continue

                # read data
                # average samples in order to have one aggregated measurement per each 0.25s time unit

########
# Main #
########
time_slot = 30
unit_interval = 0.25
