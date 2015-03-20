
###########
# Imports #
###########
import csv


#############
# Time_slot #
#############
class time_slot:

    # Constructor
    def __init__(self):

        self.raw_EDA = []
        self.raw_BVP = []
        self.raw_ACCx = []
        self.raw_ACCy = []
        self.raw_ACCz = []

        self.mean_EDA = 0.0
        self.variance_EDA = 0.0

        self.mean_BVP = 0.0
        self.variance_BVP = 0.0

        self.mean_ACCx = 0.0
        self.variance_ACCx = 0.0

        self.mean_ACCy = 0.0
        self.variance_ACCy = 0.0

        self.mean_ACCz = 0.0
        self.variance_ACCz = 0.0


#################
# Bracelet data #
#################
class bracelet_measurement_session:

    # Constructor, initialize attributes
    def __init__(self, my_patient, my_emotion, slot_time, expected_frequency):

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
                if row_id>20:
                    break
                else:
                    print row

        # print
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

                # store all values in time unit
                if (row_id-2)%round((self.BVP_hertz/self.exp_samplepersec))==0:
                    unit_avg = slot_accumulator/round((self.BVP_hertz/self.exp_samplepersec))
                    self.raw_BVP.append(unit_avg)
                    slot_accumulator = float(row[0])
                else:
                    slot_accumulator += float(row[0])

                # average samples in order to have one aggregated measurement per each 0.25s time unit
                if row_id>100:
                    break
                else:
                    print row

        # print
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

                # store all values in time unit
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

                # average samples in order to have one aggregated measurement per each 0.25s time unit
                if row_id>50:
                    break
                else:
                    print row

        # print
        print self.raw_ACCx, len(self.raw_BVP)
        print self.raw_ACCy, len(self.raw_BVP)
        print self.raw_ACCz, len(self.raw_BVP)


########
# Main #
########

data = bracelet_measurement_session("Gaziz", "calm", 30.0, 4.0)
data.read_EDA()
data.read_BVP()
data.read_ACC()