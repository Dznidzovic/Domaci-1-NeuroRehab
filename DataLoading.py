import numpy as np
import scipy.io as scio


class DataLoading:
    #Static method that returns object
    @classmethod
    def returnLoadedData(cls):
        trainData = scio.loadmat('EMGdata-obuka.mat')
        testData = scio.loadmat('EMGdata-test.mat')
        return {
            'emg_train_chs':np.array(trainData['CHS_vezbe']),
            'emg_train_type':np.array(trainData['grasp_type_vezbe']),
            'emg_test_chs':np.array(testData['CHS_provera']),
            'emg_test_type':np.array(testData['grasp_type_provera'])
        }
