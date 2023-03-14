import numpy as np

class DataProcessor:

    def calculateMeanAbsoluteValue(self, array):
        return np.mean(np.abs(array))

    def calculateRms(self, array):
        return np.sqrt(np.mean(np.square(array)))
    
    def calculateWaveFormLength(self,array):
        return np.sum(np.abs(np.diff(array)))
    
    @classmethod
    def extractFeatures(self, emgChs, emg_type):
        # Sample frequency is 2000, window size is 250ms(quater of a second).
        # Quater of a second with sampling frequency of 2000 equals 500 samples
        step = 500
        #This vector will be a 1x25 shape.
        #It holds calculated values for every channel(3x8) and a class corresponding to it
        calculatedValuesVector = []
        processedData = np.empty((0, 25), float)
        for i in range(0, emgChs.shape[1], int(step/2)):
            calculatedValuesVector = []
            for j in range(emgChs.shape[0]):
                currentWindow = emgChs[j, i: i + step - 1]
                calculatedValuesVector.append(self.calculateMeanAbsoluteValue(self, currentWindow))
                calculatedValuesVector.append(self.calculateRms(self, currentWindow))
                calculatedValuesVector.append(self.calculateWaveFormLength(self, currentWindow))

            if (emg_type[0, i:i+step-1][0] != emg_type[0, i:i+step-1][-1]):
                    continue
               
            calculatedValuesVector.append(np.mean(emg_type[0, i : i + step - 1]))
            resizedCalculatedValues = np.array(calculatedValuesVector).reshape(1,25)
            processedData = np.append(processedData, resizedCalculatedValues, axis=0)
                    
        return processedData
         
         

