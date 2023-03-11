from DataLoading import DataLoading
from DataProccessor import DataProcessor
from ClassificationModel import ClassificationModel


class myApplication:
    def prepareData(self):
        loadedData = DataLoading.returnLoadedData()
        trainDataSet = DataProcessor.extractFeatures(loadedData['emg_train_chs'],loadedData['emg_train_type'])
        testDataSet = DataProcessor.extractFeatures(loadedData['emg_test_chs'],loadedData['emg_test_type'])
        ClassificationModel.ldClassification(trainDataSet, testDataSet)
        ClassificationModel.qdaClassification(trainDataSet, testDataSet)
       
def main():
    myApp = myApplication()
    myApp.prepareData()

if __name__ == "__main__":
    main()