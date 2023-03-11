from DataLoading import DataLoading
from DataProccessor import DataProcessor
from ClassificationModel import ClassificationModel
import tkinter as tk
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class myApplication(tk.Tk):

    def prepareData(self):
        self.loaded = True
        loadedData = DataLoading.returnLoadedData()
        self.trainDataSet = DataProcessor.extractFeatures(loadedData['emg_train_chs'],loadedData['emg_train_type'])
        self.testDataSet = DataProcessor.extractFeatures(loadedData['emg_test_chs'],loadedData['emg_test_type'])
        print("Data has been loaded")
    
    def trainData(self):
        if self.loaded == False:
            print("Please load the data first")
            return
        self.model = ClassificationModel.trainModel(self.trainDataSet, self.radio_var.get())
        print("Data has been trained")

    def testData(self):
        if self.model != None:
            testResults = ClassificationModel.testModel(self.testDataSet, self.model)
            self.accuracy.set(str(testResults[1]*100) + '%')
            disp = ConfusionMatrixDisplay(confusion_matrix=testResults[0], display_labels=['1', '2', '3', '4', '5', '6'])
            disp.plot()
            plt.show()
            
    def __init__(self):
        super().__init__()

        self.trainDataSet = None
        self.testDataSet = None
        self.loaded = False
        self.model = None
        self.accuracy = tk.StringVar()
        

        self.title('Classification App')
        self.geometry("600x200")

        self.button_load = tk.Button(self, text='Load Data', command=self.prepareData)
        self.button_load.grid(row=0, column=0, padx=0,pady=0)

        self.train_button = tk.Button(self, text='Train Data', command=self.trainData)
        self.train_button.grid(row=0, column=1, padx=0,pady=0)

        self.test_button = tk.Button(self, text='Test Data', command = self.testData)
        self.test_button.grid(row=0, column=2, padx=0,pady=0)

        self.radio_var = tk.StringVar()
        self.radio_var.set('Linear Discriminant Analysis')  

        self.radio_button_1 = tk.Radiobutton(self, text='Linear Discriminant Analysis', variable=self.radio_var, value='Linear Discriminant Analysis')
        self.radio_button_1.grid(row=1, column=0, padx=0, pady=0)
        self.radio_button_2 = tk.Radiobutton(self, text='QuadraticDiscriminantAnalysis', variable=self.radio_var, value='QuadraticDiscriminantAnalysis')
        self.radio_button_2.grid(row=2,column=0, padx=0,pady=0)
        
        tk.Label(self, text="Accuracy: ",  fg="black", font=20, width=17, height=4).grid(row= 1, column = 1, pady = 0, padx=0)
        self.accuracy_label = tk.Label(self,font=20, textvariable=self.accuracy)
        self.accuracy_label.grid(row = 1, column = 2, pady = 0, padx=0)

def main():
    myApp = myApplication()
    myApp.mainloop()

if __name__ == "__main__":
    main()