from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score

class ClassificationModel:
    @classmethod
    def trainModel(self, trainDataSet, value):
    
        x_train = trainDataSet[:, :-1]
        y_train = trainDataSet[:, -1:]
        

        if value == 'Linear Discriminant Analysis':
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train,y_train.ravel())
            return lda
        else:
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(x_train,y_train.ravel())
            return qda
    
    @classmethod
    def testModel(self, testDataSet, model):
        x_test  = testDataSet[:, :-1]
        y_test = testDataSet[:, -1:]
        y_predicted = model.predict(x_test)
        acc_score = accuracy_score(y_test,y_predicted)
        cfmatrix = confusion_matrix(y_test,y_predicted)
        return cfmatrix, acc_score
        

