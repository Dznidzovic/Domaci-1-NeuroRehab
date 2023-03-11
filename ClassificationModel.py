from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score

class ClassificationModel:
    @classmethod
    def ldClassification(self, trainDataSet, testDataSet):
        x_train = trainDataSet[:, :-1]
        y_train = trainDataSet[:, -1:]
        x_test  = testDataSet[:, :-1]
        y_test = testDataSet[:, -1:]
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train,y_train.ravel())
        y_predicted = lda.predict(x_test)
        print(accuracy_score(y_test,y_predicted))

    @classmethod
    def qdaClassification(self, trainDataSet, testDataSet):
        x_train = trainDataSet[:, :-1]
        y_train = trainDataSet[:, -1:]
        x_test  = testDataSet[:, :-1]
        y_test = testDataSet[:, -1:]
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(x_train,y_train.ravel())
        y_predicted = qda.predict(x_test)
        print(accuracy_score(y_test,y_predicted))

