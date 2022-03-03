from CAModel import CAModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix
import numpy as np

class Trainer():
    """
    A class used to create our Keras model
    """

    def __init__(self,data_train,label_train,data_eval,label_eval) -> None:
        self.data_train = data_train
        self.label_train = label_train

        self.data_eval = data_eval
        self.label_eval = label_eval

        self.caModel = CAModel()
        return
    
    def train(self):
        """Returns a trained model"""
        model = self.caModel.getMode()
        #there are better techniques to train this 

        #we need to figure out how to turn the nucleotides into numbers for this to work
        model.fit(self.data_train,self.label_train,epochs=100)

        validation_data_results = model.predict(self.data_eval)


        validation_data_results.round()

        y_test_arg=np.argmax(self.label_eval,axis=1)
        Y_pred = np.argmax(model.predict(self.data_eval),axis=1)

        print('Confusion Matrix')
        print(confusion_matrix(y_test_arg, Y_pred))
        
        print('Accuracy')
        print(accuracy_score(y_test_arg,Y_pred))

        return model