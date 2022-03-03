from CAModel import CAModel
from sklearn.metrics import accuracy_score

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

        score = accuracy_score(validation_data_results,self.label_eval)

        print("score:", score)

        return model