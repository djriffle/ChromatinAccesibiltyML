from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Softmax
class CAModel():
    """
    Chromatin Accesbility Model
    """
    def __init__(self):
        #super simple multi-label model
        self.model = Sequential()
        self.model.add(Dense(units=20 ,input_shape=(1000,4)))
        self.model.add(Flatten())
        self.model.add(Dense(units=6, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model.add(Softmax())
    
    def getMode(self):
        return self.model
    