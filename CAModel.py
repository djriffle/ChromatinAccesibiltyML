from keras.models import Sequential
from keras.layers import Dense

class CAModel():
    """
    Chromatin Accesbility Model
    """
    def __init__(self):
        #super simple multi-label model
        self.model = Sequential()
        self.model.add(Dense(20, input_dim=1, kernel_initializer='he_uniform', activation='relu'))
        self.model.add(Dense(6, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
    
    def getMode(self):
        return self.model
    