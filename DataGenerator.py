import numpy as np
import keras
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    #generates data without creating RAM error
    def __init__(self, data, label, shuffle=True, fit=True):
        self.data = data
        self.label = label
        #add reverse complement
        comp_seqs = []
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N' : 'N'}
        for seq in self.data:
            comp_seq = ''
            for base in seq:
                comp_seq += complement[base]
            comp_seqs.append(comp_seq)
        
        np_comp_seqs = np.array(comp_seqs, dtype=object)
        self.data = np.append(self.data, np_comp_seqs, axis=0)
        self.label = np.append(self.label, self.label, axis=0)
        
        self.shuffle = shuffle
        self.on_epoch_end()

        self.fit = fit
        
    def __len__(self):
        #tells the number of batches in each epoch
        return self.data.shape[0]
    
    def __getitem__(self, index):
        #generate one batch of data
        #get indexes of the batch
        label = self.label[index]
        seq = self.data[index]

        row_index = 0
        feature = np.zeros((len(seq),4))
        for base in seq:
            if base == 'A':
                feature[row_index, 0] = 1
            elif base == 'T':
                feature[row_index, 1] = 1
            elif base == 'G':
                feature[row_index, 2] = 1
            elif base == 'C':
                feature[row_index, 3] = 1
            row_index += 1
        
        feature = np.array([feature], dtype='float32')

        X = np.transpose([feature]).astype(np.float32)
        y = np.array([label]).astype(np.float32)
        print(X)
        print(y)

        if(self,self.fit):
            return X,y
        else:
            return X
         
    def on_epoch_end(self):
        pass
        #update the indexes after each epoch
        self.indexes = np.arange(self.data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
