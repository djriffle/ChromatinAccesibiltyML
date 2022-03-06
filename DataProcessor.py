import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

class DataProcessor():
    """
    A class used to format sequence data for training
    """

    def __init__(self, pos_fasta_path:Path, encode_path: Path, neg_path: Path, label_path: Path, cell_cluster: list, subset: bool) -> None:
        #Todo make this so that it takes in our file inputs and returns them properly formatted in numpy lists
        print("loading pos_fasta")
        self.pos_fasta = self.fasta_to_list(pos_fasta_path)
        print("loading encode_fasta")
        self.encode_fasta = self.fasta_to_list(encode_path)
        print("loading neg_fasta")
        self.neg_fasta = self.fasta_to_list(neg_path)
        print("loading label")
        self.label = self.csv_to_list(label_path)
        self.cell_cluster = cell_cluster
        self.subset = subset # I dont think we are going to use this

    def fasta_to_list(self, file:Path) -> np.ndarray:
        """convert a fasta file into a multi-dimensional numpy array"""
        return (pd.read_csv(file,sep=">chr*",header=None, engine='python').values[1::2][:,0])

    def csv_to_list(self, file:Path) -> np.ndarray:
        """convert a csv file into a multi-dimensional numpy array"""
        return (pd.read_csv(file, delimiter = ",", header=None)).to_numpy()

    def create_data(self):
        """
        Returns our data, labels, and pos_weights
        """
        np.random.seed(0)
        encode_idx = np.random.choice(self.encode_fasta.shape[0], size = self.encode_fasta.shape[0], replace=False)
        self.encode_fasta = self.encode_fasta[encode_idx]

        neg_idx = np.random.choice(self.neg_fasta.shape[0], size = self.neg_fasta.shape[0], replace=False)
        self.neg_fasta = self.neg_fasta[neg_idx]

        data = np.concatenate([self.pos_fasta, self.encode_fasta, self.neg_fasta])
        ##select cluster for label, where cluster is a list of integers to choose out of our total labels
        #label = self.label[:, self.cell_cluster]


        encode_label = np.zeros((self.encode_fasta.shape[0], self.label.shape[1]))
        neg_label = np.zeros((self.neg_fasta.shape[0], self.label.shape[1]))

        label = np.concatenate([self.label, encode_label, neg_label])

        pos_weight = []
        for i in range(0,label.shape[1]):
            num_pos = np.count_nonzero(label[:, i])
            num_neg = label.shape[0] - num_pos
            pos_weight.append(float(num_neg)/num_pos)
        print(pos_weight)

        return (data,label,pos_weight)
    
    def split_train_test(self,data,label,test_size=0.1):
        """divides our data into training, testing, and evalutation data"""
        data_train_temp, data_test, label_train_temp, label_test\
        = train_test_split(data, label, test_size=test_size, random_state=808)

        data_train, data_eval, label_train, label_eval\
        = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=808)
        
        return data_train, data_eval, data_test, label_train, label_eval, label_test, test_size

    #the below methods are no longer needed since they are handled by our data generator
    def find_DNA_complements(self,input_data,input_labels):
        """Appends DNA complements sequences to our data and labels"""
        comp_seqs = []
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N' : 'N'}
        for seq in input_data:
            comp_seq = ''
            for base in seq:
                comp_seq += complement[base]
            comp_seqs.append(comp_seq)
        
        np_comp_seqs = np.array(comp_seqs, dtype=object)
        input_data = np.append(input_data, np_comp_seqs, axis=0)
        input_labels = np.append(input_labels, input_labels, axis=0)
    
    def convert_to_feature_vector(self,input_data):
        '''
        returns data with feature vectors instead of nucleotides
        '''
        new_data = []
        print("total sequences", input_data.shape)
        counter = 0
        for seq in input_data:
            if(counter % 10000 == 0):
                print("on sequence", counter)

            row_index = 0
            feature = np.zeros((len(seq), 4))
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
            new_data.append(feature)
            counter += 1
            
        return np.array(new_data, dtype=object).astype('float32')
        


        


        
