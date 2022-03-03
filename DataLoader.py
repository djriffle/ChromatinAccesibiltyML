import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

class DataLoader():
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
        np.random.seed(0)
        encode_idx = np.random.choice(self.encode_fasta.shape[0], size = self.encode_fasta.shape[0], replace=False)
        self.encode_fasta = self.encode_fasta[encode_idx]

        neg_idx = np.random.choice(self.neg_fasta.shape[0], size = self.neg_fasta.shape[0], replace=False)
        self.neg_fasta = self.neg_fasta[neg_idx]

        data = np.concatenate([self.pos_fasta, self.encode_fasta, self.neg_fasta])
        ##select cluster for label, where cluster is a list of integers to choose out of our total labels
        #label = self.label[:, self.cell_cluster]

        print(self.encode_fasta)

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

        #still have to do split train set, and calculate pos weight, at https://github.com/aicb-ZhangLabs/scEpiLock/blob/master/deep_learning_module/data/pre_processor.py
    
    def split_train_test():
        pass


        
