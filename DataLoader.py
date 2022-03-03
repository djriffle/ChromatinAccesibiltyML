import pandas as pd
import numpy as np
from pathlib import Path
class DataLoader():
    """
    A class used to format sequence data for training
    """

    def __init__(self, pos_fasta_path:Path, encode_path: Path, neg_path: Path, label_path: Path, cell_cluster: list, subset: bool) -> None:
        #Todo make this so that it takes in our file inputs and returns them properly formatted in numpy lists
        self.pos_fasta = self.fasta_to_list(pos_fasta_path)
        self.encode_fasta = self.fasta_to_list(encode_path)
        self.neg_fasta = self.fasta_to_list(neg_path)
        self.label = self.csv_to_list(label_path)

    def fasta_to_list(file:Path):
        """convert a fasta file into a usuable list"""
        return (pd.read_csv(file,sep=">chr*",header=None, engine='python').values[1::2][:,0]).to_numpy()

    def csv_to_list(file:Path):
        return (pd.read_csv(self.label_path, delimiter = ",", header=None)).to_numpy()

    def create_data(self):
        np.random.seed(0)
        encode_idx = np.random.choice(self.encode_fasta.shape[0], size = self.encode_fasta.shape[0], replace=False)
        self.encode_fasta = self.encode_fasta[encode_idx]

        neg_idx = np.random.choice(self.neg_fasta.shape[0], size = self.neg_fasta.shape[0], replace=False)
        self.neg_fasta = self.neg_fasta[neg_idx]

        data = np.concatenate([self.pos_fasta, self.encode_fasta, self.neg_fasta])
        ##select cluster for label, where cluster is a list of integers to choose out of our total labels
        label = label[:, self.cell_cluster]

        encode_label = np.zeros((self.encode_fasta.shape[0], len(self.cell_cluster)))
        neg_label = np.zeros((self.neg_fasta.shape[0], len(self.cell_cluster)))

        label = concatenate([label, encode_label, neg_label])

        #still have to do split train set, and calculate pos weight, at https://github.com/aicb-ZhangLabs/scEpiLock/blob/master/deep_learning_module/data/pre_processor.py
        

        
