from torch import neg
from websockets import Data
from DataLoader import DataLoader
from Trainer import Trainer


#Dylan's personal paths for testing
dylanPaths = {
    'pos_fasta_path': 'data/brain/brain_atac_hg38_final.fa',
    'encode_path': 'data/brain/ENCODE_noBrain.fa',
    'neg_path': 'data/brain/neg_noENCODE_noBrain.fa',
    'label_path': 'data/brain/brain_scATAC_label.csv',
    'cell_cluster': [],
    'subset': False}

if __name__ == "__main__":
    dataLoader = DataLoader(**dylanPaths)
    data = dataLoader.create_data()
    print("Data:", data[0])
    print("Labels", data[1])
    print("weights", data[2])
    pass