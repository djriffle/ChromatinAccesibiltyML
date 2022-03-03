from torch import neg
from websockets import Data
from DataProcessor import DataProcessor
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
    dataProcessor = DataProcessor(**dylanPaths)
    data,labels,weights = DataProcessor.create_data()
    
    print("Data Shape:", data.shape)
    print("Labels Shape:", labels.shape)
    print("Weights", weights.shape)

    data_train, data_eval, data_test, label_train, \
    label_eval, label_test, test_size = dataProcessor.split_train_test(data, labels)

    print("Double/find complemnets for sequences")
    dataProcessor.find_DNA_complements(data,labels)
    print("Complemnts Done")




    