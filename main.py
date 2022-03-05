from DataProcessor import DataProcessor
from DataGenerator import DataGenerator
from Trainer import Trainer


#Dylan's personal paths for testing
dylanPaths = {
    'pos_fasta_path': 'data/brain/small_brain.fa',
    'encode_path': 'data/brain/small_encode.fa',
    'neg_path': 'data/brain/small_neg.fa',
    'label_path': 'data/brain/small_brain_label.csv',
    'cell_cluster': [],
    'subset': False}

#Udayan's personal paths for testing
udayanPaths = {
    'pos_fasta_path': 'data/brain/brain_atac_hg38_final.fa',
    'encode_path': 'data/brain/ENCODE_noBrain.fa',
    'neg_path': 'data/brain/neg_noENCODE_noBrain.fa',
    'label_path': 'data/brain/brain_scATAC_label.csv',
    'cell_cluster': [],
    'subset': False}

if __name__ == "__main__":
    dataProcessor = DataProcessor(**dylanPaths)

    data,labels,weights = dataProcessor.create_data()
    
    print("Data Shape:", data.shape)
    print("Labels Shape:", labels.shape)
    print("Weights Length:", len(weights))

    data_train, data_eval, data_test, label_train, \
    label_eval, label_test, test_size = dataProcessor.split_train_test(data, labels)

    gen_train = DataGenerator(data_train,label_train)
    trainer = Trainer(gen_train, data_eval, label_eval)

    print("training")
    trainer.train()




    
