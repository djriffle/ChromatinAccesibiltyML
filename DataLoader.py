import pandas
from pathlib import Path
class DataLoader():
    """
    A class used to format sequence data for training
    """

    def __init__(self,) -> None:
        #Todo make this so that it takes in our file inputs and returns them properly formatted in lists
        return None

    def fasta_to_list(file:Path):
        """convert a fasta file into a usuable list"""
        return pandas.read_csv(file,sep=">chr*",header=None, engine='python').values[1::2][:,0]
