import pandas as pd
import os


class DataIngestion:
    def __init__(self, data_path:str):
        self.data_path = data_path

    def ingest_data(self):
        #read dataset into a pandas dataframe
        self.data = pd.read_csv(self.data_path)

        #Make a copy of data
        self.data_copy = self.data.copy()
        return self.data_copy
