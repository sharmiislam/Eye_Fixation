import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv('/Users/sharmiislam/Documents/EYE_Movement/eye_fixation/language_proficiency/data/lpp_all_fix_demo.csv')

    def preprocess(self):
        # Add your preprocessing steps here
        # For instance, filling missing values
        self.dataframe.fillna(0, inplace=True)

    def split(self, test_size=0.2):
        
        train, test = train_test_split(self.dataframe, test_size=test_size)
        self.train = train
        self.test = test

# Usage
dataset = Dataset('data.csv')
dataset.preprocess()
dataset.split(test_size=0.3)

print(dataset.train)
print(dataset.test)