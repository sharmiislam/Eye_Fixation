# train/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
file_path =('/Users/sharmiislam/Documents/EYE_Movement/eye_fixation/language_proficiency/data/lpp_all_fix_demo.csv')
def preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # List of categorical columns to encode
    columns_to_encode = ['Language', 'Word', 'Sentence', 'SubjectID']

    # Apply one-hot encoding to all categorical columns and return the result
    return pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

