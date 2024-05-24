# train/data_split.py
from sklearn.model_selection import train_test_split

def split_data(df):
    # Define the features (X) and target variable (y)
    X = df.drop(columns=['Target_Label'])  # Features are all columns except the target variable
    y = df['Target_Label']  # Target variable is 'Target_Label'

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Target_Label'])
    # Print the shapes of the training and test sets

    return X_train, X_test, y_train, y_test
