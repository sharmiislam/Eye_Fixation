# scripts/main.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('/Users/sharmiislam/Documents/EYE_Movement/eye_fixation/language_proficiency/src')
from models.regression import RegressionEstimators 
import pandas as pd

# Load data
eye_fixation_demo_data = pd.read_csv("data/lpp_all_fix_demo.csv")
fixation_data = pd.read_csv("data/lpp_all_fix.csv")
eye_fixation_demo_data.columns = eye_fixation_demo_data.columns.str.strip()


print (eye_fixation_demo_data.head())
print(list(eye_fixation_demo_data.columns))

# One-hot encode categorical columns
eye_fixation_demo_data = pd.get_dummies(eye_fixation_demo_data, columns=['Word', 'Sentence', 'Language', 'SubjectID'])
#fixation_data = pd.get_dummies(fixation_data, columns=['Word', 'Sentence', 'Language', 'SubjectID'])
def labelEncoder_df(df, features):
    for feature in features:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])

# List of features to encode
features_to_encode = ['Word', 'Sentence', 'Language', 'SubjectID']

# Applying label encoding to the eye_fixation_demo DataFrame
labelEncoder_df(eye_fixation_demo_data, features_to_encode)

# Split data into features and target
X_eye_fixation_demo = eye_fixation_demo_data.drop(columns=["Target_Ave", "Target_Label"])
y_eye_fixation_demo = eye_fixation_demo_data[["Target_Ave", "Target_Label"]]

X_fixation = fixation_data.drop(columns=["Target_Ave", "Target_Label"])
y_fixation = fixation_data[["Target_Ave", "Target_Label"]]

# Split data into train and test sets
X_eye_fixation_demo_train, X_eye_fixation_demo_test, y_eye_fixation_demo_train, y_eye_fixation_demo_test = \
    train_test_split(X_eye_fixation_demo, y_eye_fixation_demo, test_size=0.2, random_state=42)

X_fixation_train, X_fixation_test, y_fixation_train, y_fixation_test = \
    train_test_split(X_fixation, y_fixation, test_size=0.2, random_state=42)

# Instantiate estimators
linear_reg = LinearRegression() 
params = {}  # Hyperparameters (if any)

eye_fixation_demo_estimator = RegressionEstimators(linear_reg, params)
fixation_estimator = RegressionEstimators(linear_reg, params)

# Tune hyperparameters and train/test estimators
eye_fixation_demo_rmse = eye_fixation_demo_estimator.train_test_estimator(X_eye_fixation_demo_train, X_eye_fixation_demo_test,
                                                                y_eye_fixation_demo_train, y_eye_fixation_demo_test)

fixation_rmse = fixation_estimator.train_test_estimator(X_fixation_train, X_fixation_test,
                                                        y_fixation_train, y_fixation_test)

print("RMSE for Eye Fixation Data:", eye_fixation_demo_rmse)
print("RMSE for Combined Data:", fixation_rmse)
