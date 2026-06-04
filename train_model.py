# Install Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load and Explore Dataset
data = {
    'StudyHours':[2,5,8,1,6,7,3,9,4,5],
    'Attendance':[60,80,90,50,85,95,70,98,75,82],
    'PreviousScore':[45,65,80,35,70,85,55,90,60,75],
    'Result':['Fail','Pass','Pass','Fail','Pass','Pass','Fail','Pass','Pass','Pass']
}

df = pd.DataFrame(data)

# Preprocess Data
encoder = LabelEncoder()
df['Result'] = encoder.fit_transform(df['Result'])

X = df[['StudyHours','Attendance','PreviousScore']]
y = df['Result']

# Train Machine Learning Model
model = LogisticRegression()

model.fit(X, y)

joblib.dump(model, "student_model.pkl")

print("Model Saved Successfully")