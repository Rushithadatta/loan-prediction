import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib


# Load dataset
data = pd.read_csv('data/loan.csv')

# Drop Loan_ID column
data.drop(['Loan_ID'], axis=1, inplace=True)

# Encode categorical variables
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# Handle missing values
for col in data.columns:
    data[col] = data[col].fillna(data[col].median())

# Split features and target
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

# Logistic Regression model
lr = LogisticRegression()

# -------- Training Performance --------
lr.fit(X_train, Y_train)
Y_train_pred = lr.predict(X_train)

print("Training Accuracy =", 100 * metrics.accuracy_score(Y_train, Y_train_pred))
print("Training Confusion Matrix:\n", metrics.confusion_matrix(Y_train, Y_train_pred))
print("Training Classification Report:\n",
      metrics.classification_report(Y_train, Y_train_pred))

# -------- Testing Performance --------
Y_test_pred = lr.predict(X_test)

print("Testing Accuracy =", 100 * metrics.accuracy_score(Y_test, Y_test_pred))
print("Testing Confusion Matrix:\n", metrics.confusion_matrix(Y_test, Y_test_pred))
print("Testing Classification Report:\n",
      metrics.classification_report(Y_test, Y_test_pred))


joblib.dump(lr, 'model.pkl')