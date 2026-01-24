import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
df = pd.read_csv('data/loan.csv')
# Drop the Loan_ID column as it is not useful for analysis
df.drop("Loan_ID", axis=1, inplace=True)
# Handle missing values
# Fill categorical columns with mode and numerical columns with median
cat_cols = ['Gender', 'Married','Dependents', 'Self_Employed']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
#why median? because it is less affected by outliers
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
# print(df.head())
# print(df.isnull().sum())
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('Loan_Status', axis=1)
y = df_encoded['Loan_Status']
# print("Features shape:", X.shape)
# print("Target shape:", y.shape)
# print("Features columns:", X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#print(X_train.shape, X_test.shape)
# model = LogisticRegression(max_iter=1000 , class_weight='balanced')
# model.fit(X_train, y_train)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

df_test = pd.read_csv('data/test.csv')
df_test.drop("Loan_ID", axis=1, inplace=True)
# Replace spaces with NaN
df_test.replace(" ", pd.NA, inplace=True)

# Fill categorical missing values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

# Fill numerical missing values
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    df_test[col] = df_test[col].fillna(df_test[col].median())

df_test_encoded = pd.get_dummies(df_test, drop_first=True)

# Align columns with training data
df_test_encoded = df_test_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

test_predictions = model.predict(df_test_encoded)

df_test['Predicted_Loan_Status'] = ['Y' if x == 1 else 'N' for x in test_predictions]
df_test[['Predicted_Loan_Status']].to_csv(
    'data/test_predictions.csv',
    index=False
)

joblib.dump(model, "model.pkl")
