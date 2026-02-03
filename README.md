# 🏦 Loan Approval Prediction System

This project is a Machine Learning–based system that predicts whether a loan application will be **approved or rejected** based on applicant and financial details.  
The model is trained on real-world banking data and deployed as an interactive web application using **Streamlit**.

---

## 📌 Problem Statement
Banks receive a large number of loan applications daily. Manually evaluating each application is time-consuming and prone to inconsistency.  
The goal of this project is to build a **predictive model** that assists in loan approval decisions using historical data.

---

## 📊 Dataset
- Source: Kaggle Loan Prediction Dataset
- Records: 614 loan applications
- Target Variable: `Loan_Status`
  - `1` → Loan Approved
  - `0` → Loan Rejected

### Features include:
- Applicant & Coapplicant Income
- Loan Amount & Loan Term
- Credit History
- Gender, Marital Status, Education
- Employment Type
- Property Area

---

## 🧹 Data Preprocessing
The following preprocessing steps were performed:

- Dropped non-informative column (`Loan_ID`)
- Handled missing values:
  - **Categorical features** → filled using **mode**
  - **Numerical features** → filled using **median** (robust to outliers)
- Converted target variable (`Loan_Status`) to binary values
- Applied **one-hot encoding** to categorical variables
- Ensured consistent feature alignment for training and prediction

---

## 🧠 Model Used
### Decision Tree Classifier
- Chosen for:
  - Interpretability
  - Ability to handle non-linear decision boundaries
  - Suitability for financial decision systems
- No feature scaling required

---

## 📈 Model Evaluation
The model was evaluated using a train–test split (80% training, 20% testing).

### Results:
- **Accuracy:** ~78.9%
- **Confusion Matrix Analysis:**
  - Model performs well in identifying eligible loan applicants
  - Some false positives exist, which is a known trade-off in credit risk modeling

> Note: Extremely high accuracy is unrealistic for real-world financial datasets and may indicate overfitting.

---

## 🌐 Deployment
The trained model is deployed as a web application using **Streamlit**.

### Application Features:
- User-friendly form to input loan applicant details
- Real-time loan approval prediction
- Clean and minimal UI

---

## 🚀 How to Run the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/Rushithadatta/loan-prediction
cd loan_prediction

```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app
```bash
python -m streamlit run app.py

