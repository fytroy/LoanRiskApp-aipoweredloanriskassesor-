 🏦 AI-Powered Loan Risk Assessment App

![App Screenshot]([https://via.placeholder.com/800x400?text=App+Screenshot+Placeholder](https://github.com/fytroy/LoanRiskApp-aipoweredloanriskassesor-/blob/main/App_Screenshot.png))
(Replace this link with your actual screenshot from the browser)

 📖 Overview
This project is a Predictive Analytics application designed to help loan officers make data-driven decisions. 

It uses a Random Forest Classifier trained on historical loan data to predict the probability of a borrower defaulting. The model is wrapped in an interactive Streamlit web application for real-time predictions.

 🛠️ Tech Stack
 Python: Core logic.
 Scikit-Learn: Machine Learning (Random Forest).
 Pandas: Data manipulation and preprocessing.
 Streamlit: Web application framework.
 Joblib: Model serialization.

 🤖 The Machine Learning Model
 Algorithm: Random Forest Classifier (n_estimators=100).
 Accuracy: ~93% on test data.
 Features Used: Age, Income, Home Ownership, Employment Length, Loan Intent, Loan Grade, Amount, Interest Rate, Default History.
 Preprocessing: Missing value imputation and Label Encoding for categorical variables.

 🚀 How to Run Locally
1.  Clone the Repo:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/loan-risk-app.git](https://github.com/YOUR_USERNAME/loan-risk-app.git)
    ```
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the App:
    ```bash
    streamlit run app.py
    ```
