# 🏢 Employee Attrition Prediction System

A Complete End-to-End Data Science project built with **Python**, **Scikit-Learn**, and **Streamlit**.

## 📌 Features
- **Exploratory Data Analysis (EDA)**: Interactive visualizations for correlation, outliers, and feature distribution.
- **Machine Learning**: Comparison of 5 classification models (Random Forest, SVM, KNN, etc.).
- **Tuned Best Model**: Hyperparameter optimization using GridSearchCV for high precision.
- **Predictive Tool**: A Streamlit frontend to predict employee attrition risk in real-time.

## 📁 Project Structure
- `Employee_Attrition_Analysis.ipynb`: Full data science lifecycle (cleaning, EDA, training).
- `app.py`: Streamlit multi-page web application.
- `models/`: Processed model artifacts (`.pkl` files).
- `requirements.txt`: Project dependencies.
- `df_merged_dept_emp_det.csv`: The raw dataset.

## 🚀 How to Run Locally
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction
   ```
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Analysis** (to generate models):
   Run all cells in `Employee_Attrition_Analysis.ipynb`.
4. **Start the App**:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **ML Logic**: Scikit-Learn, Pandas, NumPy
- **Viz**: Matplotlib, Seaborn
- **Serialization**: Joblib
