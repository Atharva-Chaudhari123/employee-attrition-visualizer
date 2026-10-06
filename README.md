# 💼 Employee Attrition & Retention Intelligence (DMV Project)

An end-to-end **Data Modeling and Visualization (DMV)** project built with **Python**, **Plotly**, **Scikit-Learn**, and **Streamlit**.

---

## 🌟 DMV Highlights & Key Features

### 1. 📐 Dimensional Data Modeling (Star Schema & OLAP)
- **Conceptual Star Schema Architecture**:
  - **Fact Table**: `Fact_Attrition` (Metrics: satisfaction score, performance evaluation, average monthly hours, employment status).
  - **Dimension Tables**: `Dim_Department`, `Dim_Employee` (Demographics), `Dim_JobProfile` (Compensation band, tenure bucket, promotions).
- **OLAP Analytical Operations**: Dynamic **Slice**, **Dice**, **Drill-down**, and **Roll-up** operations across department, salary tiers, and experience levels.

### 2. 📊 High-Impact Interactive Visual Analytics (Plotly)
- **Workforce Attrition Flow (Sankey Diagram)**: Dynamic ribbons displaying employee flow through `Department` ➔ `Salary Level` ➔ `Stayed vs Left`.
- **Burnout & Flight-Risk Quadrant**: Interactive multi-attribute scatter plot mapping monthly hours vs evaluation scores, highlighting the *"Overworked Top Talent"* departure cluster.
- **Hierarchical Drill-Down (Sunburst Chart)**: Multi-level breakdown from department down to salary bands and final status with smooth zoom transitions.
- **Multi-Attribute Radar / Spider Chart**: Normalized comparison of key attributes between departing and retained employees.
- **Dynamic Correlation Matrix & Distribution Heatmap**.

### 3. 🔮 Real-Time Flight-Risk Predictor & "What-If" Retention Simulator
- **Radial Speedometer / Risk Gauge**: Visual flight-risk probability gauge with colored safety/caution/danger zones.
- **Risk Pressure Breakdown**: Identifies the primary push factors driving attrition risk.
- **Interactive "What-If" Countermeasure Simulator**: Allows HR managers to simulate interventions (e.g. reduce hours, increase salary band, rebalance projects) and watch the predicted risk gauge drop in real-time.

---

## 📁 Project Structure

- `app.py`: Streamlit multi-tab analytical web application.
- `Employee_Attrition_Analysis.ipynb`: Complete data science and modeling pipeline.
- `df_merged_dept_emp_det.csv`: Primary workforce dataset (13,234 records).
- `models/`:
  - `best_rf_model.pkl`: Tuned Random Forest classification model.
  - `scaler.pkl`: StandardScaler transformation object.
  - `feature_names.pkl`: Trained feature schema.
- `requirements.txt`: Python package requirements.

---

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Developer-Jivan/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Interactive Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack
- **Dashboard & UI**: Streamlit, Custom CSS
- **Interactive Visualizations**: Plotly (Express & Graph Objects)
- **Machine Learning**: Scikit-Learn (Random Forest, GridSearchCV, StandardScaler)
- **Data Engineering & OLAP**: Pandas, NumPy
