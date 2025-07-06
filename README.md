# 💳 Credit Scoring Prediction App

A simple machine learning web app that predicts whether a person is **creditworthy** or **not creditworthy** based on their financial details. The model is trained using the **Random Forest** algorithm and the app is built using **Python** and **Streamlit**.

---

## 📌 What This Project Does

- Takes user input like age, job, credit amount, etc.
- Uses a trained model to predict creditworthiness
- Gives a quick and easy-to-understand result
- Has a clean and simple UI using Streamlit

---

## 📊 Dataset

- **Source:** [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Records:** 1000 instances
- **Attributes:** 20 features + 1 target (`Creditability`)
- **Target Class:**
  - `1` = Creditworthy
  - `0` = Not Creditworthy

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Missing values handled using `SimpleImputer`
   - Numerical features scaled with `StandardScaler`
   - Categorical features encoded using `OneHotEncoder`
   - Column transformer combines both for pipeline

2. **Model**
   - `RandomForestClassifier` used
   - Hyperparameter tuning via `GridSearchCV`
   - Trained with `class_weight='balanced'` to handle label imbalance
   - Accuracy: **> 70%**

3. **Web App (Streamlit)**
   - Predicts creditworthiness from user input
   - Displays prediction with confidence score
   - Offers a separate "View Insights" page to show data visualizations

---

## 🧪 Example Input for Creditworthy Prediction

| Feature                   | Sample Value         |
|---------------------------|----------------------|
| Duration                  | 12                   |
| Credit amount             | 2000                 |
| Age                       | 35                   |
| Employment                | 4<=X<7               |
| Credit history            | no credits/all paid  |
| Checking account status   | >=200                |
| Foreign worker            | yes                  |
| Personal status and sex   | male single          |
| Job                       | skilled              |

Try such combinations in the app to receive a **Creditworthy** or **Not Creditworthy** result.

---

## 📈 EDA Visualizations

Run the `eda.py` script to generate graphs such as:

- Creditability distribution
- Age distribution
- Credit amount histogram
- Correlation heatmap
- Age vs Creditability

```bash
python eda.py
```

---

## 🚀 How to Run the App

1. **Clone the repository:**

```bash
git clone https://github.com/TanishaVerma-08/Credit-Scoring-App.git
cd Credit-Scoring-App
```
2. **Install Required Libraries:**
```bash
pip install -r requirements.txt
```
3. **Start the app:**
```bash
streamlit run app.py
```
4. Open the browser and go to `http://localhost:8501`
