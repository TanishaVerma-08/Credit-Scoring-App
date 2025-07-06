import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Paths
RAW_DATA_PATH = r"D:\Celebal Technologies\Random Forest Credit Score\Dataset\german.data"
DATA_CSV_PATH = os.path.join("data", "german_credit_data.csv")
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")

# Column names
COLUMNS = [
    'Checking account status', 'Duration', 'Credit history', 'Purpose', 'Credit amount',
    'Savings account/bonds', 'Employment', 'Installment commitment', 'Personal status and sex',
    'Other debtors/guarantors', 'Present residence since', 'Property magnitude',
    'Age', 'Other installment plans', 'Housing', 'Existing credits', 'Job',
    'Number of dependents', 'Owns telephone', 'Foreign worker', 'Creditability'
]

def convert_raw_to_csv():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("❌ RAW german.data file not found.")
    print("🔄 Converting to CSV...")
    os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH, sep=' ', header=None)
    df.columns = COLUMNS
    df.to_csv(DATA_CSV_PATH, index=False)
    print(f"✅ Saved to {DATA_CSV_PATH}")

def load_data():
    if not os.path.exists(DATA_CSV_PATH):
        convert_raw_to_csv()
    return pd.read_csv(DATA_CSV_PATH)

def preprocess_data(df):
    df = df.copy()
    df['Creditability'] = df['Creditability'].map({1: 1, 2: 0})  # 1 = creditworthy

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('Creditability')
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X = df.drop("Creditability", axis=1)
    y = df["Creditability"]

    X_processed = preprocessor.fit_transform(X)
    column_order = num_cols + cat_cols

    return X_processed, y, preprocessor, column_order

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                               n_jobs=-1, verbose=0, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"🎯 Accuracy: {acc * 100:.2f}%\n")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=['Not Creditworthy', 'Creditworthy']))

    return best_model

def save_model(model, preprocessor, column_order):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": model,
        "preprocessor": preprocessor,
        "columns": column_order
    }, MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    df = load_data()
    X, y, preprocessor, column_order = preprocess_data(df)
    model = train_model(X, y)
    save_model(model, preprocessor, column_order)