import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
preprocessor = model_data["preprocessor"]
column_order = model_data["columns"]

# Load training data
DATA_CSV_PATH = os.path.join("data", "german_credit_data.csv")
df = pd.read_csv(DATA_CSV_PATH)
df['Creditability'] = df['Creditability'].map({1: 'Good', 0: 'Bad'})

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Credit Scoring App", layout="centered")

st.markdown("<h1 style='text-align: center; color: teal;'>💳 Credit Scoring Dashboard</h1>", unsafe_allow_html=True)

# Navigation Sidebar
page = st.sidebar.radio("🔍 Choose Page", ["🔮 Predict Creditworthiness", "📈 EDA Analysis", "📊 View Insights"])

# -------------------------
# PREDICTION PAGE
# -------------------------
if page == "🔮 Predict Creditworthiness":
    st.subheader("🧾 Enter Client Financial Information")
    col1, col2 = st.columns(2)

    with col1:
        duration = st.slider("Loan Duration (months)", 4, 72, 24)
        credit_amount = st.slider("Credit Amount", 250, 20000, 1500)
        age = st.slider("Age", 18, 75, 35)
        installment_commitment = st.selectbox("Installment Commitment (1-4)", [1, 2, 3, 4])
        existing_credits = st.selectbox("Existing Credits", [1, 2, 3, 4])
        num_dependents = st.selectbox("Number of Dependents", [1, 2])
        checking_status = st.selectbox("Checking Account Status", ['<0', '0<=X<200', '>=200', 'no checking'])
        credit_history = st.selectbox("Credit History", [
            'no credits/all paid', 'all paid', 'existing paid',
            'delayed previously', 'critical/other existing credit'
        ])
        purpose = st.selectbox("Purpose of Loan", [
            'radio/TV', 'education', 'furniture/equipment', 'new car', 'used car',
            'business', 'domestic appliance', 'repairs', 'retraining', 'others'
        ])
        savings_status = st.selectbox("Savings/Bonds Status", [
            '<100', '100<=X<500', '500<=X<1000', '>=1000', 'no known savings'
        ])

    with col2:
        employment = st.selectbox("Employment Duration", [
            'unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'
        ])
        personal_status = st.selectbox("Personal Status and Sex", [
            'male single', 'female div/dep/mar', 'male div/sep', 'male mar/wid'
        ])
        residence_since = st.slider("Years at Present Residence", 1, 4, 2)
        property_magnitude = st.selectbox("Property Type", [
            'real estate', 'life insurance', 'car', 'no known property'
        ])
        housing = st.selectbox("Housing Type", ['own', 'for free', 'rent'])
        job = st.selectbox("Job Type", [
            'unemp/unskilled non res', 'unskilled resident', 'skilled', 'highly qualified'
        ])
        other_parties = st.selectbox("Other Debtors/Guarantors", ['none', 'guarantor', 'co applicant'])
        other_payment_plans = st.selectbox("Other Installment Plans", ['none', 'bank', 'stores'])
        own_telephone = st.selectbox("Owns Telephone?", ['yes', 'no'])
        foreign_worker = st.selectbox("Is Foreign Worker?", ['yes', 'no'])

    user_data = {
        'Duration': duration,
        'Credit amount': credit_amount,
        'Age': age,
        'Installment commitment': installment_commitment,
        'Existing credits': existing_credits,
        'Number of dependents': num_dependents,
        'Checking account status': checking_status,
        'Credit history': credit_history,
        'Purpose': purpose,
        'Savings account/bonds': savings_status,
        'Employment': employment,
        'Personal status and sex': personal_status,
        'Property magnitude': property_magnitude,
        'Housing': housing,
        'Job': job,
        'Other debtors/guarantors': other_parties,
        'Other installment plans': other_payment_plans,
        'Owns telephone': own_telephone,
        'Foreign worker': foreign_worker,
        'Present residence since': residence_since,
    }

    input_df = pd.DataFrame([user_data])
    missing_cols = set(column_order) - set(input_df.columns)
    if missing_cols:
        st.error(f"❌ Missing input fields: {missing_cols}")
        st.stop()

    input_df = input_df[column_order]
    st.markdown("### 🔍 Preview Input Data")
    st.dataframe(input_df)

    if st.button("🔮 Predict Creditworthiness"):
        try:
            X_processed = preprocessor.transform(input_df)
            prediction = model.predict(X_processed)
            proba = model.predict_proba(X_processed)[0][prediction[0]]

            result = "✅ Creditworthy" if prediction[0] == 1 else "❌ Not Creditworthy"
            color = "green" if prediction[0] == 1 else "red"

            st.markdown(f"<h2 style='text-align: center; color:{color};'>{result}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Confidence: <strong>{proba * 100:.2f}%</strong></p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# EDA ANALYSIS PAGE
# -------------------------------
elif page == "📈 EDA Analysis":
    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("#### 🔹 Creditworthiness Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Creditability', data=df, ax=ax1)
    st.pyplot(fig1)

    st.markdown("#### 🔹 Age Distribution by Creditworthiness")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Creditability', y='Age', data=df, ax=ax2)
    st.pyplot(fig2)

    st.markdown("#### 🔹 Credit Amount Histogram")
    fig3, ax3 = plt.subplots()
    sns.histplot(data=df, x='Credit amount', hue='Creditability', bins=30, kde=True, ax=ax3)
    st.pyplot(fig3)

    st.info("These insights can help identify trends in credit approval based on age and loan amount.")

# -------------------------------
# INSIGHTS PAGE
# -------------------------------
elif page == "📊 View Insights":
    st.subheader("📈 Training Data Insights")

    eda_path = "data"
    plots = [
        "creditability_distribution.png",
        "credit amount_distribution.png",
        "age_distribution.png",
        "correlation_heatmap.png",
        "age_vs_creditability.png"
    ]

    for img in plots:
        full_path = os.path.join(eda_path, img)
        if os.path.exists(full_path):
            st.image(full_path, use_column_width=True)
        else:
            st.warning(f"⚠️ Missing: {img}. Run eda.py to generate.")