import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

# Paths
RAW_DATA_PATH = r"D:\\Celebal Technologies\\Random Forest Credit Score\\Dataset\\german.data"
DATA_CSV_PATH = os.path.join("data", "german_credit_data.csv")

# Column headers
COLUMNS = [
    'Checking account status', 'Duration', 'Credit history', 'Purpose', 'Credit amount',
    'Savings account/bonds', 'Employment', 'Installment commitment', 'Personal status and sex',
    'Other debtors/guarantors', 'Present residence since', 'Property magnitude',
    'Age', 'Other installment plans', 'Housing', 'Existing credits', 'Job',
    'Number of dependents', 'Telephone', 'Foreign worker', 'Creditability'
]

# Step 1: Convert to CSV if not already
if not os.path.exists(DATA_CSV_PATH):
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH, sep=' ', header=None)
    df.columns = COLUMNS
    df.to_csv(DATA_CSV_PATH, index=False)
else:
    df = pd.read_csv(DATA_CSV_PATH)

# Step 2: Convert target for plots (1 = Good, 2 = Bad)
df['Creditability'] = df['Creditability'].map({1: "Good", 2: "Bad"})

# Step 3: Create output folder
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

#Plot 1: Creditability distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Creditability', palette="viridis")
plt.title("Creditability Distribution")
plt.savefig(os.path.join(output_dir, "creditability_distribution.png"))
plt.close()

#Plot 2: Credit amount
plt.figure(figsize=(6, 4))
sns.histplot(df['Credit amount'], bins=30, kde=True, color='skyblue')
plt.title("Credit Amount Distribution")
plt.savefig(os.path.join(output_dir, "credit amount_distribution.png"))
plt.close()

#Plot 3: Age
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=30, kde=True, color='orange')
plt.title("Age Distribution")
plt.savefig(os.path.join(output_dir, "age_distribution.png"))
plt.close()

# 📊 Plot 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# 📊 Plot 5: Age vs Creditability
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Creditability', y='Age', palette='Set2')
plt.title("Age vs Creditability")
plt.savefig(os.path.join(output_dir, "age_vs_creditability.png"))
plt.close()

print("EDA visuals saved to /data folder")