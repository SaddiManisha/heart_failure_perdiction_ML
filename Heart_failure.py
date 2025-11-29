import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# Load the dataset from kaggle

# Download latest version
path = kagglehub.dataset_download("aadarshvelu/heart-failure-prediction-clinical-records")

print("Path to dataset files:", path)

# Load the data
hr_df = pd.read_csv(path + "/heart_failure_clinical_records.csv")

# Print the dataframe
hr_df

# EDA (Exploratory data analysis)

# Checking dataset information
print(hr_df.info())

# Checking Statistics values
hr_df.describe().T

# checking missing values
print(hr_df.isnull().sum())

# checking target feature count
print(hr_df['DEATH_EVENT'].value_counts())

# Plot the target feature count
sns.countplot(x=hr_df['DEATH_EVENT'], palette="Set2")
plt.title("Target Distribution (DEATH_EVENT)")
plt.xlabel("Class (0 = Alive, 1 = Death)")
plt.ylabel("Count")
plt.show()

numeric_cols = hr_df.drop("DEATH_EVENT", axis=1).columns

for col in numeric_cols:
    plt.figure(figsize=(6,4))

    sns.histplot(
        data=hr_df,
        x=col,
        hue="DEATH_EVENT",
        bins=20,
        multiple="dodge",
        shrink=0.7,
        palette=["skyblue", "salmon"]
    )

    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Final 7 continuous features
boxplot_features = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time"
]

# Boxplots for these features
for col in boxplot_features:
    plt.figure(figsize=(5,4))
    sns.boxplot(y=hr_df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10,8))
corr = hr_df.corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Heatmap of Heart Failure Dataset")
plt.tight_layout()
plt.show()