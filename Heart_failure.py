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

# Preprocessing

# Separate independent and dependant features
X = hr_df.drop("DEATH_EVENT", axis=1)
y = hr_df["DEATH_EVENT"]

print("Data separated: Features (X) and Target (y)")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Data split completed.")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# train data - Target distribution
print("Target distribution:\n", y_train.value_counts())

# test data - Target distribution
print("Target distribution:\n", y_test.value_counts())

# Balance ONLY the training data using SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print("Class distribution AFTER SMOTE:")
print(y_train_bal.value_counts())


# Standardization (Scaling the data)
scaler = StandardScaler()

# Fit on training data and transform both
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print("Data standardized using StandardScaler")

# Print full data after scaling
X_train_scaled

# checking first row data
X_train_scaled[3]

# Model implementation

## Model Evalution function

def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Predict
    y_pred = model.predict(X_test)

    # Print classification report
    print(f"\n {model_name} - Classification Report")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Survived (0)", "Death (1)"]
    )

    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

## 1. Logistic Regression

# Train Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train_bal)

print(lr.coef_)
print(lr.intercept_)

evaluate_model(lr,X_test_scaled, y_test, "Logistic Regression")

## 2. KNN

# Train K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_bal)

evaluate_model(knn,X_test_scaled, y_test, "KNN")


 
# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train_bal)

evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# Hyperparameter tuning


## 1. Logistic Regression Tuning

# Logistic Regression Tuning
param_grid_lr = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=1000),
                       param_grid_lr,
                       cv=3,
                       scoring="recall")

grid_lr.fit(X_train_scaled, y_train_bal)

print("Best Logistic Regression Params:", grid_lr.best_params_)

best_lr = grid_lr.best_estimator_
evaluate_model(best_lr, X_test_scaled, y_test, "Logistic Regression (Tuned)")

## 2. KNN Tuning
# KNN Tuning
param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

grid_knn = GridSearchCV(KNeighborsClassifier(),
                        param_grid_knn,
                        cv=3,
                        scoring="recall")

grid_knn.fit(X_train_scaled, y_train_bal)

print("Best KNN Params:", grid_knn.best_params_)

best_knn = grid_knn.best_estimator_
evaluate_model(best_knn, X_test_scaled, y_test, "KNN (Tuned)")

## 3. Random Forest Tuning

# Random Forest Tuning
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid_rf,
                       cv=3,
                       scoring="recall")

grid_rf.fit(X_train_scaled, y_train_bal)

print("Best Random Forest Params:", grid_rf.best_params_)

best_rf = grid_rf.best_estimator_
evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest (Tuned)")

