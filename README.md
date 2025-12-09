# heart_failure_perdiction_ML

#  Heart Failure Prediction Using Machine Learning

This project focuses on predicting the risk of heart failure in patients using machine learning techniques.

It includes complete data preprocessing, exploratory data analysis, feature interpretation, model development, and performance evaluation.

The objective is to build an accurate and dependable prediction model that can support early diagnosis and clinical decision-making.

---

##  Project Overview

Heart failure is a major global health issue, and early prediction can significantly improve patient outcomes.  
This project uses machine learning techniques to classify whether a patient is at risk of death due to heart failure.

The project follows a complete ML workflow, including:

- Exploratory Data Analysis (EDA)
- Outlier detection
- Correlation analysis
- Boxplots for feature distribution
- Data balancing using SMOTE
- Standardization
- Model training (LR, KNN, Random Forest)
- Hyperparameter tuning
- Model evaluation (confusion matrix, classification report)

---

##  Dataset Description

- **Rows:** 5000  
- **Columns:** 13 features  
- **Target Variable:** `DEATH_EVENT` (0 = Alive, 1 = Death)

###  Feature List & Meaning

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| anaemia | Low red blood cells (0 = No, 1 = Yes) |
| creatinine_phosphokinase | Enzyme level indicating muscle/kidney damage |
| diabetes | Diabetes status |
| ejection_fraction | Percentage of blood leaving the heart each beat |
| high_blood_pressure | BP status |
| platelets | Platelet count in blood |
| serum_creatinine | Level indicating kidney function |
| serum_sodium | Sodium level in blood |
| sex | 0 = Female, 1 = Male |
| smoking | 0 = No, 1 = Yes |
| time | Follow-up period |
| DEATH_EVENT | Target variable |

---

##  Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
- Checked distributions of all features  
- Identified outliers  
- Observed class imbalance  
- Visualized feature differences using boxplots  

### 2. Correlation Analysis
- Heatmap used to understand relationships  
- No strong multicollinearity observed  

### 3. Train–Test Split
- 80% training  
- 20% testing  
- Stratified sampling to preserve class distribution  

### 4. Balancing the Training Data (SMOTE)
- Oversampled minority class  
- Avoided data leakage by applying SMOTE **only on training data**

### 5. Standardization
- Used StandardScaler to normalize feature values  
- Prevented models from being biased toward high-value features  

### 6.  Model Training
Trained three ML models:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  

