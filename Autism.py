"""
Extended and Detailed Autism Screening Classification Code
Author: Salman Nawaz Malik
Date: 2025-02-16

Description:
------------
This script performs an extensive classification analysis on an autism screening dataset.
It includes:
1. Data loading and initial exploration (EDA).
2. Data cleaning and preprocessing (handling missing values, outliers, etc.).
3. Exploratory Data Analysis (plots, correlations, distributions).
4. Feature engineering and selection (optional steps included).
5. Splitting the data into training and testing sets.
6. Data scaling (Standard Scaler).
7. Model definition and hyperparameter tuning (Logistic Regression, Random Forest, Decision Tree,
   SVM, Neural Network, KNN).
8. Model evaluation using accuracy, confusion matrix, classification reports, and ROC curves.
9. Visualization of results (confusion matrices, ROC curves).
10. Final model comparison and conclusion.

Note:
-----
- This code is intentionally made extensive to serve as a comprehensive example. 
- Make sure to install the required libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn (if using SMOTE)).

Usage:
------
1. Place your dataset (autism_screening_adults.csv) in the same directory or adjust the path accordingly.
2. Run the script. 
3. Observe the console output for accuracy metrics and the generated plots for further insights.

"""

# --------------------------------------------------------------------------------
# 1. Import Necessary Libraries
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score)

# If you want to handle imbalanced data using SMOTE, uncomment the lines below 
# and ensure you have "imbalanced-learn" installed (pip install imbalanced-learn).
# from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')  # To ignore any warnings that may arise

# --------------------------------------------------------------------------------
# 2. Load and Explore the Dataset
# --------------------------------------------------------------------------------

# Path to your dataset (Adjust if necessary)
DATA_PATH = "autism_screening.csv"

# Load your dataset here
try:
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset '{DATA_PATH}' loaded successfully!\n")
except FileNotFoundError:
    print(f"Error: The file '{DATA_PATH}' was not found. Please check the path.")
    data = pd.DataFrame()  # Empty dataframe as fallback

# Basic info about the dataset
print("----- Dataset Information -----")
print(data.info())
print("\n----- First 5 Rows of the Dataset -----")
print(data.head())
print("\n----- Statistical Summary (Numerical Columns) -----")
print(data.describe())

# Check shape
print(f"\nDataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")

# Check for duplicates
duplicates_count = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates_count}")

# --------------------------------------------------------------------------------
# 3. Data Cleaning and Missing Values Handling
# --------------------------------------------------------------------------------

print("\n----- Missing Values per Column -----")
print(data.isnull().sum())

# Option 1: Drop rows with any missing values (simple approach)
# data = data.dropna()

# Option 2: Fill missing values with mean/median/mode (example)
# For demonstration, let's fill numeric columns with median and categorical with mode
# We will do a more robust approach:

def fill_missing_values(df):
    """
    Fill missing values in the dataframe.
    Numerical columns: fill with median
    Categorical columns: fill with mode
    """
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    return df

data = fill_missing_values(data)

print("\n----- Missing Values After Filling -----")
print(data.isnull().sum())

# --------------------------------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------------

# 4.1. Distribution of Numerical Columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print("\n----- Numerical Columns Distribution -----")
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True, color='green')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# 4.2. Count Plots for Categorical Columns
categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n----- Categorical Columns Count Plots -----")
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=data, palette='Set2')
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4.3. Correlation Heatmap for Numerical Columns
if len(numeric_cols) > 1:
    plt.figure(figsize=(8, 6))
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.show()

# --------------------------------------------------------------------------------
# 5. Outlier Detection (Optional)
# --------------------------------------------------------------------------------

def detect_outliers_boxplot(df, col_name):
    """
    Detect outliers in a single numeric column using the boxplot IQR method.
    Returns indices of outliers.
    """
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_indices = df.index[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]
    return outlier_indices

# Example usage: detect outliers in numeric columns
outlier_indices_all = []
for col in numeric_cols:
    col_outliers = detect_outliers_boxplot(data, col)
    outlier_indices_all.extend(col_outliers)

outlier_indices_all = list(set(outlier_indices_all))  # unique indices
print(f"\nDetected {len(outlier_indices_all)} potential outlier rows across numerical columns.")

# Optionally remove these outliers if desired
# data = data.drop(outlier_indices_all, axis=0)

# --------------------------------------------------------------------------------
# 6. Encoding Categorical Variables
# --------------------------------------------------------------------------------

# Check if 'Class/ASD' is in the dataset
if 'Class/ASD' not in data.columns:
    print("\nWarning: 'Class/ASD' column not found in dataset. "
          "Please check the target column name.")
else:
    print("\nTarget column 'Class/ASD' found. Proceeding with encoding.")

# Let's do label encoding for the target if it's categorical
# (assuming 'Class/ASD' can be 'Yes' / 'No' or something similar)
target_col = 'Class/ASD'

# If the target is already numeric (e.g., 0/1), this step may not be needed
if data[target_col].dtype not in [np.int64, np.float64]:
    le = LabelEncoder()
    data[target_col] = le.fit_transform(data[target_col].values)

# Now let's get dummy variables for other categorical columns (excluding target)
categorical_cols = [col for col in categorical_cols if col != target_col]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# --------------------------------------------------------------------------------
# 7. Define Features (X) and Target (y)
# --------------------------------------------------------------------------------

X = data.drop(target_col, axis=1)
y = data[target_col]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# --------------------------------------------------------------------------------
# 8. Handling Imbalance (Optional SMOTE)
# --------------------------------------------------------------------------------

# If you have imbalanced classes and wish to apply SMOTE:
# smote = SMOTE(random_state=42)
# X, y = smote.fit_resample(X, y)
# print("\nApplied SMOTE to handle class imbalance.")
# print(f"New Feature matrix shape: {X.shape}")
# print(f"New Target vector shape: {y.shape}")

# --------------------------------------------------------------------------------
# 9. Train-Test Split
# --------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y  # ensures class proportions remain consistent
)

print("\n----- Train/Test Split -----")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# --------------------------------------------------------------------------------
# 10. Feature Scaling
# --------------------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------------
# 11. Model Definitions
# --------------------------------------------------------------------------------

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Neural Network (MLP)': MLPClassifier(random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier()
}

# --------------------------------------------------------------------------------
# 12. Hyperparameter Tuning using GridSearchCV
# --------------------------------------------------------------------------------

# Define the parameter grids
lr_param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

dt_param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (200,)],
    'activation': ['relu', 'tanh'],
    'max_iter': [200, 500]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'metric': ['euclidean', 'manhattan']
}

# Dictionary to map models to their respective parameter grids
param_grids = {
    'Logistic Regression': lr_param_grid,
    'Random Forest': rf_param_grid,
    'Decision Tree': dt_param_grid,
    'Support Vector Machine': svm_param_grid,
    'Neural Network (MLP)': nn_param_grid,
    'K-Nearest Neighbors (KNN)': knn_param_grid
}

# --------------------------------------------------------------------------------
# 13. Function to Perform Grid Search and Evaluate Models
# --------------------------------------------------------------------------------

def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    """
    This function performs:
    1. GridSearchCV for hyperparameter tuning.
    2. Uses the best estimator to predict on the test set.
    3. Returns the best model, accuracy, confusion matrix, classification report, 
       and other metrics (precision, recall, f1, ROC AUC).
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_prob = None
    
    # Check if the model has predict_proba attribute for ROC calculations
    if hasattr(best_model, "predict_proba"):
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        # Some models (like SVC with kernel=linear) might not have predict_proba by default
        # We can approximate probabilities or skip
        pass
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # ROC AUC
    roc_auc = None
    if y_pred_prob is not None:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    return {
        'best_model': best_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_auc': roc_auc
    }

# --------------------------------------------------------------------------------
# 14. Train All Models and Store Results
# --------------------------------------------------------------------------------

results = {}

for model_name, model in models.items():
    print(f"\n----- Training {model_name} -----")
    best_model_info = train_and_evaluate_model(
        model,
        param_grids[model_name],
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )
    results[model_name] = best_model_info
    print(f"Best Params: {results[model_name]['best_model']}")
    print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
    print(f"Precision: {results[model_name]['precision']:.4f}")
    print(f"Recall: {results[model_name]['recall']:.4f}")
    print(f"F1-score: {results[model_name]['f1_score']:.4f}")
    if results[model_name]['roc_auc'] is not None:
        print(f"ROC AUC: {results[model_name]['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(results[model_name]['confusion_matrix'])
    print("\nClassification Report:")
    print(results[model_name]['classification_report'])

# --------------------------------------------------------------------------------
# 15. Model Comparison
# --------------------------------------------------------------------------------

# Print all accuracies
print("\n----- Model Comparison (Accuracy) -----")
for model_name, result in results.items():
    print(f"{model_name}: {result['accuracy']:.4f}")

# Identify the best model by accuracy
best_model_name = max(results, key=lambda m: results[m]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
print(f"\nBest Model Based on Accuracy: {best_model_name} with accuracy {best_accuracy:.4f}")

# --------------------------------------------------------------------------------
# 16. Plot Confusion Matrices for All Models
# --------------------------------------------------------------------------------

plt.figure(figsize=(18, 10))
num_models = len(results.keys())
for i, (model_name, result) in enumerate(results.items(), 1):
    plt.subplot(2, 3, i)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} (Accuracy: {result['accuracy']:.2f})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 17. Plot ROC Curves for All Models (where applicable)
# --------------------------------------------------------------------------------

plt.figure(figsize=(10, 8))

for model_name, result in results.items():
    best_model = result['best_model']
    
    # Only plot ROC if we have probabilities
    if hasattr(best_model, "predict_proba"):
        y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    else:
        # If model doesn't have probabilities, skip
        pass

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# --------------------------------------------------------------------------------
# 18. Cross-Validation (Optional Additional Step)
# --------------------------------------------------------------------------------

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross validation (accuracy) on a given model using X, y.
    Returns the mean accuracy across folds.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

print("\n----- Cross Validation Scores -----")
for model_name in models.keys():
    # Use the best found hyperparameters
    best_model = results[model_name]['best_model']
    cv_accuracy = cross_validate_model(best_model, X_train_scaled, y_train, cv=5)
    print(f"{model_name}: CV Accuracy (5-fold) = {cv_accuracy:.4f}")

# --------------------------------------------------------------------------------
# 19. Feature Importance (Random Forest Example)
# --------------------------------------------------------------------------------

# We can check feature importance for tree-based models, e.g., RandomForest
if isinstance(results['Random Forest']['best_model'], RandomForestClassifier):
    rf_model = results['Random Forest']['best_model']
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n----- Random Forest Feature Importance -----")
    print(feature_importance_df.head(10))

    # Plot top 10 features
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature', palette='viridis')
    plt.title("Top 10 Important Features (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

# --------------------------------------------------------------------------------
# 20. Final Conclusion
# --------------------------------------------------------------------------------
os.makedirs("Autism_Results", exist_ok=True)
print(f"\n----- Final Conclusion -----")
print(f"The best model based on test accuracy is: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print("Below are the details of the best model:")
print(results[best_model_name]['best_model'])

print("\nEnd of Extended Code.")
