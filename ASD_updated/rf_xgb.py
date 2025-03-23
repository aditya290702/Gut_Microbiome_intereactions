import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the relative abundance data
abundance_df = pd.read_csv("ASD_relative_abundance_with_labels.csv")

# Convert column names to lowercase
abundance_df.columns = abundance_df.columns.str.lower()

# Ensure 'asd' column exists
if "asd" not in abundance_df.columns:
    raise ValueError("Missing 'ASD' column for classification.")

# Extract features (X) and labels (y)
X = abundance_df.drop(columns=["sample", "asd"])  # Remove sample & label
y = abundance_df["asd"]  # Target (ASD vs. Non-ASD)

# Convert feature names to strings and clean them
X.columns = X.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True)

# Split into Train & Test (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

# Perform 5-fold cross-validation for RandomForest
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_accuracies, xgb_cv_accuracies = [], []

for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train RandomForest on fold
    rf_model.fit(X_train_fold, y_train_fold)
    y_val_pred_rf = rf_model.predict(X_val_fold)
    rf_cv_accuracies.append(accuracy_score(y_val_fold, y_val_pred_rf))

    # Train XGBoost on fold
    xgb_model.fit(X_train_fold, y_train_fold)
    y_val_pred_xgb = xgb_model.predict(X_val_fold)
    xgb_cv_accuracies.append(accuracy_score(y_val_fold, y_val_pred_xgb))

# Train final models on entire training set
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Test the models on the unseen test set
y_test_pred_rf = rf_model.predict(X_test)
y_test_pred_xgb = xgb_model.predict(X_test)

rf_test_accuracy = accuracy_score(y_test, y_test_pred_rf)
xgb_test_accuracy = accuracy_score(y_test, y_test_pred_xgb)

# Print results
print(f"RandomForest Cross-validation accuracy: {np.mean(rf_cv_accuracies):.4f} ± {np.std(rf_cv_accuracies):.4f}")
print(f"RandomForest Test set accuracy: {rf_test_accuracy:.4f}")

print(f"XGBoost Cross-validation accuracy: {np.mean(xgb_cv_accuracies):.4f} ± {np.std(xgb_cv_accuracies):.4f}")
print(f"XGBoost Test set accuracy: {xgb_test_accuracy:.4f}")

# SHAP Analysis for RandomForest
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_train)

# If classification, select SHAP values for the positive class (index 1)
if isinstance(rf_shap_values, list):
    rf_shap_values = rf_shap_values[1]  # Get SHAP values for the positive class

# Check the shape of SHAP values
print("Shape of SHAP values for Random Forest:", rf_shap_values.shape)

# Compute feature importance by averaging SHAP values across samples
# This will give us a 1D array of feature importance
rf_feature_importance = np.abs(rf_shap_values).mean(axis=0)

# Create DataFrame for feature importance
rf_feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Importance": rf_feature_importance
}).sort_values(by="SHAP Importance", ascending=False)

# Check the DataFrame
print("Random Forest Feature Importance DataFrame:")
print(rf_feature_importance_df.head())

# Plot SHAP values for Random Forest
shap.summary_plot(rf_shap_values, X_train, plot_type="bar", feature_names=X.columns)

# SHAP Analysis for XGBoost
xgb_explainer = shap.Explainer(xgb_model)
xgb_shap_values = xgb_explainer(X_train)

# Check the shape of SHAP values for XGBoost
print("Shape of SHAP values for XGBoost:", xgb_shap_values.shape)

# Compute feature importance for XGBoost
xgb_feature_importance = np.abs(xgb_shap_values.values).mean(axis=0)

# Create DataFrame for feature importance
xgb_feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Importance": xgb_feature_importance
}).sort_values(by="SHAP Importance", ascending=False)

# Check the DataFrame
print("XGBoost Feature Importance DataFrame:")
print(xgb_feature_importance_df.head())

# Plot SHAP values for XGBoost
shap.summary_plot(xgb_shap_values.values, X_train, plot_type="bar", feature_names=X.columns)