import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

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

# Convert feature names to strings
X.columns = X.columns.astype(str)

# Split into Train & Test (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Perform 5-fold cross-validation on training data
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train the model on fold
    rf_model.fit(X_train_fold, y_train_fold)

    # Validate the model
    y_val_pred = rf_model.predict(X_val_fold)
    fold_acc = accuracy_score(y_val_fold, y_val_pred)
    cv_accuracies.append(fold_acc)

# Train final model on entire training set
rf_model.fit(X_train, y_train)

# Test the model on the unseen test set
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display results in Streamlit
st.write(f"### Cross-validation accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
st.write(f"### Test set accuracy: {test_accuracy * 100:.2f} %")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# If classification, select SHAP values for the positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Selecting class 1

# Ensure SHAP values are in correct shape
shap_values = np.array(shap_values)
st.write(f"SHAP values shape: {shap_values.shape}")
st.write(f"X_train shape: {X_train.shape}")

# Compute feature importance
if len(shap_values.shape) == 3:  # If multi-class (n_samples, n_features, n_classes)
    feature_importance = np.abs(shap_values).mean(axis=(0, 2))  # Mean over samples & classes
else:  # If binary classification (n_samples, n_features)
    feature_importance = np.abs(shap_values).mean(axis=0)  # Mean over samples only

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'SHAP Importance': feature_importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="SHAP Importance", ascending=False)

# Print top features
st.write("### Top 10 most important features")
st.dataframe(feature_importance_df.head(10))

st.write("### Top 10 least important features")
st.dataframe(feature_importance_df.tail(10))

# Plot top features
st.write("### Top 10 Most Important Features (Bar Chart)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance_df['Feature'][:10], feature_importance_df['SHAP Importance'][:10], color='royalblue')
ax.set_xlabel("SHAP Importance")
ax.set_ylabel("Feature")
ax.set_title("Top 10 Most Important Features")
ax.invert_yaxis()
st.pyplot(fig)
