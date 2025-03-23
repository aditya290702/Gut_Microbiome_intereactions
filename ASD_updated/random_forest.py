import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
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

# Print cross-validation and test accuracy
print(f"Cross-validation accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print(f"Test set accuracy: {test_accuracy:.4f}")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# Handle classification SHAP values
if isinstance(shap_values, list) and len(shap_values) > 1:
    shap_values = shap_values[1]  # Selecting positive class

# Convert to NumPy array if necessary
shap_values = np.array(shap_values)

print(f"SHAP values shape: {shap_values.shape}")
print(f"X_train shape: {X_train.shape}")

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
print("\nTop 20 most important features:")
print(feature_importance_df.head(20))

# Extract unique genus names and keep only the top feature for each genus
unique_genus_features = {}
for index, row in feature_importance_df.iterrows():
    genus = row['Feature'].split(';')[0]  # Get the genus part
    if genus not in unique_genus_features:
        unique_genus_features[genus] = row  # Store the entire row for the first occurrence

# Convert the dictionary to a DataFrame
unique_genus_df = pd.DataFrame(list(unique_genus_features.values()))

# Sort by SHAP Importance to get the top 10 unique features
top_unique_features = unique_genus_df.sort_values(by="SHAP Importance", ascending=False).head(10)

# Print the top 10 unique features
print("\nTop 10 unique features:")
print(top_unique_features)

# Plot top 10 unique features
plt.figure(figsize=(10, 6))
plt.barh(top_unique_features['Feature'], top_unique_features['SHAP Importance'], color='royalblue')
plt.xlabel("SHAP Importance")
plt.ylabel("Feature")
plt.title("Top 10 Unique Features")
plt.gca().invert_yaxis()
plt.show()

# # SHAP summary plot
# shap.summary_plot(shap_values, X_train, show=True)