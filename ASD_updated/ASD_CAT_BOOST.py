import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
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

# Define the CatBoost model
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42)

# Perform 5-fold cross-validation on training data
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train the model on fold
    catboost_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, verbose=0)

    # Validate the model
    y_val_pred = catboost_model.predict(X_val_fold)
    fold_acc = accuracy_score(y_val_fold, y_val_pred)
    cv_accuracies.append(fold_acc)

# Train final model on entire training set
catboost_model.fit(X_train, y_train, verbose=0)

# Test the model on the unseen test set
y_test_pred = catboost_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print(f"Cross-validation accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print(f"Test set accuracy: {test_accuracy:.4f}")

# Get feature importance from CatBoost
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': catboost_model.get_feature_importance()
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Print top features
print("\nTop 10 most important features:\n", feature_importance_df.head(10))
print("\nTop 10 least important features:\n", feature_importance_df.tail(10))

# Show top features in a bar plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color='royalblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Most Important Features (CatBoost)")
plt.gca().invert_yaxis()
plt.show()


# Initialize SHAP explainer
explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(X_train)

# Convert to numpy array if necessary
shap_values = np.array(shap_values)

# Ensure correct shape
print(f"SHAP values shape: {shap_values.shape}")  # Should be (n_samples, n_features)

# Beeswarm Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train)
plt.show()
