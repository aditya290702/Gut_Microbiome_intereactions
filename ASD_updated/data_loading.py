import pandas as pd

# Load the dataset
Dataset = pd.read_csv("ASD meta abundance.csv", sep=",")

# Transpose dataset with "Taxonomy" as index
Dataset_T = Dataset.set_index("Taxonomy").T

# Rename the index to "Sample"
Dataset_T.index.name = "Sample"

# Remove columns where all values are 0
Dataset_T_cleaned = Dataset_T.loc[:, (Dataset_T != 0).any(axis=0)]

# Remove columns where "Unclassified" is in the Taxonomy name
Dataset_T_cleaned = Dataset_T_cleaned.loc[:, ~Dataset_T_cleaned.columns.str.contains("Unclassified", case=False)]

# Save cleaned dataset
Dataset_T_cleaned.to_csv("ASD_cleaned_dataset.csv")

# Compute relative abundance
relative_abundance = Dataset_T_cleaned.div(Dataset_T_cleaned.sum(axis=0), axis=1)

# Save results
relative_abundance.to_csv("ASD_relative_abundance.csv")

# Display first few rows
print(relative_abundance.head())
