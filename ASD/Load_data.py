import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
Dataset = pd.read_csv("ASD meta abundance.csv", sep=",")
print(Dataset)

# # Transpose dataset with "Microbe" as index
Dataset_T = Dataset.set_index("Taxonomy").T

# # Rename the index to "Sample"
Dataset_T.index.name = "Sample"
print(Dataset_T)

# Remove columns where all values are 0
Dataset_T_cleaned = Dataset_T.loc[:, (Dataset_T != 0).any(axis=0)]

# Save cleaned dataset
Dataset_T_cleaned.to_csv("ASD_cleaned_dataset.csv")

# Print results (optional)
print(Dataset_T_cleaned.head())

# Generate pie chart for each sample
# Generate bar plot for each sample
for sample in Dataset_T_cleaned.index:
    data = Dataset_T_cleaned.loc[sample]

    # Keep only non-zero values
    data = data[data > 0].sort_values(ascending=True)  # Sort for better visualization

    if not data.empty:  # Avoid empty plots
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(data))

        plt.barh(data.index, data.values, color=colors)
        plt.xlabel("Abundance", fontsize=8)
        plt.ylabel("Microbe", fontsize=8)
        plt.title(f'Microbial Abundance in {sample}', fontsize=14, fontweight='bold')

        plt.grid(axis='x', linestyle='--', alpha=1)  # Gridlines for readability
        plt.tight_layout()
        plt.show()
