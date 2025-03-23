import pandas as pd

# Load data
df = pd.read_csv("ASD_relative_abundance.csv")

# Convert from wide to long format
df_long = df.melt(id_vars=["Sample"], var_name="Microbe", value_name="Abundance")

# Create Node Table
samples = df["Sample"].unique()
microbes = df_long["Microbe"].unique()

node_data = pd.DataFrame(
    {"ID": list(samples) + list(microbes),
     "Type": ["Sample"] * len(samples) + ["Microbe"] * len(microbes)}
)

node_data.to_csv("ASD_node_table.csv", index=False)

# Create Edge Table
edge_data = df_long.rename(columns={"Sample": "Source", "Microbe": "Target", "Abundance": "Weight"})
edge_data["Interaction"] = "presence"

edge_data.to_csv("ASD_edge_table.csv", index=False)

print("Node and Edge tables saved successfully!")
