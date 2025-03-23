import pandas as pd
import networkx as nx
from pyvis.network import Network

# Load data
nodes_df = pd.read_csv("ASD_node_table.csv")
edges_df = pd.read_csv("ASD_edge_table.csv")
abundance_df = pd.read_csv("ASD_relative_abundance.csv")

# Standardize column names
nodes_df.columns = nodes_df.columns.str.lower()
edges_df.columns = edges_df.columns.str.lower()
abundance_df.columns = abundance_df.columns.str.lower()

# Merge relative abundance data
if "id" in abundance_df.columns:
    nodes_df = nodes_df.merge(abundance_df, on="id", how="left")

# Handle missing values by setting a default size
nodes_df["abundance"] = nodes_df.iloc[:, 2:].sum(axis=1)  # Summing all abundance columns
nodes_df["size"] = nodes_df["abundance"].fillna(10).astype(int)

# Create a NetworkX graph
G = nx.Graph()

# Add nodes
for _, row in nodes_df.iterrows():
    G.add_node(row["id"], label=row["id"], size=row["size"], group=row.get("type", None))

# Add edges
for _, row in edges_df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row.get("weight", 1))

# Create Pyvis network
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)

# Save the visualization
net.save_graph("microbial_network.html")
print("Graph saved as microbial_network.html")
