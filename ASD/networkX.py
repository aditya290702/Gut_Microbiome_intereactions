import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
from scipy.cluster.hierarchy import linkage, dendrogram

#Loading Preprocessed Data
df = pd.read_csv("cleaned_dataset.csv", index_col=0)

#Compute Correlation Matrix
corr_matrix = df.corr(method='spearman')

#Filter Correlations
threshold = 0.5
corr_matrix_filtered = corr_matrix.where(np.abs(corr_matrix) > threshold, other=np.nan)

#Build the Interaction Network
G = nx.Graph()

# Add all microbes as nodes.
for microbe in corr_matrix_filtered.index:
    G.add_node(microbe)

# Add edges for strong correlations (avoid self-loops and duplicates).
for i in corr_matrix_filtered.index:
    for j in corr_matrix_filtered.columns:
        if i < j:  # ensures each pair is processed only once
            corr_val = corr_matrix_filtered.loc[i, j]
            if not np.isnan(corr_val):
                G.add_edge(i, j, weight=corr_val)

#Visualize the Network
plt.figure(figsize=(12, 12), dpi=300)
pos = nx.spring_layout(G, k=0.5, seed=42)  # layout that spreads out the nodes

# Get edge weights for coloring
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.coolwarm, width=1)

# Draw nodes (small size)
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')

# Draw labels with small font size
nx.draw_networkx_labels(G, pos, font_size=5)
plt.title("Microbial Interaction Network")
plt.savefig("microbial_network.png", bbox_inches='tight', dpi=300)
plt.show()

#Community Detection (Clustering) using Positive Weights Only
G_pos = nx.Graph()
for u, v, d in G.edges(data=True):
    if d['weight'] > 0:
        G_pos.add_edge(u, v, weight=d['weight'])

# Now, compute the partition on the positive-weight subgraph
partition = community_louvain.best_partition(G_pos)

# Assign the partition to the original graph (for visualization)
nx.set_node_attributes(G, partition, 'community')

# Visualize network with nodes colored by community
plt.figure(figsize=(12, 12), dpi=300)
pos = nx.spring_layout(G, k=0.5, seed=42)
node_colors = [partition.get(node, 0) for node in G.nodes()]  # Use 0 as default if missing
nx.draw_networkx_nodes(G, pos, node_size=50, cmap=plt.cm.jet, node_color=node_colors)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=5)
plt.title("Microbial Interaction Network with Communities (Positive Edges)")
plt.savefig("microbial_network_communities.png", bbox_inches='tight', dpi=300)
plt.show()


# ----------------------------
# 7. (Optional) Functional Annotation & Clustering
# ----------------------------
# If you have a functional annotation file mapping microbes to functions, load it.
# Expected format: a CSV file with index as 'Microbe' and one or more columns representing functional features.
try:
    func_df = pd.read_csv("functional_annotation.csv", index_col=0)
    # Filter to only microbes present in the network
    func_df = func_df.loc[G.nodes(), :]

    # For hierarchical clustering, assume the functional features are numeric.
    functional_data = func_df.values
    linkage_matrix = linkage(functional_data, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=func_df.index, leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering of Microbial Functions")
    plt.xlabel("Microbe")
    plt.ylabel("Distance")
    plt.savefig("functional_clustering.png", bbox_inches='tight', dpi=300)
    plt.show()
except FileNotFoundError:
    print("No functional annotation file found. Skipping functional clustering.")


# ----------------------------
# 8. Plot Top 10 Interacting Microbes and Their Connections
# ----------------------------

# Compute degree centrality for all microbes in the network
degree_centrality = nx.degree_centrality(G)

# Sort microbes by degree centrality (descending order) and select the top 10
top_10_microbes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
print("Top 10 interacting microbes:", top_10_microbes)

# Create a subgraph containing only the top 10 microbes
subG = G.subgraph(top_10_microbes)

# Visualize the subgraph
plt.figure(figsize=(8, 8), dpi=300)
pos_sub = nx.spring_layout(subG, k=0.7, seed=42)

# Get edge weights for coloring (if there are any edges)
if subG.number_of_edges() > 0:
    edges, weights = zip(*nx.get_edge_attributes(subG, 'weight').items())
    nx.draw_networkx_edges(subG, pos_sub, edge_color=weights, edge_cmap=plt.cm.coolwarm, width=1)
else:
    print("No edges among top 10 microbes.")

# Draw nodes with a smaller size
nx.draw_networkx_nodes(subG, pos_sub, node_size=100, node_color='lightgreen')

# Draw labels with small font
nx.draw_networkx_labels(subG, pos_sub, font_size=8, font_weight='bold')

plt.title("Top 10 Interacting Microbes")
plt.savefig("top10_interacting_microbes.png", bbox_inches='tight', dpi=300)
plt.show()
