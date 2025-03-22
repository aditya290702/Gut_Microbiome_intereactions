import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, VGAE
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

# Load microbial abundance data
df = pd.read_csv("ASD_cleaned_dataset.csv", sep=",", index_col=0)

# Construct microbial co-occurrence network
G = nx.Graph()
corr_matrix = df.corr(method="spearman")
for species1 in df.columns:
    for species2 in df.columns:
        if species1 != species2:
            corr, _ = spearmanr(df[species1], df[species2])
            if abs(corr) > 0.6:
                G.add_edge(species1, species2, weight=corr)

# Convert network to PyTorch Geometric format
graph_data = from_networkx(G)
node_features = torch.tensor(df.mean().values, dtype=torch.float).view(-1, 1)
graph_data.x = node_features

# Define GNN Encoder
def gnn_encoder(x, edge_index):
    x = F.relu(GCNConv(1, 16)(x, edge_index))
    x = GCNConv(16, 16)(x, edge_index)
    return x

# Train Graph Autoencoder
model = VGAE(lambda x, edge_index: gnn_encoder(x, edge_index))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    z = model.encode(graph_data.x, graph_data.edge_index)
    loss = model.recon_loss(z, graph_data.edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Extract node embeddings
model.eval()
with torch.no_grad():
    embeddings = model.encode(graph_data.x, graph_data.edge_index).numpy()

# Cluster species using K-Means
clusters = KMeans(n_clusters=3, random_state=42).fit_predict(embeddings)

# Identify keystone species by centrality
degree_centrality = nx.degree_centrality(G)
keystone_species = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
print("Keystone species:", keystone_species)
