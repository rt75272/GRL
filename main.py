"""Main script to demonstrate the GraphEmbedding model on a simple graph.

This script initializes a GraphEmbedding model, trains it on a small line graph 
with a random number of nodes between 5 and 20, and prints the learned 
embeddings for each node after training.

Usage:
    python main.py
"""
import random
from graph_embedding import GraphEmbedding
from visualization import plot_graph_embeddings

# The big red activation button.
if __name__ == "__main__":
    num_nodes = random.randint(5, 20) # Generate a random number of nodes between 5 and 20.
    # Generate random edges for the graph.
    edges = []
    for i in range(num_nodes):
        # Connect to at least one previous node to keep the graph mostly connected.
        if i > 0:
            edges.append((i, random.randint(0, i - 1)))
        # Add a few more random edges per node to create interesting structures.
        for _ in range(random.randint(0, 2)):
            u, v = i, random.randint(0, num_nodes - 1)
            # Avoid self-loops and duplicate edges.
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))
    print(f"Initializing GraphEmbedding model with {num_nodes} nodes and {len(edges)} edges...")
    model = GraphEmbedding(num_nodes=num_nodes, embedding_dim=2, lr=0.1)
    print("Training starting...")
    model.train(edges, epochs=150, negative_samples=4)
    print("Learned Embeddings:")
    for i in range(num_nodes):
        emb = [round(x, 4) for x in model.get_embedding(i)]
        print(f"Node {i}: {emb}")
    plot_graph_embeddings(model, num_nodes, edges) # Plot the embeddings using the visualization module.



