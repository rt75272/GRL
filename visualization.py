""" Visualization module for graph embeddings.

This module provides a function to generate and save a visual plot of the graph embeddings and their edges.
The plot displays the nodes in the embedding space, with edges drawn between them to represent the graph 
structure. Each node is annotated with its number and embedding coordinates for clarity.

Usage:
    from graph_embedding import GraphEmbedding
    from visualization import plot_graph_embeddings
    # Initialize and train the GraphEmbedding model...
    plot_graph_embeddings(model, num_nodes, edges)
"""
import matplotlib.pyplot as plt

def plot_graph_embeddings(model, num_nodes, edges, output_filename='graph_embeddings.png'):
    """Generate and save a visual plot of the graph embeddings and their edges.
    
    Args:
        model: The trained GraphEmbedding model.
        num_nodes (int): Number of nodes in the graph.
        edges (list of tuple): List of edges connecting the nodes.
        output_filename (str): The filename to save the generated plot.
    """
    print("\nGenerating graph visualization...")
    x_coords = [model.get_embedding(i)[0] for i in range(num_nodes)]
    y_coords = [model.get_embedding(i)[1] for i in range(num_nodes)]
    plt.figure(figsize=(10, 8))
    # Draw the edges.
    for u, v in edges:
        plt.plot([x_coords[u], x_coords[v]], [y_coords[u], y_coords[v]], 'k-', alpha=0.3)
    plt.scatter(x_coords, y_coords, s=500, c='skyblue', edgecolors='black', zorder=5) # Draw the nodes.
    # Annotate the nodes with their numbers (coordinates).
    for i in range(num_nodes):
        x, y = x_coords[i], y_coords[i]
        label = f"N{i}\n[{x:.2f}, {y:.2f}]"
        plt.annotate(
            label, 
            (x, y), 
            textcoords="offset points", 
            xytext=(0, 15), 
            ha='center', 
            fontsize=9,
            backgroundcolor="white"
        )
    plt.title("Graph Embeddings Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.scatter(x_coords, y_coords, s=500, c='skyblue', edgecolors='black', zorder=5) # Draw the nodes.
    # Annotate the nodes with their numbers (coordinates).
    for i in range(num_nodes):
        x, y = x_coords[i], y_coords[i]
        label = f"N{i}\n[{x:.2f}, {y:.2f}]"
        plt.annotate(
            label, 
            (x, y), 
            textcoords="offset points", 
            xytext=(0, 15), 
            ha='center', 
            fontsize=9,
            backgroundcolor="white"
        )
    plt.title("Graph Embeddings Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.scatter(x_coords, y_coords, s=500, c='skyblue', edgecolors='black', zorder=5) # Draw the nodes.
    # Annotate the nodes with their numbers (coordinates).
    for i in range(num_nodes):
        x, y = x_coords[i], y_coords[i]
        label = f"N{i}\n[{x:.2f}, {y:.2f}]"
        plt.annotate(
            label, 
            (x, y), 
            textcoords="offset points", 
            xytext=(0, 15), 
            ha='center', 
            fontsize=9,
            backgroundcolor="white"
        )
    plt.title("Graph Embeddings Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    # Save and optionally display.
    plt.savefig(output_filename)
    print(f"Visualization saved to '{output_filename}'")
    plt.show()  # Comment this out if you want it to NOT pop up in a UI window when run.
