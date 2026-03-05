"""A simple Graph Representation Learning model from scratch using pure Python.

It optimizes node embeddings to preserve first-order proximity using
stochastic gradient descent with negative sampling.

Args:
    num_nodes (int): Number of nodes in the graph.
    embedding_dim (int): Dimensionality of the node embeddings.
    lr (float): Learning rate for optimization.

Returns:
    A GraphEmbedding instance with trained node embeddings.

Usage:
    from graph_embedding import GraphEmbedding
    model = GraphEmbedding(num_nodes=10, embedding_dim=2, lr=0.1)
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    model.train(edges, epochs=100, negative_samples=3)
    embedding = model.get_embedding(0)  # Get embedding for node 0 
"""
import random
import math

class GraphEmbedding:
    """A simple Graph Representation Learning model from scratch using pure Python.
    
    It optimizes node embeddings to preserve first-order proximity using
    stochastic gradient descent with negative sampling.
    
    Args:
        num_nodes (int): Number of nodes in the graph.
        embedding_dim (int): Dimensionality of the node embeddings.
        lr (float): Learning rate for optimization.
    
    Returns:
        A GraphEmbedding instance with trained node embeddings.
    """
    def __init__(self, num_nodes, embedding_dim=2, lr=0.05):
        """Initialize the GraphEmbedding model.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
            embedding_dim (int): Dimensionality of the node embeddings.
            lr (float): Learning rate for optimization.
        
        Returns:
            None
        """
        self.num_nodes = num_nodes
        self.dim = embedding_dim
        self.lr = lr
        # Initialize embeddings with small random values.
        self.embeddings = [
            [random.uniform(-0.1, 0.1) for _ in range(self.dim)]
            for _ in range(num_nodes)
        ]

    def _dot_product(self, vec1, vec2):
        """Compute the dot product of two vectors.
        
        Args:
            vec1 (list): First vector.
            vec2 (list): Second vector.
            
        Returns:
            float: The dot product of vec1 and vec2.
        """
        result = 0.0
        for x, y in zip(vec1, vec2):
            result += x * y
        return result

    def _sigmoid(self, x):
        """Compute the sigmoid of a value.
        
        Args:
            x (float): Input value.
            
        Returns:
            float: Sigmoid of x.
        """
        if x < -10:
            return 0.0
        if x > 10:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def train(self, edges, epochs=100, negative_samples=3):
        """Train the GraphEmbedding model using stochastic gradient descent with negative sampling.
        
        Args:
            edges (list of tuple): List of edges in the graph.
            epochs (int): Number of training epochs.
            negative_samples (int): Number of negative samples per positive edge.
        
        Returns:
            None
        """
        # Create an adjacency set for fast negative sampling check.
        adj = {}
        for i in range(self.num_nodes):
            adj[i] = set()
        # Populate the adjacency set with all the given edges.
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Iterate over the specified number of training epochs.
        for epoch in range(epochs):
            total_loss = 0.0
            # Process each edge as a positive relationship.
            for u, v in edges:
                # Positive sample update (u, v).
                # Compute the dot product score representing node similarity.
                score = self._dot_product(self.embeddings[u], self.embeddings[v])
                prob = self._sigmoid(score) # Convert the score to a probability using the sigmoid function.
                total_loss -= math.log(prob + 1e-9)# Accumulate the positive sample loss, adding a small epsilon to avoid math domain errors.    
                grad_pos = prob - 1.0 # Calculate the gradient for the positive sample.
                self._update_embeddings(u, v, grad_pos) # Apply the gradient to update the embeddings of the connected nodes.
                # Negative samples update.
                # Draw random negative samples to contrast with the positive edge.
                for _ in range(negative_samples):
                    neg_v = random.randint(0, self.num_nodes - 1) # Pick a target node uniformly at random.
                    # Ensure the randomly selected node is not actually connected to 'u' and is not 'u' itself.
                    if neg_v not in adj[u] and neg_v != u:
                        score_neg = self._dot_product(self.embeddings[u], self.embeddings[neg_v]) # Compute the dot product score for the negative pair.
                        prob_neg = self._sigmoid(score_neg) # Convert the score to a probability using the sigmoid function.
                        total_loss -= math.log(1.0 - prob_neg + 1e-9) # Accumulate the negative sample loss.
                        grad_neg = prob_neg # Calculate the gradient for the negative sample.
                        self._update_embeddings(u, neg_v, grad_neg) # Apply the gradient to update the embeddings to push the nodes apart.
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def _update_embeddings(self, u, v, grad):
        """Update the embeddings of nodes u and v based on the computed gradient.
        
        Args:
            u (int): Index of the first node.
            v (int): Index of the second node.
            grad (float): Gradient to apply for the update.
        
        Returns:
            None
        """
        # Iterate over each dimension of the embedding vectors.
        for i in range(self.dim):
            grad_u_i = grad * self.embeddings[v][i] # Compute the partial derivative for the current dimension of node u's embedding.
            grad_v_i = grad * self.embeddings[u][i] # Compute the partial derivative for the current dimension of node v's embedding.
            self.embeddings[u][i] -= self.lr * grad_u_i # Update the current dimension of node u's embedding using the learning rate.
            self.embeddings[v][i] -= self.lr * grad_v_i # Update the current dimension of node v's embedding using the learning rate.

    def get_embedding(self, node):
        """Get the embedding of a specific node.
        
        Args:
            node (int): Index of the node.
        
        Returns:
            list: The embedding vector of the node.
        """
        return self.embeddings[node]
