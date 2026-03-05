# GRL (Graph Representation Learning)

Graph representation learning is a machine learning field that automatically encodes complex graph-structured data—nodes and edges—into low-dimensional vector spaces (embeddings). It captures network structure, connectivity, and features, allowing standard machine learning models to perform tasks like node classification, link prediction, and graph visualization.

## Key Components and Techniques
- **Encoders/Embeddings:** Methods like Graph Neural Networks (GNNs), DeepWalk, and node2vec map nodes to vectors, preserving topological information.
- **Graph Neural Networks (GNNs):** Deep learning architectures that operate directly on graphs, using message-passing to update node representations based on neighbors.
- **Applications:** Used for recommender systems, drug discovery, social network analysis, and knowledge graph completion.
- **Types of Learning:** Transductive learning (uses all node information during training) and inductive learning (generalizes to unseen graphs).

## Advantages over Traditional Methods
Unlike manual feature engineering, GRL automatically learns meaningful representations from raw graph inputs, handling non-Euclidean data more effectively.

## Project Layout

```
.
├── main.py              # Main application logic
├── graph_embedding.py   # Core GraphEmbedding model logic
├── visualization.py     # Graph plotting and visualization utilities
├── requirements.txt     # Python project dependencies
├── README.md            # Project documentation
└── grl/                 # Python virtual environment 
```

## How to Build and Run

1. **Activate the Virtual Environment**
   Activate the included `grl` python environment:
   ```bash
   source grl/bin/activate
   ```

2. **Install Dependencies**
   Install the required python packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project**
   Execute the main script:
   ```bash
   python main.py
   ```
   This will train the graph embedding model on a randomly generated graph and automatically create a visual representation saved as `graph_embeddings.png` in the project root!