import os
import matplotlib.pyplot as plt


def save_graph_statistics(graph_data, output_dir="outputs/graphs"):
    """
    Saves bar graphs describing the heterogeneous graph structure.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating graph statistics...")

    # -------------------------
    # Node counts
    # -------------------------
    node_types = []
    node_counts = []

    for node_type in graph_data.node_types:
        node_types.append(node_type)
        node_counts.append(graph_data[node_type].num_nodes)

    plt.figure()
    plt.bar(node_types, node_counts)
    plt.title("Node Type Distribution")
    plt.xlabel("Node Type")
    plt.ylabel("Number of Nodes")

    node_path = os.path.join(output_dir, "node_distribution.png")
    plt.savefig(node_path)
    plt.close()

    print(f"Saved node distribution → {node_path}")

    # -------------------------
    # Edge counts
    # -------------------------
    edge_types = []
    edge_counts = []

    for edge_type in graph_data.edge_types:
        edge_types.append(str(edge_type))
        edge_counts.append(graph_data[edge_type].edge_index.shape[1])

    plt.figure()
    plt.bar(range(len(edge_types)), edge_counts)
    plt.xticks(range(len(edge_types)), edge_types, rotation=45)
    plt.title("Edge Type Distribution")
    plt.xlabel("Edge Type")
    plt.ylabel("Number of Edges")

    edge_path = os.path.join(output_dir, "edge_distribution.png")
    plt.tight_layout()
    plt.savefig(edge_path)
    plt.close()

    print(f"Saved edge distribution → {edge_path}")

    print("Graph statistics saved successfully.")