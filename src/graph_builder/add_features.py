import torch


def attach_node_features(graph, df):

    print("Attaching node features to graph...")

    # -----------------------------
    # Select usable feature columns
    # -----------------------------

    feature_cols = [
        col for col in df.columns
        if col not in ["TransactionID", "isFraud"]
    ]

    # Convert dataframe → numpy
    transaction_features = df[feature_cols].values

    # Convert numpy → torch tensor
    graph["transaction"].x = torch.tensor(
        transaction_features,
        dtype=torch.float
    )

    # -----------------------------
    # Transaction labels
    # -----------------------------

    labels = df["isFraud"].values

    graph["transaction"].y = torch.tensor(
        labels,
        dtype=torch.long
    )

    print("Transaction feature matrix shape:",
          graph["transaction"].x.shape)

    print("Transaction labels shape:",
          graph["transaction"].y.shape)

    # -----------------------------
    # Create dummy features for
    # other node types
    # -----------------------------

    embedding_dim = 16

    for node_type in graph.node_types:

        if node_type != "transaction":

            num_nodes = graph[node_type].num_nodes

            graph[node_type].x = torch.randn(
                (num_nodes, embedding_dim),
                dtype=torch.float
            )

            print(f"{node_type} feature matrix shape:",
                  graph[node_type].x.shape)

    return graph