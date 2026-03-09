import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


def build_hetero_graph(df):
    """
    Build heterogeneous graph from preprocessed dataframe.
    """

    print("Building heterogeneous graph...")

    data = HeteroData()

    # --------------------------------
    # 1. Handle missing values
    # --------------------------------

    df["DeviceInfo"] = df["DeviceInfo"].fillna("unknown_device")
    df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown_email")
    df["addr1"] = df["addr1"].fillna(-1)

    # --------------------------------
    # 2. Create node mappings
    # --------------------------------

    customer_ids = df["card1"].unique()
    transaction_ids = df["TransactionID"].unique()
    device_ids = df["DeviceInfo"].unique()
    email_ids = df["P_emaildomain"].unique()
    address_ids = df["addr1"].unique()

    customer_map = {v: i for i, v in enumerate(customer_ids)}
    transaction_map = {v: i for i, v in enumerate(transaction_ids)}
    device_map = {v: i for i, v in enumerate(device_ids)}
    email_map = {v: i for i, v in enumerate(email_ids)}
    address_map = {v: i for i, v in enumerate(address_ids)}

    # --------------------------------
    # 3. Assign node counts
    # --------------------------------

    data["customer"].num_nodes = len(customer_map)
    data["transaction"].num_nodes = len(transaction_map)
    data["device"].num_nodes = len(device_map)
    data["email"].num_nodes = len(email_map)
    data["address"].num_nodes = len(address_map)

    # --------------------------------
    # 4. Create edge indices
    # --------------------------------

    cust_idx = torch.tensor(df["card1"].map(customer_map).values, dtype=torch.long)
    tx_idx = torch.tensor(df["TransactionID"].map(transaction_map).values, dtype=torch.long)
    device_idx = torch.tensor(df["DeviceInfo"].map(device_map).values, dtype=torch.long)
    email_idx = torch.tensor(df["P_emaildomain"].map(email_map).values, dtype=torch.long)
    address_idx = torch.tensor(df["addr1"].map(address_map).values, dtype=torch.long)

    # --------------------------------
    # 5. Define edges (ALL → transaction)
    # --------------------------------

    # Customer → Transaction
    data["customer", "makes", "transaction"].edge_index = torch.stack([cust_idx, tx_idx])

    # Device → Transaction
    data["device", "used_in", "transaction"].edge_index = torch.stack([device_idx, tx_idx])

    # Email → Transaction
    data["email", "linked_to", "transaction"].edge_index = torch.stack([email_idx, tx_idx])

    # Address → Transaction
    data["address", "located_at", "transaction"].edge_index = torch.stack([address_idx, tx_idx])

    # --------------------------------
    # 6. Make graph undirected (adds reverse edges automatically)
    # --------------------------------

    data = ToUndirected()(data)

    print("Graph construction complete.")
    print(data)

    return data