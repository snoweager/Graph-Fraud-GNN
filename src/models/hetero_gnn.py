import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroFraudGNN(nn.Module):

    def __init__(self, metadata, hidden_dim=64, in_channels_dict=None):
        """
        Args:
            metadata:         graph metadata (node types, edge types)
            hidden_dim:       size of hidden representations
            in_channels_dict: {node_type: input_feature_dim}
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # ----------------------------------------------------------------
        # Input projection: project every node type → hidden_dim
        # Ensures all nodes enter conv1 with consistent feature sizes.
        # ----------------------------------------------------------------
        self.input_proj = nn.ModuleDict()
        if in_channels_dict:
            for node_type, in_channels in in_channels_dict.items():
                self.input_proj[node_type] = nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )

        edge_types = [
            ('customer',    'makes',      'transaction'),
            ('device',      'used_in',    'transaction'),
            ('email',       'linked_to',  'transaction'),
            ('address',     'located_at', 'transaction'),
        ]
        rev_edge_types = [
            ('transaction', 'rev_makes',      'customer'),
            ('transaction', 'rev_used_in',    'device'),
            ('transaction', 'rev_linked_to',  'email'),
            ('transaction', 'rev_located_at', 'address'),
        ]

        # ----------------------------------------------------------------
        # Conv Layer 1 + BatchNorm per node type
        # ----------------------------------------------------------------
        self.conv1 = HeteroConv(
            {et: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
             for et in edge_types + rev_edge_types},
            aggr='sum'
        )
        self.bn1 = nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_dim)
            for node_type in ['customer', 'transaction', 'device', 'email', 'address']
        })

        # ----------------------------------------------------------------
        # Conv Layer 2 + BatchNorm per node type
        # ----------------------------------------------------------------
        self.conv2 = HeteroConv(
            {et: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
             for et in edge_types + rev_edge_types},
            aggr='sum'
        )
        self.bn2 = nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_dim)
            for node_type in ['customer', 'transaction', 'device', 'email', 'address']
        })

        # ----------------------------------------------------------------
        # Step 7: Improved Prediction Layer
        #
        # Returns a single fraud probability per transaction.
        # Sigmoid is applied here so train_gnn can use BCELoss +
        # tune the decision threshold independently of the model.
        # ----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, graph):

        x_dict          = graph.x_dict
        edge_index_dict = graph.edge_index_dict

        # -------- Input Projection --------
        projected = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_proj:
                projected[node_type] = self.input_proj[node_type](x)
            else:
                projected[node_type] = x
        x_dict = projected

        # -------- Conv Layer 1 --------
        out_dict = self.conv1(x_dict, edge_index_dict)
        for key in x_dict:
            if key not in out_dict:
                out_dict[key] = x_dict[key]
        x_dict = {
            k: F.relu(self.bn1[k](v)) if k in self.bn1 else F.relu(v)
            for k, v in out_dict.items()
        }

        # -------- Conv Layer 2 --------
        out_dict = self.conv2(x_dict, edge_index_dict)
        for key in x_dict:
            if key not in out_dict:
                out_dict[key] = x_dict[key]
        x_dict = {
            k: F.relu(self.bn2[k](v)) if k in self.bn2 else F.relu(v)
            for k, v in out_dict.items()
        }

        # -------- Predict Fraud Probability --------
        txn_embeddings = x_dict["transaction"]
        fraud_prob = self.classifier(txn_embeddings)  # shape: [N, 1]

        return fraud_prob.squeeze(1)                  # shape: [N]
