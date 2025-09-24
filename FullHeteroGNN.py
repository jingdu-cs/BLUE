import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import numpy as np


class MRFCorrection(nn.Module):
    """
    Markov Random Field (MRF) correction layer for sparse heterogeneous graphs.
    This layer applies iterative message passing to refine node features based on
    local graph structure, suitable for handling potentially sparse connections.
    """
    def __init__(self, node_types, edge_types, hidden_dim=64, num_iterations=3):
        """
        Initializes the MRFCorrection layer.

        Args:
            node_types (list[str]): List of node type names.
            edge_types (list[tuple[str, str, str]]): List of edge types, where each
                tuple represents (source_node_type, relation_type, destination_node_type).
            hidden_dim (int): The dimensionality of the hidden node features.
            num_iterations (int): The number of message passing iterations to perform.
        """
        super(MRFCorrection, self).__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # Message passing weights for each edge type
        self.edge_weights = nn.ParameterDict({
            f"{src}_{rel}_{dst}": nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            for src, rel, dst in edge_types
        })
        
        # Node type specific biases
        self.node_biases = nn.ParameterDict({
            node_type: nn.Parameter(torch.zeros(hidden_dim))
            for node_type in node_types
        })
        
        # Initialize weights
        for param in self.edge_weights.values():
            nn.init.xavier_uniform_(param)
    
    def message_passing(self, x_dict, edge_index_dict):
        """
        Performs one iteration of MRF message passing.

        Args:
            x_dict (dict[str, torch.Tensor]): Dictionary mapping node types to their
                feature tensors.
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Dictionary
                mapping edge types to their edge index tensors (COO format).

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping node types to their updated
                feature tensors after one message passing step.
        """
        new_x_dict = {}
        
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue

            messages = torch.zeros_like(x_dict[node_type])
            # Keep track of how many messages each node receives for normalization
            message_count = torch.zeros(x_dict[node_type].size(0), device=x_dict[node_type].device)

            # Aggregate messages from neighbors for the current node type
            for src, rel, dst in self.edge_types:
                if dst == node_type and (src, rel, dst) in edge_index_dict:
                    edge_index = edge_index_dict[(src, rel, dst)]
                    weight = self.edge_weights[f"{src}_{rel}_{dst}"]

                    src_features = x_dict[src]
                    # Aggregate weighted source features to destination nodes
                    messages.index_add_(0, edge_index[1], torch.matmul(src_features[edge_index[0]], weight))
                    # Count messages received by each destination node
                    message_count.index_add_(0, edge_index[1], torch.ones_like(edge_index[1], dtype=torch.float))
            
            # Normalize messages by the count (average aggregation)
            mask = message_count > 0
            if mask.any():
                messages[mask] = messages[mask] / message_count[mask].unsqueeze(-1)
            
            # Apply bias and activation
            new_x_dict[node_type] = F.relu(messages + self.node_biases[node_type])
        
        return new_x_dict
    
    def forward(self, x_dict, edge_index_dict):
        """
        Applies multiple iterations of MRF message passing.

        Args:
            x_dict (dict[str, torch.Tensor]): Initial dictionary mapping node types
                to their feature tensors.
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Dictionary
                mapping edge types to their edge index tensors.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping node types to their refined
                feature tensors after MRF iterations.
        """
        current_x_dict = x_dict.copy()
        
        for _ in range(self.num_iterations):
            new_x_dict = self.message_passing(current_x_dict, edge_index_dict)
            
            # Update node features with residual connection
            for node_type in new_x_dict:
                if node_type in current_x_dict:
                    # Residual connection
                    current_x_dict[node_type] = current_x_dict[node_type] + new_x_dict[node_type]
                else:
                    # If node type didn't exist before, just assign the new features
                    current_x_dict[node_type] = new_x_dict[node_type]
        
        return current_x_dict


class FullHeteroGNN(nn.Module):
    """
    A Heterogeneous Graph Neural Network designed for temporal prediction tasks,
    specifically predicting future county infection counts based on spatial,
    genetic, and county-case relationships. Includes an MRF correction layer
    and supports autoregressive prediction.
    """

    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.3, pred_horizon=4, num_mrf=3):
        """
        Initializes the FullHeteroGNN model.

        Args:
            hidden_dim (int): The dimensionality of hidden layers.
            num_layers (int): The number of HeteroConv GNN layers.
            dropout (float): Dropout rate applied within the GNN layers.
            pred_horizon (int): The number of future time steps to predict.
            num_mrf (int): The number of iterations for the MRF correction layer.
        """
        super(FullHeteroGNN, self).__init__()

        self.num_mrf = num_mrf
        self.node_types = ['county', 'case']
        self.edge_types = [
            ('county', 'spatial', 'county'),
            ('case', 'genetic', 'case'),
            ('case', 'belongs_to', 'county')
        ]
        # Define the reverse edge type for message passing from county to case
        self.reverse_edge_types = {
             ('case', 'belongs_to', 'county'): ('county', 'contains', 'case')
        }
        self.pred_horizon = pred_horizon

        # Input feature encoders for different node types
        # Input: [normalized_infection_count, normalized_sample_count]
        self.county_encoder = nn.Linear(2, hidden_dim)
        # Input: Single feature (e.g., genetic marker presence)
        self.case_encoder = nn.Linear(1, hidden_dim)

        # MRF correction layer
        self.mrf_correction = MRFCorrection(
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_dim=hidden_dim,
            num_iterations=self.num_mrf
        )

        # Heterogeneous Graph Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {etype: SAGEConv((-1, -1), hidden_dim) for etype in self.edge_types}
            # Add convolution for the reverse edge ('county', 'contains', 'case')
            reverse_etype = self.reverse_edge_types[('case', 'belongs_to', 'county')]
            conv_dict[reverse_etype] = SAGEConv((-1, -1), hidden_dim)

            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Attention mechanism to fuse initial and GNN-processed county features
        self.county_attn = nn.Linear(hidden_dim * 2, 1)

        # Final linear layer for predicting the target value (e.g., infection count)
        self.lin_county = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def encode_features(self, x_dict, edge_index_dict, step=0):
        """
        Encodes initial node features, applies MRF correction, and runs HeteroGNN layers.

        Args:
            x_dict (dict[str, torch.Tensor]): Dictionary of node features.
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Dictionary of edge indices.
            step (int): The current time step in autoregressive prediction (0 for initial encoding).

        Returns:
            torch.Tensor: Enhanced county feature representations after encoding and convolution.
        """
        # Encode raw features only at the first step
        if step == 0:
            if 'case' in x_dict:
                # Ensure case features are 2D [num_cases, 1] before encoding
                case_feat = x_dict['case']
                if case_feat.dim() == 1:
                    case_feat = case_feat.unsqueeze(-1)
                x_dict['case'] = self.case_encoder(case_feat)
            if 'county' in x_dict:
                x_dict['county'] = self.county_encoder(x_dict['county'])

        # Apply MRF correction
        x_dict = self.mrf_correction(x_dict, edge_index_dict)

        # Store initial county features after MRF for later fusion
        if 'county' in x_dict:
             initial_county_features = x_dict['county'].clone() # Clone to prevent modification
        else:
             # Handle case where 'county' might not be present after MRF (e.g., empty graph)
             # This might need specific handling based on expected data sparsity.
             # For now, assume it exists or handle potential errors downstream.
             pass # Or raise an error, or initialize with zeros if appropriate

        # Dynamically add reverse edges if they are not present in the input
        temp_edge_index_dict = edge_index_dict.copy()
        for etype, reverse_etype in self.reverse_edge_types.items():
            if etype in temp_edge_index_dict and reverse_etype not in temp_edge_index_dict:
                 # Ensure edge_index has shape [2, num_edges]
                 edge_index = temp_edge_index_dict[etype]
                 if edge_index.numel() > 0: # Check if there are edges of this type
                     temp_edge_index_dict[reverse_etype] = edge_index.flip(0)
                 else:
                     # Handle empty edge index case if necessary
                     temp_edge_index_dict[reverse_etype] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)


        # Apply HeteroGNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, temp_edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Fuse case information into county features using attention
        if 'county' in x_dict and 'county' in initial_county_features.keys(): # Check if county features exist
            # Concatenate current GNN features with initial (post-MRF) features
            county_features_combined = torch.cat([x_dict['county'], initial_county_features], dim=1)
            # Calculate attention weights
            attention_weights = torch.sigmoid(self.county_attn(county_features_combined))
            # Apply attention for weighted fusion
            enhanced_county_features = attention_weights * x_dict['county'] + (1 - attention_weights) * initial_county_features
        elif 'county' in x_dict:
             # If initial features weren't stored (e.g., no county nodes initially), just use current GNN output
             enhanced_county_features = x_dict['county']
        else:
             # Handle case where county features are missing after GNN layers
             # This indicates a potential issue or requires specific logic.
             # Returning None or an empty tensor might be options depending on downstream use.
             return None # Or appropriate handling

        return enhanced_county_features

    def predict_step(self, county_features):
        """
        Predicts the target value for a single time step using county features.

        Args:
            county_features (torch.Tensor): The county feature tensor from the GNN.

        Returns:
            torch.Tensor: The prediction for the next time step. [num_counties, 1]
        """
        if county_features is None:
            # Handle missing county features if necessary
             # Returning zeros, None, or raising an error are options
             return None # Or appropriate tensor of zeros
        return self.lin_county(county_features)

    def forward(self, x_dict, edge_index_dict):
        """
        Main forward pass of the model. Uses autoregressive prediction.

        Args:
            x_dict (dict[str, torch.Tensor]): Initial dictionary of node features for the first time step.
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Dictionary of edge indices.
                Assumed to be constant across the prediction horizon for simplicity,
                but could be adapted for dynamic graphs.

        Returns:
            torch.Tensor: Predictions for all time steps in the horizon.
                          Shape: [num_counties, pred_horizon]
        """
        # Use autoregressive prediction
        return self.autoregressive_predict(x_dict, edge_index_dict)

    def autoregressive_predict(self, x_dict_init, edge_index_dict):
        """
        Performs autoregressive prediction over the prediction horizon.
        The prediction for time t is used (implicitly) as part of the input features for time t+1.
        Note: This implementation currently re-encodes features from scratch at each step,
              using the GNN output features (`county_features`) from step t-1 as the county input for step t.
              It does NOT explicitly merge the `current_pred` back into the input `x_dict['county']` features yet.

        Args:
            x_dict_init (dict[str, torch.Tensor]): Initial dictionary of node features.
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Dictionary of edge indices.

        Returns:
            torch.Tensor: Concatenated predictions for the entire horizon.
                          Shape: [num_counties, pred_horizon]
                          Returns None if prediction fails at any step (e.g., missing county features).
        """
        # Clone initial features to avoid modifying the original input dict
        current_x_dict = {k: v.clone() for k, v in x_dict_init.items()}

        predictions = []
        county_features = None # Initialize county_features

        for t in range(self.pred_horizon):
            # In step t>0, use the *output* county features from the previous step (t-1)
            # as the *input* county features for the current step t.
            # This assumes the GNN output captures the relevant temporal evolution.
            if t > 0 and county_features is not None:
                 # Update the 'county' entry in the dictionary passed to encode_features
                 # Note: This replaces the original encoded county features from step 0
                 current_x_dict['county'] = county_features
            elif t > 0 and county_features is None:
                 # If previous step failed to produce county features, cannot continue
                 print(f"Warning: Missing county features at step {t}. Stopping prediction.")
                 # Return None or partially filled predictions depending on requirements
                 return None # Or handle appropriately


            # Encode features and get county representations for the current step t
            county_features = self.encode_features(current_x_dict, edge_index_dict, step=t)

            if county_features is None:
                # If encoding fails (e.g., no county nodes), stop prediction
                print(f"Warning: Failed to encode features at step {t}. Stopping prediction.")
                return None # Or handle appropriately

            # Predict the next time step using the obtained county features
            current_pred = self.predict_step(county_features)

            if current_pred is None:
                 print(f"Warning: Failed to predict at step {t}. Stopping prediction.")
                 return None # Or handle appropriately

            predictions.append(current_pred)

        # Concatenate predictions along the time dimension
        if not predictions:
            # Handle case where no predictions were made
            # Need to determine the expected shape based on potential input county count
            num_counties = x_dict_init['county'].shape[0] if 'county' in x_dict_init else 0
            return torch.empty((num_counties, 0)) # Return empty tensor with correct first dim

        return torch.cat(predictions, dim=1)

