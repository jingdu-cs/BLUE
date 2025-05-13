import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFCorrection(nn.Module):
    """
    MRF correction layer for sparse heterogeneous graphs.
    This layer applies iterative message passing to refine node features based on
    local graph structure, suitable for handling potentially sparse connections.
    """
    def __init__(self, node_types, edge_types, hidden_dim=64, num_iterations=3):
        super(MRFCorrection, self).__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        self.edge_weights = nn.ParameterDict({
            f"{src}_{rel}_{dst}": nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            for src, rel, dst in edge_types
        })
        
        self.node_biases = nn.ParameterDict({
            node_type: nn.Parameter(torch.zeros(hidden_dim))
            for node_type in node_types
        })
        
        for param in self.edge_weights.values():
            nn.init.xavier_uniform_(param)
    
    def message_passing(self, x_dict, edge_index_dict):
        """
        MRF message passing
        """
        new_x_dict = {}
        
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue

            num_nodes = x_dict[node_type].size(0)
            dtype = x_dict[node_type].dtype
            device = x_dict[node_type].device
            
            messages = torch.zeros(num_nodes, self.hidden_dim, dtype=dtype, device=device)
            message_count = torch.zeros(num_nodes, dtype=dtype, device=device)

            for src, rel, dst in self.edge_types:
                if dst == node_type and (src, rel, dst) in edge_index_dict:
                    edge_index = edge_index_dict[(src, rel, dst)]
                    weight = self.edge_weights[f"{src}_{rel}_{dst}"]

                    src_features = x_dict[src]
                    weighted_features = torch.matmul(src_features, weight)  # [num_src_nodes, hidden_dim]
                    
                    dst_indices = edge_index[1]  # destination node indices
                    src_indices = edge_index[0]  # source node indices
                    
                    ones = torch.ones(dst_indices.size(0), dtype=dtype, device=device)
                    message_count.index_add_(0, dst_indices, ones)
                    
                    for i in range(weighted_features.size(1)):
                        messages[:, i].index_add_(0, dst_indices, weighted_features[src_indices, i])
             
            mask = message_count > 0
            if mask.any():
                messages[mask] = messages[mask] / message_count[mask].unsqueeze(-1)
            
            new_x_dict[node_type] = F.relu(messages + self.node_biases[node_type])
        
        return new_x_dict
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass, perform multiple MRF iterations
        """
        current_x_dict = x_dict.copy()
        
        for _ in range(self.num_iterations):
            new_x_dict = self.message_passing(current_x_dict, edge_index_dict)
            
            for node_type in new_x_dict:
                if node_type in current_x_dict:
                    current_x_dict[node_type] = current_x_dict[node_type] + new_x_dict[node_type]
                else:
                    current_x_dict[node_type] = new_x_dict[node_type]
        
        return current_x_dict

