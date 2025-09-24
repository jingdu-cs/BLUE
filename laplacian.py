import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def compute_complete_heterogeneous_laplacian(x_dict, edge_index_dict, device, 
                                           node_order=['county', 'case'],
                                           edge_weights=None):
    node_counts = {}
    node_mapping = {}
    total_nodes = 0
    
    for node_type in node_order:
        if node_type in x_dict and x_dict[node_type] is not None:
            count = x_dict[node_type].size(0)
            node_counts[node_type] = count
            node_mapping[node_type] = (total_nodes, total_nodes + count)
            total_nodes += count
        else:
            logger.warning(f"Node type '{node_type}' not found in x_dict")
            node_counts[node_type] = 0
            node_mapping[node_type] = (total_nodes, total_nodes)
    
    if total_nodes == 0:
        logger.error("No valid nodes found for heterogeneous Laplacian computation")
        return None, None, 0
    
    adj_matrix = torch.zeros((total_nodes, total_nodes), device=device, dtype=torch.float32)
    edge_type_weights = edge_weights if edge_weights is not None else {}
    
    for edge_key, edge_index in edge_index_dict.items():
        if edge_index is None or edge_index.size(1) == 0:
            continue
            
        src_type, edge_type, dst_type = edge_key

        if src_type not in node_mapping or dst_type not in node_mapping:
            logger.warning(f"Unknown node types in edge {edge_key}")
            continue
            
        src_offset = node_mapping[src_type][0]
        dst_offset = node_mapping[dst_type][0]
        src_max = node_mapping[src_type][1]
        dst_max = node_mapping[dst_type][1]
        
        src_nodes = edge_index[0] + src_offset
        dst_nodes = edge_index[1] + dst_offset
        
        valid_mask = (src_nodes < src_max) & (dst_nodes < dst_max) & (src_nodes >= src_offset) & (dst_nodes >= dst_offset)
        if not valid_mask.all():
            logger.warning(f"Invalid edges found in {edge_key}, filtering {(~valid_mask).sum().item()} edges")
            src_nodes = src_nodes[valid_mask]
            dst_nodes = dst_nodes[valid_mask]
        
        if src_nodes.size(0) == 0:
            continue
        
        edge_weight = edge_type_weights.get(edge_key, 1.0)
        adj_matrix[src_nodes, dst_nodes] = edge_weight
        if src_type == dst_type and edge_type in ['spatial', 'genetic']:
            adj_matrix[dst_nodes, src_nodes] = edge_weight
    
    degree_vector = torch.sum(adj_matrix, dim=1)
    degree_matrix = torch.diag(degree_vector)
    laplacian_matrix = degree_matrix - adj_matrix
    
    return laplacian_matrix, node_mapping, total_nodes


def build_projection_matrix(node_mapping, case_to_county_edges, device):
    if 'county' not in node_mapping or 'case' not in node_mapping:
        logger.error("Missing county or case nodes for projection matrix")
        return None, None
    
    county_start, county_end = node_mapping['county']
    case_start, case_end = node_mapping['case']
    num_counties = county_end - county_start
    num_cases = case_end - case_start
    total_nodes = county_end + (case_end - case_start) if case_end > county_end else county_end
    P = torch.zeros((total_nodes, num_counties), device=device, dtype=torch.float32)
    county_indices = torch.arange(county_start, county_end, device=device)
    P[county_indices, torch.arange(num_counties, device=device)] = 1.0
    
    if case_to_county_edges is not None and case_to_county_edges.size(1) > 0:
        case_nodes = case_to_county_edges[0] + case_start
        county_targets = case_to_county_edges[1]
        
        valid_case_mask = (case_nodes >= case_start) & (case_nodes < case_end)
        valid_county_mask = (county_targets >= 0) & (county_targets < num_counties)
        valid_mask = valid_case_mask & valid_county_mask

        valid_case_nodes = case_nodes[valid_mask]
        valid_county_targets = county_targets[valid_mask]
        P[valid_case_nodes, valid_county_targets] = 1.0
    return P, county_indices


def compute_spectral_loss_l2_norm(L_hetero, L_fusion, P=None, device='cpu'):
    if L_hetero is None or L_fusion is None:
        return torch.tensor(0.0, device=device, requires_grad=True)

    diff_matrix = L_hetero - L_fusion
    l2_norm = torch.norm(diff_matrix, p='fro')
    
    hetero_norm = torch.norm(L_hetero, p='fro')
    if hetero_norm > 1e-8:
        normalized_loss = l2_norm / hetero_norm
    else:
        normalized_loss = l2_norm
    
    return normalized_loss


def compute_enhanced_spectral_loss(L_hetero, L_fusion, P=None, k=10, alpha=0.7, device='cpu'):
    l2_loss = compute_spectral_loss_l2_norm(L_hetero, L_fusion, P, device)
    
    eigenvalue_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if P is not None:
        P_L_fusion = torch.matmul(P, L_fusion)
        projected_L_fusion = torch.matmul(P_L_fusion, P.t())
        comparison_matrix = projected_L_fusion
    else:
        comparison_matrix = L_fusion
        
    if comparison_matrix.shape == L_hetero.shape and L_hetero.size(0) >= 2:
        eigvals_hetero = torch.linalg.eigvalsh(L_hetero.float())
        eigvals_fusion = torch.linalg.eigvalsh(comparison_matrix.float())
        
        eigvals_hetero_sorted, _ = torch.sort(eigvals_hetero)
        eigvals_fusion_sorted, _ = torch.sort(eigvals_fusion)
        
        k_eff = min(k, len(eigvals_hetero_sorted), len(eigvals_fusion_sorted))
        if k_eff > 1:
            vec_hetero = eigvals_hetero_sorted[1:k_eff]
            vec_fusion = eigvals_fusion_sorted[1:k_eff]
            eigenvalue_loss = 1.0 - F.cosine_similarity(vec_hetero, vec_fusion, dim=0, eps=1e-8)
            
    combined_loss = alpha * l2_loss + (1 - alpha) * eigenvalue_loss
    
    return combined_loss
