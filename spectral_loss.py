import torch
import logging
from complete_hetero_laplacian import compute_complete_heterogeneous_laplacian, build_projection_matrix, compute_enhanced_spectral_loss
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_complete_heterogeneous_laplacian_wrapper(x_dict, edge_index_dict, device):

    L_hetero, node_mapping, total_nodes = compute_complete_heterogeneous_laplacian(
        x_dict, edge_index_dict, device, node_order=['county', 'case'])
    P = None
    if ('case', 'belongs_to', 'county') in edge_index_dict:
        case_to_county_edges = edge_index_dict[('case', 'belongs_to', 'county')]
        P, county_indices = build_projection_matrix(node_mapping, case_to_county_edges, device)
    
    return L_hetero, P, node_mapping


def compute_fusion_laplacian_learned(fusion_features, fusion_edge_index, device):
    num_fusion_nodes = fusion_features.size(0)
    if num_fusion_nodes == 0 or fusion_edge_index.size(1) == 0:
        logger.warning("No fusion nodes or learned edges for Fusion Laplacian.")
        return torch.eye(num_fusion_nodes, device=device) if num_fusion_nodes > 0 else None

    adj = torch.zeros((num_fusion_nodes, num_fusion_nodes), device=device)
    src, dst = fusion_edge_index
    valid_mask = (src < num_fusion_nodes) & (dst < num_fusion_nodes)
    src, dst = src[valid_mask], dst[valid_mask]
    adj[src, dst] = 1
    deg = torch.diag(torch.sum(adj, dim=1))
    laplacian = deg - adj
    return laplacian


def compute_eigenvalue_constrained_spectral_loss(L_het, L_fus, k=10, device='cpu', 
                                               loss_type='cosine_similarity'):
    if L_het is None or L_fus is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    if L_het.size(0) < 2 or L_fus.size(0) < 2:
        logger.debug("Matrix size too small for eigenvalue computation")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    eigvals_het = torch.linalg.eigvalsh(L_het.float())
    
    stabilization_term = 1e-6
    L_fus_stabilized = L_fus + stabilization_term * torch.eye(L_fus.size(0), device=L_fus.device, dtype=L_fus.dtype)
    eigvals_fus = torch.linalg.eigvalsh(L_fus_stabilized.float())
    
    eigvals_het_sorted, _ = torch.sort(eigvals_het)
    eigvals_fus_sorted, _ = torch.sort(eigvals_fus)
    
    k_eff = min(k, len(eigvals_het_sorted), len(eigvals_fus_sorted))
    if k_eff <= 1:
        logger.debug(f"k_eff ({k_eff}) too small, using all available eigenvalues")
        k_eff = min(len(eigvals_het_sorted), len(eigvals_fus_sorted))
        if k_eff <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    start_idx = 1 if k_eff > 1 else 0
    vec_het = eigvals_het_sorted[start_idx:start_idx + k_eff - (1 if k_eff > 1 else 0)]
    vec_fus = eigvals_fus_sorted[start_idx:start_idx + k_eff - (1 if k_eff > 1 else 0)]
    
    min_len = min(len(vec_het), len(vec_fus))
    if min_len == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    vec_het = vec_het[:min_len]
    vec_fus = vec_fus[:min_len]
    
    loss = 1.0 - torch.nn.functional.cosine_similarity(vec_het, vec_fus, dim=0, eps=1e-8)
    
    return loss


def compute_spectral_loss(L_het, L_fus, k=10, device='cpu', P=None, use_l2_norm=False, 
                         use_eigenvalue_constraint=True, eigenvalue_loss_type='cosine_similarity'):
    if L_het is None or L_fus is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return compute_eigenvalue_constrained_spectral_loss(
        L_het, L_fus, k, device, eigenvalue_loss_type)

