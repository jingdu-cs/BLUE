import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_scatter import scatter
from MRF import MRFCorrection
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch.nn.utils import parametrizations
import torch.nn.functional as F


def apply_spectral_norm_dynamic(weight, u=None, eps=1e-12, n_power_iterations=1):
    if u is None:
        u = F.normalize(torch.randn(height, device=weight.device, dtype=weight.dtype), dim=0, eps=eps)
    
    with torch.no_grad():
        for _ in range(n_power_iterations):
            # v = W^T @ u / ||W^T @ u||
            v = F.normalize(weight.t() @ u, dim=0, eps=eps)
            # u = W @ v / ||W @ v||
            u = F.normalize(weight @ v, dim=0, eps=eps)
        
        sigma = torch.dot(u, weight @ v)
        
        if sigma.item() < 0:
            u = -u
            sigma = -sigma
    
    normalized_weight = weight / (sigma + eps)
    
    return normalized_weight, u.detach()

class TransformerGateNetwork(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.5, use_edge_features=True):
        super(TransformerGateNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features
        
        self.edge_type_embedding = nn.Embedding(2, hidden_dim)

        self.edge_feat_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.node_pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.num_layers = num_layers
        if num_layers > 1:
            self.additional_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim, 
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_layers - 1)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
            ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        if not use_edge_features:
            self.node_based_weight_generator = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, node_i_feat, node_j_feat, edge_features, edge_types, padding_mask=None):
        if len(edge_features) == 0:
            return torch.empty(0, 1, device=node_i_feat.device)
        
        device = node_i_feat.device
        num_edges = len(edge_features)
        
        if not self.use_edge_features:
            node_pair = torch.cat([node_i_feat, node_j_feat])
            
            unique_types = torch.unique(torch.tensor(edge_types, device=device))
            edge_weights = torch.zeros(num_edges, 1, device=device)
            
            for edge_type in unique_types:
                type_indices = torch.tensor([i for i, t in enumerate(edge_types) if t == edge_type.item()], 
                                            device=device)
                if len(type_indices) == 0:
                    continue
                    
                type_weight = self.node_based_weight_generator(node_pair.unsqueeze(0))
                edge_weights.index_fill_(0, type_indices, type_weight)
            
            if not padding_mask.all():
                edge_weights = edge_weights.masked_fill(padding_mask.unsqueeze(1) == 0, 0)
            
            valid_sum = edge_weights.sum()
            if valid_sum > 0:
                edge_weights = edge_weights / valid_sum
            
            return edge_weights
        
        edge_features = torch.stack(edge_features)
        edge_types = torch.tensor(edge_types, device=edge_features.device)
        
        type_embeddings = self.edge_type_embedding(edge_types)
        edge_inputs = edge_features + type_embeddings
        edge_inputs = self.edge_feat_proj(edge_inputs)  # [num_edges, hidden_dim]
        
        node_pair = torch.cat([node_i_feat, node_j_feat]).unsqueeze(0)  # [1, hidden_dim*2]
        query = self.node_pair_proj(node_pair)  # [1, hidden_dim]

        query = query.unsqueeze(0)  # [1, 1, hidden_dim]
        key_value = edge_inputs.unsqueeze(0)  # [1, num_edges, hidden_dim]
        
        if padding_mask is None:
            padding_mask = torch.ones(edge_inputs.size(0), device=edge_inputs.device)
        
        attn_mask = ~padding_mask.bool()  # [num_edges]
        attn_mask = attn_mask.unsqueeze(0) if attn_mask.any() else None  # [1, num_edges]
        
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=attn_mask
        )
        
        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                layer_output, layer_weights = self.additional_layers[i](
                    query=attn_output,
                    key=key_value,
                    value=key_value,
                    key_padding_mask=attn_mask
                )
                attn_output = self.layer_norms[i](layer_output)
                attn_weights = layer_weights
        
        edge_weights = attn_weights.squeeze(0).squeeze(0).unsqueeze(-1)  # [num_edges, 1]
        
        if not padding_mask.all():
            edge_weights = edge_weights.masked_fill(
                padding_mask.unsqueeze(1) == 0, 0)
        
        return edge_weights


class TransformerTemporalFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1, max_len=20):
        super(TransformerTemporalFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.position_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))
        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.single_step_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, features_sequence, attention_mask=None):
        batch_size, seq_len, _ = features_sequence.shape
        device = features_sequence.device
        
        if seq_len == 1:
            return self.single_step_proj(features_sequence.squeeze(1))
        
        position_encodings = self.position_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        features = features_sequence + position_encodings
        
        features = self.feature_proj(features)
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        transformer_mask = ~attention_mask
        
        encoded_features = self.transformer_encoder(
            features, 
            src_key_padding_mask=transformer_mask if transformer_mask.any() else None
        )
        
        if attention_mask.any():
            last_valid_indices = attention_mask.sum(dim=1, dtype=torch.long) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            
            batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)
            last_features = encoded_features[batch_indices, last_valid_indices]
        else:
            last_features = encoded_features[:, -1]
        
        output = self.output_proj(last_features)
        
        return output


class FusionGraphBuilder(nn.Module):
    def __init__(self, hidden_dim, link_threshold=0.5, use_top_k=False, top_k=10):
        super(FusionGraphBuilder, self).__init__()
        self.hidden_dim = hidden_dim
        self.link_threshold = link_threshold
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.county_indices_last = None

        self.case_fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.county_fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_combiner = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.transformer_gate = TransformerGateNetwork(hidden_dim)
        
        self.n_projections = 10
        self.proj_dim = hidden_dim
        self.projections = None
    
    def _init_lsh_projections(self, device):
        if self.projections is None:
            self.projections = torch.randn(self.n_projections, self.proj_dim, device=device)
            self.projections = F.normalize(self.projections, p=2, dim=1)
            
    def _compute_lsh_hash(self, features):
        self._init_lsh_projections(features.device)
        projections = torch.matmul(features, self.projections.t())
        hash_codes = (projections > 0).float()
        return hash_codes

    def generate_fusion_nodes(self, x_dict, edge_index_dict):
        if 'county' not in x_dict:
            raise ValueError("The graph must contain 'county' nodes")
            
        county_features = x_dict['county']
        num_counties = county_features.size(0)
        county_indices = torch.arange(num_counties, device=county_features.device)
        
        county_spatial_features = self._aggregate_neighbors(x_dict, edge_index_dict, 'county', 'spatial')
        if county_spatial_features is None:
            county_spatial_features = torch.zeros_like(county_features)
        
        enhanced_county_features = torch.cat([county_features, county_spatial_features], dim=1)
        
        if 'case' in x_dict and ('case', 'belongs_to', 'county') in edge_index_dict:
            belongs_to_edges = edge_index_dict[('case', 'belongs_to', 'county')]
            case_indices = belongs_to_edges[0]
            county_targets = belongs_to_edges[1]
            
            case_features = x_dict['case']
            
            case_genetic_features = self._aggregate_neighbors(x_dict, edge_index_dict, 'case', 'genetic')
            if case_genetic_features is None:
                case_genetic_features = torch.zeros_like(case_features)
            
            enhanced_case_features = torch.cat([case_features, case_genetic_features], dim=1)
            enhanced_case_features = self.case_fusion_proj(enhanced_case_features)
            
            county_from_cases_features = torch.zeros(num_counties, self.hidden_dim, device=county_features.device)
            county_case_counts = torch.zeros(num_counties, 1, device=county_features.device)
            scatter(enhanced_case_features, county_targets, dim=0, dim_size=num_counties, reduce="add", out=county_from_cases_features)
            
            ones = torch.ones(case_indices.size(0), 1, device=county_features.device)
            scatter(ones, county_targets, dim=0, dim_size=num_counties, reduce="add", out=county_case_counts)
            
            county_case_counts[county_case_counts == 0] = 1
            county_from_cases_features = county_from_cases_features / county_case_counts
            enhanced_county_features = self.county_fusion_proj(enhanced_county_features)
            
            fusion_node_features = torch.cat([enhanced_county_features, county_from_cases_features], dim=1)
            fusion_node_features = self.fusion_combiner(fusion_node_features)
        else:
            fusion_node_features = self.county_fusion_proj(enhanced_county_features)
        
        fusion_mapping = {idx.item(): idx.item() for idx in county_indices}
        self.county_indices_last = county_indices
        case_indices = torch.tensor([], dtype=torch.long, device=county_features.device)
        return fusion_node_features, fusion_mapping, case_indices, county_indices
    
    def _aggregate_neighbors(self, x_dict, edge_index_dict, node_type, edge_type=None):
        if node_type not in x_dict:
            return None
            
        node_features = x_dict[node_type]
        num_nodes = node_features.size(0)
        device = node_features.device
        
        aggregated_features = torch.zeros_like(node_features)
        
        neighbor_counts = torch.zeros(num_nodes, 1, device=device)
        
        for (src, rel, dst) in edge_index_dict.keys():
            if dst == node_type and (edge_type is None or rel == edge_type):
                edge_index = edge_index_dict[(src, rel, dst)]
                src_features = x_dict[src]
                
                dst_indices = edge_index[1]
                src_indices = edge_index[0]
                
                ones = torch.ones(dst_indices.size(0), 1, device=device)
                scatter(ones, dst_indices, dim=0, dim_size=num_nodes, reduce="add", out=neighbor_counts)
                scatter(src_features[src_indices], dst_indices, dim=0, dim_size=num_nodes, reduce="add", out=aggregated_features)
        
        safe_divisor = neighbor_counts.clone()
        safe_divisor[safe_divisor == 0] = 1.0
        
        normalized_features = aggregated_features / safe_divisor
        
        return normalized_features
    
    def build_fusion_graph(self, fusion_features, fusion_mapping, case_indices, county_indices, 
                          original_edge_index_dict):
        num_fusion_nodes = fusion_features.size(0)
        device = fusion_features.device
        
        if num_fusion_nodes <= 100:
            rows = torch.arange(num_fusion_nodes, device=device)
            cols = torch.arange(num_fusion_nodes, device=device)
            row_idx, col_idx = torch.meshgrid(rows, cols, indexing='ij')
            mask = row_idx < col_idx
            
            src_indices = row_idx[mask]
            dst_indices = col_idx[mask]
            src_features = fusion_features[src_indices]
            dst_features = fusion_features[dst_indices]
            pair_features = torch.cat([src_features, dst_features], dim=1)
            link_probs = self.link_predictor(pair_features)
            fusion_edge_scores = link_probs.squeeze(-1)
            candidate_edges = torch.stack([src_indices, dst_indices], dim=1)

        else:
            max_candidate_pairs = min(10000, num_fusion_nodes * 20)
            candidate_edges = self._find_candidate_pairs_lsh(
                fusion_features, max_pairs=max_candidate_pairs)
            
            if candidate_edges.size(0) > 0:
                batch_size = 1024
                all_link_probs = []
                for start_idx in range(0, candidate_edges.size(0), batch_size):
                    end_idx = min(start_idx + batch_size, candidate_edges.size(0))
                    batch_edges = candidate_edges[start_idx:end_idx]
                    
                    src_nodes, dst_nodes = batch_edges[:, 0], batch_edges[:, 1]
                    src_features = fusion_features[src_nodes]
                    dst_features = fusion_features[dst_nodes]
                    
                    pair_features = torch.cat([src_features, dst_features], dim=1)
                    link_probs = self.link_predictor(pair_features)
                    all_link_probs.append(link_probs)
                fusion_edge_scores = torch.cat(all_link_probs, dim=0).squeeze(-1)

            else:
                fusion_edge_scores = torch.tensor([], device=device)
        
        if fusion_edge_scores.numel() > 0:
            if self.use_top_k:
                if fusion_edge_scores.size(0) <= self.top_k:
                    selected_mask = torch.ones_like(fusion_edge_scores, dtype=torch.bool)
                else:
                    _, top_indices = torch.topk(fusion_edge_scores, self.top_k)
                    selected_mask = torch.zeros_like(fusion_edge_scores, dtype=torch.bool)
                    selected_mask[top_indices] = True
            else:
                selected_mask = fusion_edge_scores >= self.link_threshold
            
            if selected_mask.any():
                selected_edges = candidate_edges[selected_mask]
                
                src_nodes, dst_nodes = selected_edges[:, 0], selected_edges[:, 1]
                edge_index = torch.stack([
                    torch.cat([src_nodes, dst_nodes]),
                    torch.cat([dst_nodes, src_nodes])
                ])
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        return edge_index
    
    def transform_hetero_edges_to_fusion(self, edge_index_dict, fusion_mapping, case_indices, county_indices, fusion_features=None):
        
        device = county_indices.device
        num_counties = len(county_indices)
        
        county_to_fusion = torch.zeros(num_counties, dtype=torch.long, device=device) - 1
        for old_idx, new_idx in fusion_mapping.items():
            county_to_fusion[old_idx] = new_idx
            
        spatial_county_pairs = []
        genetic_county_pairs = []
        
        if ('county', 'spatial', 'county') in edge_index_dict:
            spatial_edges = edge_index_dict[('county', 'spatial', 'county')]
            
            valid_src_mask = county_to_fusion[spatial_edges[0]] >= 0
            valid_dst_mask = county_to_fusion[spatial_edges[1]] >= 0
            valid_mask = valid_src_mask & valid_dst_mask
            
            if valid_mask.any():
                valid_src = spatial_edges[0, valid_mask]
                valid_dst = spatial_edges[1, valid_mask]
                
                fusion_src = county_to_fusion[valid_src]
                fusion_dst = county_to_fusion[valid_dst]
                
                edge_min = torch.min(fusion_src, fusion_dst)
                edge_max = torch.max(fusion_src, fusion_dst)
                spatial_edges_tensor = torch.stack([edge_min, edge_max], dim=1)
                
                spatial_county_pairs = torch.unique(spatial_edges_tensor, dim=0)
                
                spatial_types = torch.zeros(spatial_county_pairs.size(0), dtype=torch.long, device=device)
        
        if ('case', 'genetic', 'case') in edge_index_dict and ('case', 'belongs_to', 'county') in edge_index_dict:
            belongs_to_edges = edge_index_dict[('case', 'belongs_to', 'county')]
            genetic_edges = edge_index_dict[('case', 'genetic', 'case')]
            
            num_cases = torch.max(belongs_to_edges[0]) + 1 if belongs_to_edges.size(1) > 0 else 0
            case_to_county = torch.zeros(num_cases.item(), dtype=torch.long, device=device) - 1
            
            valid_counties_mask = county_to_fusion[belongs_to_edges[1]] >= 0
            if valid_counties_mask.any():
                valid_case_indices = belongs_to_edges[0, valid_counties_mask]
                valid_county_indices = belongs_to_edges[1, valid_counties_mask]
                case_to_county[valid_case_indices] = valid_county_indices
            
            valid_src_case_mask = (genetic_edges[0] < num_cases) & (case_to_county[genetic_edges[0]] >= 0)
            valid_dst_case_mask = (genetic_edges[1] < num_cases) & (case_to_county[genetic_edges[1]] >= 0)
            valid_genetic_mask = valid_src_case_mask & valid_dst_case_mask
            
            if valid_genetic_mask.any():
                valid_src_cases = genetic_edges[0, valid_genetic_mask]
                valid_dst_cases = genetic_edges[1, valid_genetic_mask]
                
                src_counties = case_to_county[valid_src_cases]
                dst_counties = case_to_county[valid_dst_cases]
                
                different_counties_mask = src_counties != dst_counties
                
                if different_counties_mask.any():
                    src_counties = src_counties[different_counties_mask]
                    dst_counties = dst_counties[different_counties_mask]
                    
                    fusion_src = county_to_fusion[src_counties]
                    fusion_dst = county_to_fusion[dst_counties]
                    
                    edge_min = torch.min(fusion_src, fusion_dst)
                    edge_max = torch.max(fusion_src, fusion_dst)
                    genetic_edges_tensor = torch.stack([edge_min, edge_max], dim=1)
                    
                    genetic_county_pairs = torch.unique(genetic_edges_tensor, dim=0)
                    
                    genetic_types = torch.ones(genetic_county_pairs.size(0), dtype=torch.long, device=device)
        
        if len(spatial_county_pairs) > 0 and len(genetic_county_pairs) > 0:
            all_county_pairs = torch.cat([spatial_county_pairs, genetic_county_pairs], dim=0)
            all_edge_types = torch.cat([spatial_types, genetic_types])
            
            unique_county_pairs, inverse_indices = torch.unique(all_county_pairs, dim=0, return_inverse=True)
            
            if fusion_features is not None:
                num_edges = unique_county_pairs.size(0)
                if num_edges == 0:
                    valid_county_pairs = unique_county_pairs
                else:
                    valid_edges_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
                    
                    type_counts = torch.zeros(num_edges, dtype=torch.long, device=device)
                    ones = torch.ones(inverse_indices.size(0), dtype=torch.long, device=device)
                    type_counts.scatter_add_(0, inverse_indices, ones)
                    
                    single_type_mask = type_counts == 1
                    valid_edges_mask[single_type_mask] = True
                    
                    multi_type_idx = torch.where(~single_type_mask)[0]
                    
                    if multi_type_idx.size(0) > 0:
                        batch_size = multi_type_idx.size(0)
                        
                        for start_idx in range(0, multi_type_idx.size(0), batch_size):
                            end_idx = min(start_idx + batch_size, multi_type_idx.size(0))
                            batch_indices = multi_type_idx[start_idx:end_idx]
                            
                            for idx in batch_indices:
                                county_i, county_j = unique_county_pairs[idx]
                                
                                edge_mask = (all_county_pairs[:, 0] == county_i) & (all_county_pairs[:, 1] == county_j)
                                edge_indices = torch.where(edge_mask)[0]
                                
                                edge_types_list = all_edge_types[edge_indices]
                                
                                node_i_feat = fusion_features[county_i]
                                node_j_feat = fusion_features[county_j]
                                
                                edge_features = [torch.zeros(self.hidden_dim, device=device) for _ in range(edge_indices.size(0))]
                                
                                edge_weights = self.transformer_gate(node_i_feat, node_j_feat, edge_features, edge_types_list.tolist())
                                
                                if edge_weights.size(0) > 0 and edge_weights.max().item() > 0.1:
                                    valid_edges_mask[idx] = True
                    
                    valid_county_pairs = unique_county_pairs[valid_edges_mask]
            else:
                valid_county_pairs = unique_county_pairs
            
            if valid_county_pairs.size(0) > 0:
                src_nodes, dst_nodes = valid_county_pairs[:, 0], valid_county_pairs[:, 1]
                
                all_src = torch.cat([src_nodes, dst_nodes])
                all_dst = torch.cat([dst_nodes, src_nodes])
                
                fusion_edge_dict = {
                    ('fusion', 'fused', 'fusion'): torch.stack([all_src, all_dst])
                }
            else:
                fusion_edge_dict = {
                    ('fusion', 'fused', 'fusion'): torch.zeros((2, 0), dtype=torch.long, device=device)
                }
        elif len(spatial_county_pairs) > 0:
            src_nodes, dst_nodes = spatial_county_pairs[:, 0], spatial_county_pairs[:, 1]
            all_src = torch.cat([src_nodes, dst_nodes])
            all_dst = torch.cat([dst_nodes, src_nodes])
            fusion_edge_dict = {
                ('fusion', 'fused', 'fusion'): torch.stack([all_src, all_dst])
            }
        elif len(genetic_county_pairs) > 0:
            src_nodes, dst_nodes = genetic_county_pairs[:, 0], genetic_county_pairs[:, 1]
            all_src = torch.cat([src_nodes, dst_nodes])
            all_dst = torch.cat([dst_nodes, src_nodes])
            fusion_edge_dict = {
                ('fusion', 'fused', 'fusion'): torch.stack([all_src, all_dst])
            }
        else:
            fusion_edge_dict = {
                ('fusion', 'fused', 'fusion'): torch.zeros((2, 0), dtype=torch.long, device=device)
            }
        
        return fusion_edge_dict
    
    def forward(self, x_dict, edge_index_dict):
        fusion_features, fusion_mapping, case_indices, county_indices = self.generate_fusion_nodes(
            x_dict, edge_index_dict)
        
        fusion_edge_index = self.build_fusion_graph(
            fusion_features, fusion_mapping, case_indices, county_indices, edge_index_dict)
        
        fusion_hetero_edges = self.transform_hetero_edges_to_fusion(
            edge_index_dict, fusion_mapping, case_indices, county_indices, fusion_features)
        
        return fusion_features, fusion_edge_index, fusion_hetero_edges

    def _find_candidate_pairs_lsh(self, fusion_features, max_pairs=None):
        
        num_nodes = fusion_features.size(0)
        device = fusion_features.device
        
        if num_nodes <= 100:
            rows = torch.arange(num_nodes, device=device)
            cols = torch.arange(num_nodes, device=device)
            row_idx, col_idx = torch.meshgrid(rows, cols, indexing='ij')
            mask = row_idx < col_idx
            i_indices = row_idx[mask]
            j_indices = col_idx[mask]
            all_pairs = torch.stack([i_indices, j_indices], dim=1)
            
            if max_pairs is not None and all_pairs.size(0) > max_pairs:
                perm = torch.randperm(all_pairs.size(0), device=device)[:max_pairs]
                return all_pairs[perm]
                
            return all_pairs
        
        actual_max_pairs = min(max_pairs or 10000, num_nodes * 10)
        
        batch_size = min(1024, num_nodes)
        num_batches = (num_nodes + batch_size - 1) // batch_size
        hash_codes_list = []
        
        self._init_lsh_projections(device)
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_nodes)
                batch_features = fusion_features[start_idx:end_idx]
                
                projections = torch.matmul(batch_features, self.projections.t())
                batch_hash_codes = (projections > 0).to(torch.int8)
                hash_codes_list.append(batch_hash_codes)
            
            hash_codes = torch.cat(hash_codes_list, dim=0)
        
        hash_values = torch.zeros(num_nodes, dtype=torch.int32, device=device)
        for i in range(self.n_projections):
            hash_values = hash_values * 2 + hash_codes[:, i]
        
        sorted_hashes, sort_indices = torch.sort(hash_values)
        
        same_hash = sorted_hashes[1:] == sorted_hashes[:-1]
        
        edge_list = []
        
        if same_hash.any():
            is_start = torch.cat([torch.tensor([True], device=device),
                                 same_hash[1:] != same_hash[:-1]])
            group_starts = torch.where(is_start)[0]
            group_ends = torch.cat([group_starts[1:], torch.tensor([len(same_hash)], device=device)])
            
            for start_idx, end_idx in zip(group_starts.cpu().tolist(), group_ends.cpu().tolist()):
                group_node_indices = sort_indices[start_idx:end_idx+1]
                num_pairs = len(group_node_indices) * (len(group_node_indices) - 1) // 2
                if num_pairs > 10000:
                    continue
                    
                rows = torch.arange(len(group_node_indices), device=device)
                cols = torch.arange(len(group_node_indices), device=device)
                row_idx, col_idx = torch.meshgrid(rows, cols, indexing='ij')
                mask = row_idx < col_idx
                
                if mask.any():
                    local_i = row_idx[mask]
                    local_j = col_idx[mask]
                    global_i = group_node_indices[local_i]
                    global_j = group_node_indices[local_j]
                    
                    new_edges = torch.stack([global_i, global_j], dim=1)
                    edge_list.append(new_edges)
                    
                    total_edges = sum(e.size(0) for e in edge_list)
                    if total_edges >= actual_max_pairs:
                        break
        
        total_edges = sum(e.size(0) for e in edge_list) if edge_list else 0
        
        if total_edges < min(actual_max_pairs, num_nodes * 2) and self.n_projections > 2:
            match_threshold = 1
                        
            if num_nodes > 100:
                src_sample = torch.randperm(num_nodes, device=device)[:100]
                dst_range = torch.arange(min(20, num_nodes), device=device)
                
                for src_idx in src_sample:
                    dst_sample = (src_idx + dst_range + 1) % num_nodes
                    src_hash = hash_codes[src_idx:src_idx+1]
                    dst_hash = hash_codes[dst_sample]
                    hamming_dist = torch.sum(torch.abs(src_hash - dst_hash), dim=1)
                    valid_dst = dst_sample[hamming_dist <= match_threshold]
                    
                    if valid_dst.size(0) > 0:
                        src_expanded = src_idx.expand(valid_dst.size(0))
                        new_edges = torch.stack([src_expanded, valid_dst], dim=1)
                        mask = new_edges[:, 0] < new_edges[:, 1]
                        if mask.any():
                            new_edges = new_edges[mask]
                            edge_list.append(new_edges)
                        total_edges = sum(e.size(0) for e in edge_list)
                        if total_edges >= actual_max_pairs:
                            break
        
        if edge_list:
            candidate_edges = torch.cat(edge_list, dim=0)
            candidate_edges = torch.unique(candidate_edges, dim=0)
            
            if max_pairs is not None and candidate_edges.size(0) > max_pairs:
                perm = torch.randperm(candidate_edges.size(0), device=device)[:max_pairs]
                candidate_edges = candidate_edges[perm]
        else:
            candidate_edges = torch.zeros((0, 2), dtype=torch.long, device=device)
            
        return candidate_edges


class FusionGNN(nn.Module):

    def __init__(self, channel=2, hidden_dim=64, num_layers=2, dropout=0.3, pred_horizon=4,
                 link_threshold=0.5, use_top_k=False, top_k=10, num_mrf=3, device=None,
                 use_transformer_temporal=True, transformer_heads=4, transformer_layers=2,
                 county_input_dim=None, previous_weight=0.15, initial_weight=0.1):
        super(FusionGNN, self).__init__()
        
        self.previous_weight = previous_weight
        self.initial_weight = initial_weight
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.num_mrf = num_mrf
        self.num_layers = num_layers
        self.device = device
        self.use_transformer_temporal = use_transformer_temporal

        self.node_types = ['county', 'case']
        self.edge_types = [
            ('county', 'spatial', 'county'),
            ('case', 'genetic', 'case'),
            ('case', 'belongs_to', 'county'),
            ('county', 'contains', 'case')
        ]
        
        self.mrf_correction = MRFCorrection(
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_dim=hidden_dim,
            num_iterations=self.num_mrf
        ).to(self.device)
        
        self.fusion_builder = FusionGraphBuilder(
            hidden_dim=hidden_dim,
            link_threshold=link_threshold,
            use_top_k=use_top_k,
            top_k=top_k
        ).to(self.device)
        
        self.county_input_dim = county_input_dim
        self.county_encoder = parametrizations.weight_norm(nn.Linear(self.county_input_dim, hidden_dim)).to(self.device)
        self.case_encoder = parametrizations.weight_norm(nn.Linear(1, hidden_dim)).to(self.device)
        
        self.encoder_convs = nn.ModuleList()
        self.encoder_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        for _ in range(num_layers):
            conv = SAGEConv(hidden_dim, hidden_dim).to(self.device)
            self.encoder_convs.append(conv)
        
        self.decoder_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_dim, hidden_dim).to(self.device)
            self.decoder_convs.append(conv)
        
        self.bottleneck = parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim)).to(self.device)
        
        self.prediction_head = nn.Sequential(
            parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            parametrizations.weight_norm(nn.Linear(hidden_dim, channel))
        ).to(self.device)
        
        self.prediction_to_feature = parametrizations.weight_norm(nn.Linear(channel, hidden_dim)).to(self.device)
        
        if use_transformer_temporal:
            self.temporal_integration = TransformerTemporalFusion(
                hidden_dim=hidden_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
                max_len=20
            ).to(self.device)
        else:
            self.temporal_integration = nn.GRU(hidden_dim, hidden_dim, batch_first=True).to(self.device)
        
        self.year_encoder = None
        self.week_encoder = None
        self.temporal_proj = parametrizations.weight_norm(nn.Linear(hidden_dim * 3, hidden_dim)).to(self.device)
        self.proj = parametrizations.weight_norm(nn.Linear(hidden_dim * 2, hidden_dim)).to(self.device)
        
        self.dropout = nn.Dropout(dropout).to(self.device)
        
        self.spectral_u_buffers = nn.ParameterDict()
        for i in range(num_layers):
            self.spectral_u_buffers[f'encoder_{i}_lin_l_u'] = nn.Parameter(
                torch.randn(hidden_dim), requires_grad=False)
            self.spectral_u_buffers[f'encoder_{i}_lin_r_u'] = nn.Parameter(
                torch.randn(hidden_dim), requires_grad=False)
            self.spectral_u_buffers[f'decoder_{i}_lin_l_u'] = nn.Parameter(
                torch.randn(hidden_dim), requires_grad=False)
            self.spectral_u_buffers[f'decoder_{i}_lin_r_u'] = nn.Parameter(
                torch.randn(hidden_dim), requires_grad=False)

    def apply_spectral_norm_to_conv(self, conv, layer_idx, conv_type='encoder'):
        if hasattr(conv, 'lin_l') and conv.lin_l is not None:
            u_key = f'{conv_type}_{layer_idx}_lin_l_u'
            if u_key in self.spectral_u_buffers:
                weight = conv.lin_l.weight
                u = self.spectral_u_buffers[u_key]
                normalized_weight, new_u = apply_spectral_norm_dynamic(weight, u)
                conv.lin_l.weight.data = normalized_weight
                self.spectral_u_buffers[u_key].data = new_u
        
        if hasattr(conv, 'lin_r') and conv.lin_r is not None:
            u_key = f'{conv_type}_{layer_idx}_lin_r_u'
            if u_key in self.spectral_u_buffers:
                weight = conv.lin_r.weight
                u = self.spectral_u_buffers[u_key]
                normalized_weight, new_u = apply_spectral_norm_dynamic(weight, u)
                conv.lin_r.weight.data = normalized_weight
                self.spectral_u_buffers[u_key].data = new_u
        
    def encode_heterograph(self, x_dict, edge_index_dict):
        if 'case' in x_dict:
            x_dict['case'] = self.case_encoder(x_dict['case'].unsqueeze(-1) if x_dict['case'].dim() == 1 else x_dict['case'])
        if 'county' in x_dict:
            x_dict['county'] = self.county_encoder(x_dict['county'])
            
        return x_dict
    
    def encoder_process(self, fusion_features, fusion_edge_index, temporal_info=None):
        x = fusion_features
        
        for i, conv in enumerate(self.encoder_convs):
            self.apply_spectral_norm_to_conv(conv, i, 'encoder')
            
            if fusion_edge_index.numel() == 0 and x.numel() > 0:
                x = conv(x, fusion_edge_index)
            else:
                x = conv(x, fusion_edge_index)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        if temporal_info is not None:
            if isinstance(temporal_info, list):
                if len(temporal_info) > 0:
                    temp_info = temporal_info[0]
                else:
                    return x
            elif isinstance(temporal_info, tuple) and len(temporal_info) == 2:
                temp_info = temporal_info
            else:
                try:
                    year_encoding, week_encoding = temporal_info
                    temp_info = (year_encoding, week_encoding)
                except:
                    return x
            
            year_encoding, week_encoding = temp_info
            
            if self.year_encoder is None:
                self.year_encoder = nn.Linear(len(year_encoding), self.hidden_dim).to(self.device)
            if self.week_encoder is None:
                self.week_encoder = nn.Linear(len(week_encoding), self.hidden_dim).to(self.device)
                
            year_features = self.year_encoder(year_encoding)
            week_features = self.week_encoder(week_encoding)
            
            combined_features = torch.cat([x, 
                                          year_features.unsqueeze(0).expand(x.size(0), -1), 
                                          week_features.unsqueeze(0).expand(x.size(0), -1)], 
                                         dim=1)
            x = self.temporal_proj(combined_features)
        
        return x
    
    def decoder_process(self, hidden_state, fusion_edge_index, prev_pred_features=None, temporal_info=None):
        x = self.bottleneck(hidden_state)
        
        if prev_pred_features is not None:
            attention_input = torch.cat([x, prev_pred_features], dim=1)
            x = self.proj(attention_input)
        
        for i, conv in enumerate(self.decoder_convs):
            self.apply_spectral_norm_to_conv(conv, i, 'decoder')
            
            if isinstance(fusion_edge_index, dict):
                x_dict = {}
                x_dict['fusion'] = x
                
                fusion_out = torch.zeros_like(x)
                edge_count = torch.zeros(x.size(0), 1, device=x.device)
                
                for edge_type, edge_index in fusion_edge_index.items():
                    type_out = conv((x, x), edge_index)
                    
                    dst_indices = edge_index[1]
                    
                    ones = torch.ones(dst_indices.size(0), 1, device=x.device)
                    edge_count.index_add_(0, dst_indices, ones)
                    
                    for j in range(type_out.size(1)):
                        fusion_out[:, j].index_add_(0, dst_indices, type_out[edge_index[0], j])
                
                mask = edge_count > 0
                if mask.any():
                    fusion_out[mask.squeeze(-1)] = fusion_out[mask.squeeze(-1)] / edge_count[mask].unsqueeze(-1)
                
                x = F.relu(fusion_out)
                x = self.dropout(x)
            else:
                if fusion_edge_index.numel() == 0 and x.numel() > 0:
                    x = conv(x, fusion_edge_index)
                else:
                    x = conv(x, fusion_edge_index)
                    x = F.relu(x)
                    x = self.dropout(x)
        
        if temporal_info is not None:
            if isinstance(temporal_info, list):
                if len(temporal_info) > 0:
                    temp_info = temporal_info[0]
                else:
                    return x
            elif isinstance(temporal_info, tuple) and len(temporal_info) == 2:
                temp_info = temporal_info
            else:
                try:
                    year_encoding, week_encoding = temporal_info
                    temp_info = (year_encoding, week_encoding)
                except:
                    return x
            
            year_encoding, week_encoding = temp_info
            
            if self.year_encoder is None:
                self.year_encoder = nn.Linear(len(year_encoding), self.hidden_dim).to(self.device)
            if self.week_encoder is None:
                self.week_encoder = nn.Linear(len(week_encoding), self.hidden_dim).to(self.device)
                
            year_features = self.year_encoder(year_encoding)
            week_features = self.week_encoder(week_encoding)
            
            combined_features = torch.cat([x, 
                                           year_features.unsqueeze(0).expand(x.size(0), -1), 
                                           week_features.unsqueeze(0).expand(x.size(0), -1)], 
                                          dim=1)
            x = self.temporal_proj(combined_features)
        
        return x
    
    def single_step_predict(self, node_features):
        return self.prediction_head(node_features)

    def forward(self, sequence, temporal_info=None, pred_temporal_info=None):
        
        device = self.device
        seq_len = len(sequence)                       # T
        batch_size = len(sequence[0])                    # B
        assert all(len(step) == batch_size for step in sequence)

        fused_graphs = []
        bt_index = []
        county_nums = [None]*batch_size
        fusion_edges = [None]*batch_size
        original_node_features = [None]*batch_size

        for t in range(seq_len):
            for b in range(batch_size):
                hetero = sequence[t][b].to(device)
                x_dict = self.encode_heterograph(hetero.x_dict, hetero.edge_index_dict)
                x_dict = self.mrf_correction(x_dict, hetero.edge_index_dict)

                fusion_feat, fusion_ei, fusion_hetero_edges =  self.fusion_builder(x_dict, hetero.edge_index_dict)

                data = Data(x=fusion_feat, edge_index=fusion_ei)
                fused_graphs.append(data)
                bt_index.append((b, t))

                if t == seq_len-1:
                    county_nums[b] = fusion_feat.size(0)
                    fusion_edges[b] = fusion_hetero_edges
                    original_node_features[b] = fusion_feat

        num_graphs = len(fused_graphs)
        big_batch = Batch.from_data_list(fused_graphs).to(device)
        x = big_batch.x
        edge_index = big_batch.edge_index

        x_original = x.clone()
        for i, conv in enumerate(self.encoder_convs):
            self.apply_spectral_norm_to_conv(conv, i, 'encoder')
            
            x_conv = conv(x, edge_index)
            x = x_conv + x
            x = F.relu(x)
            x = self.dropout(x)
            if i % 2 == 1:
                x = x + 0.1 * x_original

        encoded_by_sample = [ [None]*seq_len for _ in range(batch_size) ]
        graph_id_of_node = big_batch.batch

        for g_idx in range(num_graphs):
            b, t = bt_index[g_idx]
            node_mask = (graph_id_of_node == g_idx)
            node_feats = x[node_mask]

            if temporal_info is not None:
                if b < len(temporal_info) and t < len(temporal_info[b]):
                    year_enc, week_enc = temporal_info[b][t]
                    if self.year_encoder is None:
                        self.year_encoder = nn.Linear(len(year_enc), self.hidden_dim).to(device)
                        self.week_encoder = nn.Linear(len(week_enc), self.hidden_dim).to(device)
                    yf = self.year_encoder(year_enc).unsqueeze(0).expand_as(node_feats)
                    wf = self.week_encoder(week_enc).unsqueeze(0).expand_as(node_feats)
                    node_feats = self.temporal_proj(torch.cat([node_feats, yf, wf], dim=1))

            encoded_by_sample[b][t] = node_feats

        hidden_by_sample = []
        for b in range(batch_size):
            seq_tensor = torch.stack(encoded_by_sample[b], dim=1)
            if self.use_transformer_temporal:
                attention_mask = torch.ones(seq_tensor.shape[0], seq_tensor.shape[1], device=device)
                fused = self.temporal_integration(seq_tensor, attention_mask)
                
                if original_node_features[b] is not None:
                    orig_feat = original_node_features[b]
                    if orig_feat.shape[0] == fused.shape[0]:
                        fused = fused + 0.2 * orig_feat
                
                hidden_state = fused
            else:
                batch_nodes = seq_tensor.shape[0]
                hidden_outputs = []
                for node_idx in range(batch_nodes):
                    node_seq = seq_tensor[node_idx:node_idx+1]  
                    node_fused, _ = self.temporal_integration(node_seq)
                    hidden_outputs.append(node_fused[:, -1, :])
                
                hidden_state = torch.cat(hidden_outputs, dim=0)
                
                if original_node_features[b] is not None:
                    orig_feat = original_node_features[b]
                    if orig_feat.shape[0] == hidden_state.shape[0]:
                        hidden_state = hidden_state + 0.2 * orig_feat
            
            hidden_by_sample.append(hidden_state)

        batch_predictions = []
        for b in range(batch_size):
            hidden_state = hidden_by_sample[b]
            fusion_edge_dict = fusion_edges[b]
            preds = []

            node_count = hidden_state.shape[0]
            prev_feats = []
            for node_idx in range(node_count):
                node_hidden = hidden_state[node_idx:node_idx+1]  
                node_pred = self.single_step_predict(node_hidden)
                node_feat = self.prediction_to_feature(node_pred)
                prev_feats.append(node_feat)
            
            prev_feat = torch.cat(prev_feats, dim=0)
            
            initial_hidden = hidden_state.clone()
            
            for step in range(self.pred_horizon):
                if pred_temporal_info is not None and step < len(pred_temporal_info[b]):
                    year_enc, week_enc = pred_temporal_info[b][step]
                    step_time_info = (year_enc, week_enc)
                else:
                    step_time_info = None

                decoded = self.decoder_process(hidden_state,
                                              fusion_edge_dict,
                                              prev_feat,
                                              step_time_info)
                
                decoded = (1 - self.previous_weight - self.initial_weight) * decoded + self.previous_weight * prev_feat + self.initial_weight * initial_hidden
                
                step_predictions = []
                for node_idx in range(node_count):
                    node_decoded = decoded[node_idx:node_idx+1]
                    node_pred = self.single_step_predict(node_decoded)
                    step_predictions.append(node_pred)
                
                step_pred = torch.cat(step_predictions, dim=0)
                preds.append(step_pred)
                
                node_prev_feats = []
                for node_idx in range(node_count):
                    node_pred = step_pred[node_idx:node_idx+1]
                    node_feat = self.prediction_to_feature(node_pred)
                    node_prev_feats.append(node_feat)
                
                prev_feat = torch.cat(node_prev_feats, dim=0)
                
                hidden_state = 0.7 * hidden_state + 0.2 * decoded + 0.1 * initial_hidden

            batch_predictions.append(torch.stack(preds, dim=0))

        return batch_predictions
