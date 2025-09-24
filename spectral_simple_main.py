import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import logging
from simple_graph_dataset import SimpleGraphDataset
from HeteroGraphNetwork import FusionGNN
from FullHeteroGNN import FullHeteroGNN
from spectral_loss import compute_complete_heterogeneous_laplacian_wrapper, compute_fusion_laplacian_learned, compute_spectral_loss
from metrics import calculate_metrics
import datetime
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import torch.linalg
from torch.nn.utils import parametrize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfectionWeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=1.0, low_weight=5.0, med_weight=10.0, high_weight=20.0,
                 low_threshold=0.5, med_threshold=5.0, high_threshold=20.0):
        super(InfectionWeightedMSELoss, self).__init__()
        self.zero_weight = zero_weight
        self.low_weight = low_weight 
        self.med_weight = med_weight
        self.high_weight = high_weight
        self.low_threshold = low_threshold
        self.med_threshold = med_threshold
        self.high_threshold = high_threshold
        
    def forward(self, input, target):
        mse_loss = (input - target) ** 2
        weights = torch.ones_like(target) * self.zero_weight
        
        low_mask = (target > self.low_threshold) & (target <= self.med_threshold)
        weights[low_mask] = self.low_weight
        
        med_mask = (target > self.med_threshold) & (target <= self.high_threshold)
        weights[med_mask] = self.med_weight
        
        high_mask = target > self.high_threshold
        weights[high_mask] = self.high_weight
        
        weighted_loss = weights * mse_loss
        
        return weighted_loss.mean()


def compute_county_spatial_laplacian(x_dict, edge_index_dict, device):
    if 'county' not in x_dict or ('county', 'spatial', 'county') not in edge_index_dict:
        logger.warning("County nodes or spatial edges not found for Hetero Laplacian.")
        return None

    county_features = x_dict['county']
    num_counties = county_features.size(0)
    spatial_edges = edge_index_dict[('county', 'spatial', 'county')]

    if num_counties == 0 or spatial_edges.size(1) == 0:
        logger.warning("No counties or spatial edges for Hetero Laplacian computation.")
        return torch.eye(num_counties, device=device) if num_counties > 0 else None

    adj = torch.zeros((num_counties, num_counties), device=device)
    src, dst = spatial_edges
    valid_mask = (src < num_counties) & (dst < num_counties)
    src, dst = src[valid_mask], dst[valid_mask]
    adj[src, dst] = 1
    adj[dst, src] = 1

    deg = torch.diag(torch.sum(adj, dim=1))

    laplacian = deg - adj
    return laplacian



def collate_fn(batch):
    graph_sequences, targets, metadata, temporal_info = zip(*batch)
    batch_size = len(graph_sequences)

    if batch_size > 0:
        seq_length = len(graph_sequences[0]) if graph_sequences[0] else 0
        if seq_length == 0:
            return [], [], metadata, ([], [])

        batched_graphs = []
        for t in range(seq_length):
            time_t_graphs = [seq[t] for seq in graph_sequences if len(seq) > t]  # Check sequence length
            if time_t_graphs:
                batched_graphs.append(time_t_graphs)

        batched_targets = []
        if targets and targets[0]:
            target_len = len(targets[0])
            for t in range(target_len):
                time_t_targets = [seq[t] for seq in targets if len(seq) > t]  # Check target sequence length
                if time_t_targets:
                    batched_targets.append(time_t_targets)

        input_temporals = []
        target_temporals = []
        if temporal_info and all(temp is not None for temp in temporal_info):
            input_temporals = [temp[0] for temp in temporal_info if temp and len(temp) > 0]
            target_temporals = [temp[1] for temp in temporal_info if temp and len(temp) > 1]

        batched_temporal_info = (input_temporals, target_temporals)

        return batched_graphs, batched_targets, metadata, batched_temporal_info

    else:
        return [], [], metadata, ([], [])


def train(model, train_loader, optimizer, criterion, device, dataset, writer=None, epoch=None, fold=None,
          spectral_gamma=0.3, spectral_k=10, use_eigenvalue_constraint=True, eigenvalue_loss_type='cosine_similarity'):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_spec_loss = 0
    num_batches = 0

    all_preds = []
    all_targets = []

    fusion_builder = model.fusion_builder if isinstance(model, FusionGNN) else None
    for batch_idx, (batched_graphs, batched_targets, _, batched_temporal_info) in enumerate(
            tqdm(train_loader, desc="Training")):

        if not batched_graphs or not batched_graphs[0]:
            continue
        batch_size = len(batched_graphs[0])
        if batch_size == 0:
            continue

        optimizer.zero_grad()

        device_graphs = []
        for time_graphs in batched_graphs:
            device_graphs.append([g.to(device) for g in time_graphs if g is not None])

        device_targets = []
        for time_targets in batched_targets:
            device_targets.append([t.to(device) for t in time_targets if t is not None])

        input_temporals, target_temporals = batched_temporal_info
        device_input_temporals = []
        device_output_temporals = []
        if input_temporals:
            for seq_temporal in input_temporals:
                if seq_temporal:
                    device_input_temporals.append(
                        [(y.to(device), w.to(device)) for y, w in seq_temporal if y is not None and w is not None]
                    )

        if target_temporals:
            for seq_temporal in target_temporals:
                if seq_temporal:
                    device_output_temporals.append(
                        [(y.to(device), w.to(device)) for y, w in seq_temporal if y is not None and w is not None]
                    )

        batch_outputs = model(device_graphs, device_input_temporals, device_output_temporals)

        if batch_outputs is None or not isinstance(batch_outputs, list) or not batch_outputs or any(
                output is None or not output.numel() for output in batch_outputs):
            logger.warning(f"Skipping batch {batch_idx} due to empty or invalid predictions")
            continue

        batch_loss = 0
        batch_pred_loss = 0
        batch_spec_loss = 0
        valid_samples_in_batch = 0

        for i in range(batch_size):
            outputs = batch_outputs[i]
            if not device_targets or len(device_targets) == 0:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to empty targets.")
                continue

            if i == 0:
                num_time_steps = len(device_targets)
                batch_size_actual = len(device_targets[0]) if num_time_steps > 0 else 0
                
                reorganized_targets = []
                
                for b in range(batch_size_actual):
                    reorganized_targets.append([])

                for t in range(num_time_steps):
                    for b in range(batch_size_actual):
                        if len(device_targets[t]) > b:
                            reorganized_targets[b].append(device_targets[t][b])
                
                device_targets = reorganized_targets

            sample_targets = device_targets[i] if i < len(device_targets) else []
            if not sample_targets:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to missing targets.")
                continue

            sample_pred_loss = 0
            valid_timepoints = 0

            for t, target_t in enumerate(sample_targets):
                if target_t is None or target_t.numel() == 0:
                    continue
                
                if t >= outputs.size(0):
                    logger.warning(f"Sample {i}, exceeds prediction horizon. Skipping time step {t}.")
                    continue
                
                preds_t = outputs[t]
                num_counties_pred = preds_t.size(0)
                num_counties_target = target_t.size(0)
                
                if num_counties_pred != num_counties_target:
                    logger.warning(f"Sample {i}, Time {t}: Mismatch counties pred={num_counties_pred}, target={num_counties_target}. Skipping time step.")
                    continue
                
                time_loss = criterion(preds_t, target_t)
                sample_pred_loss += time_loss
                valid_timepoints += 1

            if valid_timepoints == 0:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to no valid timepoints for prediction loss.")
                continue
            
            sample_pred_loss = sample_pred_loss / valid_timepoints

            sample_spec_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if spectral_gamma > 0 and fusion_builder is not None:
                if device_graphs and len(device_graphs[-1]) > i:
                    last_input_graph = device_graphs[-1][i]
                    if last_input_graph and hasattr(last_input_graph, 'x_dict') and hasattr(last_input_graph,
                                                                                            'edge_index_dict'):
                        L_het, P, node_mapping = compute_complete_heterogeneous_laplacian_wrapper(
                            last_input_graph.x_dict, last_input_graph.edge_index_dict, device)
                        
                        if L_het is None:
                            L_het = compute_county_spatial_laplacian(
                                last_input_graph.x_dict, last_input_graph.edge_index_dict, device)
                                
                        encoded_x_dict = model.encode_heterograph(
                            last_input_graph.x_dict, last_input_graph.edge_index_dict)
                        corrected_x_dict = model.mrf_correction(
                            encoded_x_dict, last_input_graph.edge_index_dict)

                        fusion_features_i, fusion_edge_index_i, _ = fusion_builder(
                            corrected_x_dict, last_input_graph.edge_index_dict)

                        L_fus = compute_fusion_laplacian_learned(
                            fusion_features_i, fusion_edge_index_i, device)

                        if L_het is not None and L_fus is not None:
                            use_l2_norm = P is not None
                            sample_spec_loss = compute_spectral_loss(
                                L_het, L_fus, k=spectral_k, device=device, P=P, 
                                use_l2_norm=use_l2_norm, 
                                use_eigenvalue_constraint=use_eigenvalue_constraint,
                                eigenvalue_loss_type=eigenvalue_loss_type)
                        else:
                            logger.warning(f"Could not compute Laplacians for sample {i}, batch {batch_idx}")
                    else:
                        logger.warning(
                            f"Missing graph data for spectral loss calculation for sample {i}, batch {batch_idx}.")
                else:
                    logger.warning(f"Missing last input graph for spectral loss for sample {i}, batch {batch_idx}.")

            sample_total_loss = sample_pred_loss + spectral_gamma * sample_spec_loss
            batch_loss += sample_total_loss
            batch_pred_loss += sample_pred_loss.item()
            batch_spec_loss += sample_spec_loss.item()
            valid_samples_in_batch += 1

        if valid_samples_in_batch > 0:
            avg_batch_loss = batch_loss / valid_samples_in_batch
            avg_batch_loss.backward()
            optimizer.step()
            total_loss += avg_batch_loss.item()
            total_pred_loss += (batch_pred_loss / valid_samples_in_batch)
            total_spec_loss += (batch_spec_loss / valid_samples_in_batch)
            num_batches += 1
        else:
            logger.warning(f"Batch {batch_idx} had no valid samples.")

    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_epoch_pred_loss = total_pred_loss / num_batches if num_batches > 0 else float('inf')
    avg_epoch_spec_loss = total_spec_loss / num_batches if num_batches > 0 else float('inf')

    if writer is not None and epoch is not None:
        tb_prefix = f"fold_{fold}/" if fold is not None else ""
        writer.add_scalar(f"{tb_prefix}train/total_loss", avg_epoch_loss, epoch)
        writer.add_scalar(f"{tb_prefix}train/pred_loss", avg_epoch_pred_loss, epoch)
        writer.add_scalar(f"{tb_prefix}train/spectral_loss", avg_epoch_spec_loss, epoch)

    logger.info(
        f"Epoch Avg Losses - Total: {avg_epoch_loss:.4f}, Prediction: {avg_epoch_pred_loss:.4f}, Spectral: {avg_epoch_spec_loss:.4f} (gamma={spectral_gamma})")

    return avg_epoch_pred_loss, (all_preds, all_targets)


def evaluate(model, val_loader, criterion, device, dataset, writer=None, epoch=None, fold=None, mode="val"):
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (batched_graphs, batched_targets, _, batched_temporal_info) in enumerate(
                tqdm(val_loader, desc="Evaluating")):

            if not batched_graphs or not batched_graphs[0]: continue
            batch_size = len(batched_graphs[0])
            if batch_size == 0: continue

            device_graphs = []
            for time_graphs in batched_graphs:
                device_graphs.append([g.to(device) for g in time_graphs if g is not None])

            device_targets = []
            for time_targets in batched_targets:
                device_targets.append([t.to(device) for t in time_targets if t is not None])

            input_temporals, _ = batched_temporal_info
            device_input_temporals = []
            if input_temporals:
                for seq_temporal in input_temporals:
                    if seq_temporal:
                        device_input_temporals.append(
                            [(y.to(device), w.to(device)) for y, w in seq_temporal if y is not None and w is not None]
                        )

            batch_outputs = model(device_graphs, device_input_temporals)

            if batch_outputs is None or not isinstance(batch_outputs, list) or not batch_outputs or any(
                    output is None or not output.numel() for output in batch_outputs):
                logger.warning(f"Skipping batch {batch_idx} during eval due to empty or invalid predictions")
                continue

            num_time_steps = len(device_targets)
            batch_size_actual = len(device_targets[0]) if num_time_steps > 0 else 0
            
            reorganized_targets = []
            
            for b in range(batch_size_actual):
                reorganized_targets.append([])
            
            for t in range(num_time_steps):
                for b in range(batch_size_actual):
                    if len(device_targets[t]) > b:
                        reorganized_targets[b].append(device_targets[t][b])
            device_targets = reorganized_targets
            batch_loss = 0
            valid_samples_in_batch = 0

            for i in range(batch_size):
                outputs = batch_outputs[i]
                sample_targets = device_targets[i] if i < len(device_targets) else []
                if not sample_targets:
                    logger.warning(f"Skipping sample {i} in batch {batch_idx} during eval due to missing targets.")
                    continue

                sample_loss = 0
                valid_timepoints = 0

                for t, target_t in enumerate(sample_targets):
                    if target_t is None or target_t.numel() == 0: continue
                    
                    if t >= outputs.size(0):
                        continue
                        
                    preds_t = outputs[t]
                    num_counties_pred = preds_t.size(0)
                    num_counties_target = target_t.size(0)

                    if num_counties_pred != num_counties_target:
                        logger.warning(
                            f"Eval Sample {i}, Time {t}: Mismatch counties pred={num_counties_pred}, target={num_counties_target}. Skipping time step.")
                        continue

                    time_loss = criterion(preds_t, target_t)
                    sample_loss += time_loss
                    valid_timepoints += 1
                    
                    all_preds.append(preds_t.cpu())
                    all_targets.append(target_t.cpu())

                if valid_timepoints > 0:
                    sample_loss = sample_loss / valid_timepoints
                    batch_loss += sample_loss
                    valid_samples_in_batch += 1

            if valid_samples_in_batch > 0:
                batch_loss = batch_loss / valid_samples_in_batch
                total_loss += batch_loss.item()
                num_batches += 1
            else:
                logger.warning(f"Eval Batch {batch_idx} had no valid samples.")

    eval_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    if writer is not None and epoch is not None:
        tb_prefix = f"fold_{fold}/" if fold is not None else ""
        writer.add_scalar(f"{tb_prefix}{mode}/loss", eval_loss, epoch)
    return eval_loss, (all_preds, all_targets)


def main():
    parser = argparse.ArgumentParser(description='Train a temporal GNN model for infection prediction')
    parser.add_argument('--dataset', type=str, default='avian',
                        help='Dataset to use, japan or avian')
    parser.add_argument('--data_dir', type=str, default='/scratch/processed_graphs',
                        help='Directory containing graph pickle files, for japan, it is /scratch/processed_japan, for avian, it is /scratch/processed_graphs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden dimension size')
    parser.add_argument('--num_mrf', type=int, default=1, help='Iteration of MRF correction')
    parser.add_argument('--window_size', type=int, default=4, help='Input window size (weeks)')
    parser.add_argument('--pred_horizon', type=int, default=4, help='Prediction horizon (weeks)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--spectral_gamma', type=float, default=0.2,
                        help='Weight for the spectral regularization loss (default: 0.0, disabled)')
    parser.add_argument('--spectral_k', type=int, default=10,
                        help='Number of eigenvalues (k) to compare for spectral loss')
    parser.add_argument('--use_eigenvalue_constraint', type=bool, default=True,
                        help='Use eigenvalue constraint to handle dimension mismatch')
    parser.add_argument('--eigenvalue_loss_type', type=str, default='cosine_similarity',
                        choices=['cosine_similarity', 'mse', 'kl_divergence', 'wasserstein'],
                        help='Type of loss function for eigenvalue constraint')
    parser.add_argument('--model_dir', type=str, default='./saved_results', help='Directory to save models')
    parser.add_argument('--model_type', type=str, default='FusionGNN',
                        choices=['FullHeteroGNN', 'FusionGNN'], help='Type of model to use')
    parser.add_argument('--link_threshold', type=float, default=0.5,
                        help='Threshold for learnable linking in FusionGNN')
    parser.add_argument('--use_top_k', type=bool, default=True,
                        help='Use top-k linking instead of threshold in FusionGNN')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Value of K for top-k linking in FusionGNN')
    parser.add_argument('--use_temporal', type=bool, default=True,
                        help='Use temporal information (year/week) for prediction')
    parser.add_argument('--use_kfold', type=bool, default=True,
                        help='Use k-fold cross validation')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross validation')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (used only if use_kfold=False)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation (used only if use_kfold=False)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing (used only if use_kfold=False)')
    parser.add_argument('--use_cuda', type=str, default='cuda',
                        help='Use CUDA for training')
    parser.add_argument('--device', type=int, default=1, 
                        help='Device to use for training')
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                    help='Weight decay (L2 penalty) for optimizer')
    parser.add_argument('--norm_mode', type=str, default='z_score',
                        choices=['minmax', 'z_score', 'log_minmax', 'log_plus_one'],
                        help='Normalization mode for dataset')
    parser.add_argument('--previous_weight', type=float, default=0.1,
                        help='Weight for the previous step in FusionGNN')
    parser.add_argument('--initial_weight', type=float, default=0.3,
                        help='Weight for the initial hidden state in FusionGNN')
    
    # loss arguments
    parser.add_argument('--loss_type', type=str, default='infection_weighted',
                        choices=['mse', 'infection_weighted'],
                        help='Type of loss function to use')
    parser.add_argument('--infection_zero_weight', type=float, default=1.0,
                        help='Weight for zero infection cases')
    parser.add_argument('--infection_low_weight', type=float, default=5.0,
                        help='Weight for low infection cases')
    parser.add_argument('--infection_med_weight', type=float, default=10.0,
                        help='Weight for medium infection cases')
    parser.add_argument('--infection_high_weight', type=float, default=20.0,
                        help='Weight for high infection cases')
    parser.add_argument('--infection_low_threshold', type=float, default=0.5,
                        help='Threshold for low infection cases')
    parser.add_argument('--infection_med_threshold', type=float, default=5.0,
                        help='Threshold for medium infection cases')
    parser.add_argument('--infection_high_threshold', type=float, default=20.0,
                        help='Threshold for high infection cases')
    
    # post-processing arguments
    parser.add_argument('--use_post_processing', type=bool, default=True,
                        help='Whether to use post-processing for predictions')
    parser.add_argument('--detection_threshold', type=float, default=0.3,
                        help='Threshold for infection detection in post-processing')
    parser.add_argument('--min_prediction', type=float, default=0.0,
                        help='Minimum allowed prediction value')
    
    args = parser.parse_args()
    
    args.model_dir = args.model_dir + '_' + args.loss_type + '_' + str(args.pred_horizon) + '_' + str(args.spectral_gamma)
    os.makedirs(args.model_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device(args.use_cuda if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    logger.info("Creating datasets...")

    full_dataset = SimpleGraphDataset(
        graphs_dir=args.data_dir,
        window_size=args.window_size,
        prediction_horizon=args.pred_horizon,
        dataset=args.dataset,
        norm_mode=args.norm_mode
    )
    logger.info(f"Full dataset size: {len(full_dataset)}")
    if len(full_dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return

    if args.use_kfold:
        logger.info(f"Using {args.num_folds}-fold cross validation")
        if args.num_folds < 2:
            logger.error("Number of folds must be at least 2 for K-fold cross validation.")
            return
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
        dataset_indices = list(range(len(full_dataset)))
        all_fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_indices)):
            logger.info(f"Starting fold {fold + 1}/{args.num_folds}")
            if len(train_idx) < 2:
                logger.warning(f"Fold {fold + 1} has insufficient data for train/val split. Skipping fold.")
                continue

            logger.info(f"Fold {fold + 1} split: Train={len(train_idx)}, Test={len(test_idx)}")

            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )

            test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )

            logger.info(f"Creating model for fold {fold + 1}...")

            sample_batch = next(iter(train_loader))
            if sample_batch and len(sample_batch) > 0 and sample_batch[0]:
                sample_graphs = sample_batch[0]
                if sample_graphs and len(sample_graphs) > 0:
                    sample_graph = sample_graphs[0][0]
                    county_input_dim = sample_graph['county'].x.size(-1)
                    logger.info(f"Detected county input dimension: {county_input_dim}")
                else:
                    county_input_dim = 1
                    logger.warning("Could not detect county input dimension, using default: 1")
            else:
                county_input_dim = 1
                logger.warning("Could not detect county input dimension, using default: 1")

            if args.model_type == 'FusionGNN':
                channel = 1
                model = FusionGNN(
                    channel=channel,
                    hidden_dim=args.hidden_dim,
                    num_layers=1,
                    dropout=args.dropout,
                    pred_horizon=args.pred_horizon,
                    link_threshold=args.link_threshold,
                    use_top_k=args.use_top_k,
                    top_k=args.top_k,
                    num_mrf=args.num_mrf,
                    device=args.device,
                    county_input_dim=county_input_dim
                ).to(args.device)
            else:
                model = FullHeteroGNN(
                    hidden_dim=args.hidden_dim,
                    num_layers=1,
                    dropout=args.dropout,
                    pred_horizon=args.pred_horizon,
                    num_mrf=args.num_mrf,
                    device=args.device
                ).to(args.device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.loss_type == 'mse':
                criterion = nn.MSELoss()
                logger.info("Using MSE Loss")
            elif args.loss_type == 'infection_weighted':
                criterion = InfectionWeightedMSELoss(
                    zero_weight=args.infection_zero_weight,
                    low_weight=args.infection_low_weight,
                    med_weight=args.infection_med_weight,
                    high_weight=args.infection_high_weight,
                    low_threshold=args.infection_low_threshold,
                    med_threshold=args.infection_med_threshold,
                    high_threshold=args.infection_high_threshold
                )
                logger.info(f"Using Infection Weighted MSE Loss (weights: {args.infection_zero_weight}, {args.infection_low_weight}, {args.infection_med_weight}, {args.infection_high_weight})")
            else:
                criterion = nn.MSELoss()
                logger.warning(f"Unknown loss type '{args.loss_type}', using MSE Loss as default")

            logger.info(f"Starting training for fold {fold + 1}...")
            best_mse = float('inf')
            best_f1 = 0.0
            best_train_loss = float('inf')
            best_mae = float('inf')
            best_model_state = None 
            patience = 10
            counter = 0
            
            for epoch in range(args.epochs):
                train_pred_loss, _ = train(
                    model, train_loader, optimizer, criterion, args.device, full_dataset,
                    writer=None, epoch=epoch, fold=fold + 1,
                    spectral_gamma=args.spectral_gamma, spectral_k=args.spectral_k,
                    use_eigenvalue_constraint=args.use_eigenvalue_constraint,
                    eigenvalue_loss_type=args.eigenvalue_loss_type)

                logger.info(
                    f"Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs} - Train Pred Loss: {train_pred_loss:.4f}")
                if train_pred_loss < best_train_loss:
                    best_train_loss = train_pred_loss
                    logger.info(f"Best Train Loss: {best_train_loss:.4f}")
                    best_model_state = model.state_dict().copy()

                    save_path = os.path.join(args.model_dir, f'best_model_fold{fold + 1}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'train_loss': train_pred_loss,
                        'best_mse': best_mse,
                        'best_f1': best_f1,
                        'best_mae': best_mae,
                        'args': vars(args)
                    }, save_path)

                    logger.info(f"Saved new best model for fold {fold + 1} to {save_path} with Train Loss: {train_pred_loss:.4f}")
                else:
                    print('not save best model on validation MAE for this fold')
                    counter += 1
                    logger.info(f"Early stopping counter: {counter}/{patience}")
                    if counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs for fold {fold + 1}")
                        break

            if best_model_state is not None:
                model.load_state_dict(best_model_state, strict=True)
                logger.info(f"Successfully loaded model state dict for fold {fold + 1}")
            else:
                logger.warning(f"No best model saved for fold {fold + 1}. Testing with final model state.")

            prediction_save_dir = os.path.join(args.model_dir, f'fold_{fold+1}_predictions')

            multistep_test_metrics, avg_test_metrics = calculate_metrics(args,
                                             model, test_loader, args.device, full_dataset,
                                             pred_horizon=args.pred_horizon, mode=args.dataset, set='test',
                                             save_predictions=False, save_dir=prediction_save_dir)

            if avg_test_metrics:
                logger.info(f"Fold {fold + 1} Test Metrics:")
                logger.info(f"MSE: {avg_test_metrics['mse']:.4f}, RMSE: {avg_test_metrics['rmse']:.4f}, "
                            f"MAE: {avg_test_metrics['mae']:.4f}, F1: {avg_test_metrics['f1']:.4f}, "
                            f"FPR: {avg_test_metrics.get('fpr', 0.0):.4f}")
                
                if 'detection_precision' in avg_test_metrics:
                    logger.info(f"Enhanced Test Metrics:")
                    logger.info(f"  Detection - Precision: {avg_test_metrics['detection_precision']:.3f}, "
                                f"Recall: {avg_test_metrics['detection_recall']:.3f}, "
                                f"F1: {avg_test_metrics['detection_f1']:.3f}, "
                                f"Accuracy: {avg_test_metrics['detection_accuracy']:.3f}, "
                                f"FPR: {avg_test_metrics.get('detection_fpr', 0.0):.3f}")
                    logger.info(f"  Regression (infection samples) - MSE: {avg_test_metrics.get('regression_mse', 0):.4f}, "
                                f"MAE: {avg_test_metrics.get('regression_mae', 0):.4f}")
                    logger.info(f"  Data Distribution - Infection Rate: {avg_test_metrics.get('infection_rate', 0):.2%}")
                
                all_fold_metrics.append({
                    'fold': fold + 1,
                    'test_mse': avg_test_metrics['mse'],
                    'test_rmse': avg_test_metrics['rmse'],
                    'test_mae': avg_test_metrics['mae'],
                    'test_f1': avg_test_metrics['f1'],
                    'test_fpr': avg_test_metrics.get('fpr', 0.0),
                    'test_pearson': avg_test_metrics.get('pearson', 0.0),
                    'test_spearman': avg_test_metrics.get('spearman', 0.0),
                })
            else:
                logger.warning(f"Could not calculate test metrics for fold {fold + 1}")

        if not all_fold_metrics:
            logger.error("No folds completed successfully. Cannot compute cross-validation summary.")
            return

        test_mse_values = [fold_data['test_mse'] for fold_data in all_fold_metrics]
        test_rmse_values = [fold_data['test_rmse'] for fold_data in all_fold_metrics]
        test_mae_values = [fold_data['test_mae'] for fold_data in all_fold_metrics]
        test_f1_values = [fold_data['test_f1'] for fold_data in all_fold_metrics]
        test_fpr_values = [fold_data['test_fpr'] for fold_data in all_fold_metrics]
        test_pearson_values = [fold_data['test_pearson'] for fold_data in all_fold_metrics]
        test_spearman_values = [fold_data['test_spearman'] for fold_data in all_fold_metrics]

        avg_test_mse = np.mean(test_mse_values)
        avg_test_rmse = np.mean(test_rmse_values)
        avg_test_mae = np.mean(test_mae_values)
        avg_test_f1 = np.mean(test_f1_values)
        avg_test_fpr = np.mean(test_fpr_values)
        avg_test_pearson = np.mean(test_pearson_values)
        avg_test_spearman = np.mean(test_spearman_values)

        std_test_mse = np.std(test_mse_values)
        std_test_rmse = np.std(test_rmse_values)
        std_test_mae = np.std(test_mae_values)
        std_test_f1 = np.std(test_f1_values)
        std_test_fpr = np.std(test_fpr_values)
        std_test_pearson = np.std(test_pearson_values)
        std_test_spearman = np.std(test_spearman_values)

        logger.info("===== Cross-Validation Results =====")
        logger.info(f"Average Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")
        logger.info(f"Average Test RMSE: {avg_test_rmse:.4f} ± {std_test_rmse:.4f}")
        logger.info(f"Average Test MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")        
        logger.info(f"Average Test F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}")
        logger.info(f"Average Test FPR: {avg_test_fpr:.4f} ± {std_test_fpr:.4f}")
        logger.info(f"Average Test Pearson: {avg_test_pearson:.4f} ± {std_test_pearson:.4f}")
        logger.info(f"Average Test Spearman: {avg_test_spearman:.4f} ± {std_test_spearman:.4f}")

        results_df = pd.DataFrame(all_fold_metrics)
        result_name = 'cross_validation_results_window' + str(args.window_size) + '_horizon' + str(
            args.pred_horizon) + '.csv'
        results_df.to_csv(os.path.join(args.model_dir, result_name), index=False)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = os.path.join(args.model_dir,
                                        f'{args.model_type}_w{args.window_size}_h{args.pred_horizon}_gamma{args.spectral_gamma}_k{args.spectral_k}_{timestamp}.log')

        with open(summary_filename, 'w') as log_file:
            log_file.write(f"=== {args.num_folds}-Fold Cross Validation Results for {args.model_type} ===\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"Args: {vars(args)}\n\n")

            log_file.write("Results for each fold:\n")
            for fold_data in all_fold_metrics:
                fold = fold_data['fold']
                log_file.write(f"Fold {fold} - MSE: {fold_data['test_mse']:.4f}, RMSE: {fold_data['test_rmse']:.4f}, "
                               f"MAE: {fold_data['test_mae']:.4f}, F1: {fold_data['test_f1']:.4f}, FPR: {fold_data['test_fpr']:.4f}\n")

            log_file.write("\nAverage Results across all folds:\n")
            log_file.write(f"MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}\n")
            log_file.write(f"RMSE: {avg_test_rmse:.4f} ± {std_test_rmse:.4f}\n")
            log_file.write(f"MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}\n")
            log_file.write(f"F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}\n")
            log_file.write(f"FPR: {avg_test_fpr:.4f} ± {std_test_fpr:.4f}\n")
            log_file.write(f"Pearson Correlation: {avg_test_pearson:.4f} ± {std_test_pearson:.4f}\n")
            log_file.write(f"Spearman Correlation: {avg_test_spearman:.4f} ± {std_test_spearman:.4f}\n")

        logger.info(f"Cross-validation summary saved to {summary_filename}")

if __name__ == "__main__":
    main()
