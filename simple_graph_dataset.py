import os
import glob
import torch
from torch.utils.data import Dataset
import pickle
from torch_geometric.data import HeteroData
import re
import numpy as np


class SimpleGraphDataset(Dataset):
    """
    Simple dataset that loads preprocessed PyG graphs.
    """
    
    def __init__(self, graphs_dir='processed_graphs', window_size=4, prediction_horizon=4, train_ratio=0.8, norm_mode='z_score', dataset='avian'):
        """
        Initialize the dataset.
        
        Args:
            graphs_dir: Directory containing preprocessed PyG graph files
            window_size: Number of consecutive weeks to use as input
            prediction_horizon: Number of weeks to predict
            train_ratio: Ratio of data to use for training
            norm_mode: Normalization mode, 'minmax', 'z_score', 'log_minmax', or 'log_plus_one'
            dataset: Dataset type, 'avian' or 'japan'
        """
        self.graphs_dir = graphs_dir
        self.dataset = dataset
        print(f"Initializing dataset with graphs_dir: {os.path.abspath(self.graphs_dir)}, dataset: {self.dataset}")
        
        if not os.path.exists(self.graphs_dir):
            print(f"WARNING: Graphs directory does not exist: {self.graphs_dir}")
            os.makedirs(self.graphs_dir, exist_ok=True)
            print(f"Created directory: {self.graphs_dir}")
        
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.is_train = True
        self.norm_mode = norm_mode
        self.norm_list = []  # Store normalization parameters for each channel
        self.epsilon = 1e-6  # 用于log变换中避免log(0)
        
        self._load_file_paths()
        
        self._prepare_time_windows()
        
        self.graph_cache = {}
        
        self._extract_temporal_ranges()
        
        self._compute_normalization_params()
    
    def _load_file_paths(self):
        """Load and sort graph file paths by year and week."""
        print("Finding graph files...")
        print(f"Looking in directory: {self.graphs_dir}")
        print(f"Directory exists: {os.path.exists(self.graphs_dir)}")
        
        graph_files_pt = glob.glob(os.path.join(self.graphs_dir, "pyg_*.pt"))
        graph_files_pkl = glob.glob(os.path.join(self.graphs_dir, "*.pkl"))
        graph_files = graph_files_pt + graph_files_pkl
        
        if self.dataset == 'japan':
            japan_files = glob.glob(os.path.join(self.graphs_dir, "japan_week_*.pt"))
            graph_files += japan_files
        elif self.dataset == 'state':
            state_files = glob.glob(os.path.join(self.graphs_dir, "state_week_*.pt"))
            graph_files += state_files
        
        print(f"Found {len(graph_files)} graph files ({len(graph_files_pt)} .pt files, {len(graph_files_pkl)} .pkl files)")
        
        if not graph_files:
            print("No graph files found! Please check the directory and file naming.")
            print(f"Directory contents: {os.listdir(self.graphs_dir) if os.path.exists(self.graphs_dir) else 'Directory does not exist'}")
            self.time_points = []
            return
        
        self.time_points = []
        for file_path in graph_files:
            filename = os.path.basename(file_path)
            try:
                if self.dataset == 'japan' and filename.startswith('japan_week_') and filename.endswith('.pt'):
                    match = re.match(r'japan_week_(\d+)', filename.replace('.pt', ''))
                    if match:
                        year = 0
                        week = int(match.group(1))
                        self.time_points.append((year, week, file_path))
                    else:
                        print(f"Skipping file with invalid japan format: {filename}")
                elif self.dataset == 'state' and filename.startswith('state_week_') and filename.endswith('.pt'):
                    match = re.match(r'state_week_(\d+)', filename.replace('.pt', ''))
                    if match:
                        year = 0
                        week = int(match.group(1))
                        self.time_points.append((year, week, file_path))
                    else:
                        print(f"Skipping file with invalid state format: {filename}")
                else:
                    if "pyg_" in filename:
                        base_name = filename.replace('pyg_', '').replace('.pt', '').replace('.pkl', '')
                    else:
                        base_name = filename.replace('.pt', '').replace('.pkl', '')
                    parts = base_name.split('_')
                    if len(parts) >= 2:
                        year = int(parts[-2])
                        week_str = parts[-1]
                        if 'week' in week_str:
                            week = int(re.search(r'\d+', week_str).group())
                        else:
                            week = int(week_str)
                        self.time_points.append((year, week, file_path))
                    else:
                        print(f"Skipping file with invalid format: {filename} (not enough parts in name)")
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Skipping file with invalid format: {filename}, error: {str(e)}")
        
        self.time_points.sort()
        print(f"Found {len(self.time_points)} valid graph files.")
    
    def _prepare_time_windows(self):
        """Prepare time windows for training and prediction."""
        self.time_windows = []
        
        for i in range(len(self.time_points) - self.window_size - self.prediction_horizon + 1):
            input_weeks = self.time_points[i:i+self.window_size]
            output_weeks = self.time_points[i+self.window_size:i+self.window_size+self.prediction_horizon]
            self.time_windows.append((input_weeks, output_weeks))
        
        num_train = int(len(self.time_windows) * self.train_ratio)
        self.train_windows = self.time_windows[:num_train]
        self.test_windows = self.time_windows[num_train:]
        
        print(f"Created {len(self.train_windows)} training sequences and {len(self.test_windows)} test sequences.")
    
    def set_mode(self, is_train=True):
        """Switch between train and test modes."""
        self.is_train = is_train
    
    def load_graph(self, file_path):
        """Load PyG graph from file with caching."""
        if file_path not in self.graph_cache:
            try:
                if not os.path.exists(file_path):
                    print(f"File does not exist: {file_path}")
                    return HeteroData()
                
                if file_path.endswith('.pt'):
                    pyg_data = torch.load(file_path, weights_only=False)
                elif file_path.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        pyg_data = pickle.load(f)
                else:
                    print(f"Unsupported file extension: {file_path}")
                    return HeteroData()
                
                if not isinstance(pyg_data, HeteroData):
                    print(f"Warning: Loaded data is not a HeteroData object: {type(pyg_data)}")
                    try:
                        hetero_data = HeteroData()
                        if hasattr(pyg_data, 'x_dict') and hasattr(pyg_data, 'edge_index_dict'):
                            hetero_data.x_dict = pyg_data.x_dict
                            hetero_data.edge_index_dict = pyg_data.edge_index_dict
                            pyg_data = hetero_data
                            print("Successfully converted to HeteroData")
                        else:
                            print("Could not convert to HeteroData format")
                            return HeteroData()
                    except Exception as e:
                        print(f"Error converting to HeteroData: {e}")
                        return HeteroData()
                
                new_pyg_data = HeteroData()
                
                for edge_type, edge_index in pyg_data.edge_index_dict.items():
                    new_pyg_data[edge_type].edge_index = edge_index

                for node_type, features in pyg_data.x_dict.items():
                    new_pyg_data[node_type].x = features.clone()
                
                self.graph_cache[file_path] = new_pyg_data
                
            except Exception as e:
                print(f"Error loading graph from {file_path}: {e}")
                return HeteroData()  # Return empty graph if loading fails
        return self.graph_cache[file_path]
    
    def normalize_features(self, x):
        """Normalize features using specified mode."""
        if self.norm_mode == 'minmax':
            x_max = x.max(dim=0)[0]  # Get the values, not the indices
            x_min = x.min(dim=0)[0]
            denominator = x_max - x_min
            denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
            x = (x - x_min) / denominator
            self.x_max = x_max
            self.x_min = x_min
        elif self.norm_mode == 'z_score':
            x_mean = x.mean(dim=0)
            x_std = x.std(dim=0)
            x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
            x = (x - x_mean) / x_std
            self.x_mean = x_mean
            self.x_std = x_std
        else:
            raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")
        return x
    
    def denormalize_features(self, x):
        """Denormalize features using specified mode."""
        if self.norm_mode == 'minmax':
            x = (self.x_max - self.x_min) * x + self.x_min
        elif self.norm_mode == 'z_score':
            x = self.x_std * x + self.x_mean
        else:
            raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")
        return x
    
    def channel_wise_normalize(self, x):
        """Normalize features channel-wise using pre-computed statistics."""
        assert len(x.shape) in [2, 3]  # [num_nodes, num_features] or [batch_size, num_nodes, num_features]
        
        if not self.norm_list:
            print("WARNING: Normalization parameters are not computed. Cannot normalize features.")
            return x
        
        x_norm = []
        for c in range(x.shape[-1]):
            if c >= len(self.norm_list):
                print(f"WARNING: Channel {c} exceeds norm_list length ({len(self.norm_list)}). Skipping normalization.")
                x_norm.append(x[..., c])
                continue
                
            if self.norm_mode == 'minmax':
                c_min = self.norm_list[c]['min']
                c_max = self.norm_list[c]['max']
                denominator = c_max - c_min
                denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
                flow_c = (x[..., c] - c_min) / denominator
                x_norm.append(flow_c)
            elif self.norm_mode == 'z_score':
                c_mean = self.norm_list[c]['mean']
                c_std = self.norm_list[c]['std']
                flow_c = (x[..., c] - c_mean) / c_std
                x_norm.append(flow_c)
            elif self.norm_mode == 'log_minmax':
                log_min = self.norm_list[c]['log_min']
                log_max = self.norm_list[c]['log_max']
                # 应用log变换
                log_transformed = torch.log(x[..., c] + self.epsilon)
                # 应用min-max归一化
                denominator = log_max - log_min
                denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
                flow_c = (log_transformed - log_min) / denominator
                x_norm.append(flow_c)
            elif self.norm_mode == 'log_plus_one':
                # 直接应用log(y+1)变换
                flow_c = torch.log(x[..., c] + 1.0)
                x_norm.append(flow_c)
            else:
                raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")

        x_norm = torch.stack(x_norm, dim=-1)
        return x_norm
    
    def channel_wise_denormalize(self, y, size):
        """Denormalize features channel-wise."""
        y_denorm = []
        for c in range(y.shape[-1]):
            if self.norm_mode == 'minmax':
                y_denorm.append(self.minmax_denormalize(y[..., c], c))
            elif self.norm_mode == 'z_score':
                y_denorm.append(self.z_score_denormalize(y[..., c], c))
            elif self.norm_mode == 'log_minmax':
                y_denorm.append(self.log_minmax_denormalize(y[..., c], c))
            elif self.norm_mode == 'log_plus_one':
                y_denorm.append(self.log_plus_one_denormalize(y[..., c], c))
            else:
                raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")
        y_denorm = torch.stack(y_denorm, dim=-1)
        return y_denorm
    
    def minmax_normalize(self, x):
        """Min-max normalization to [0, 1]."""
        # x_max = x.max()
        # x_min = x.min()
        # denominator = x_max - x_min
        # denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        # x = (x - x_min) / denominator
        # return x, x_min, x_max
        x_max, x_min = x.max(), x.min()
        x = (x - x_min) / (x_max - x_min)
        #x = x * 2 - 1
        return x, x_min, x_max
    
    def minmax_denormalize(self, x, c):
        """Min-max denormalization."""
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float() if not x.is_floating_point() else x
        if self.norm_list[c]['min'].device != x_tensor.device:
            x_tensor = x_tensor.to(self.norm_list[c]['min'].device)
            
        x = (self.norm_list[c]['max'] - self.norm_list[c]['min']) * x_tensor + self.norm_list[c]['min']
        return x
    
    def z_score_normalize(self, x):
        """Z-score normalization to N(0, 1)."""
        x_mean = x.mean()
        x_std = x.std()
        x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
        x = (x - x_mean) / x_std
        return x, x_mean, x_std
    
    def z_score_denormalize(self, x, c):
        """Z-score denormalization."""
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float() if not x.is_floating_point() else x
        if self.norm_list[c]['mean'].device != x_tensor.device:
            x_tensor = x_tensor.to(self.norm_list[c]['mean'].device)
            
        x = x_tensor * self.norm_list[c]['std'] + self.norm_list[c]['mean']
        return x
    
    def log_minmax_denormalize(self, x, c):
        """Log-minmax denormalization."""
        log_min = self.norm_list[c]['log_min']
        log_max = self.norm_list[c]['log_max']
        
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float() if not x.is_floating_point() else x
        if log_min.device != x_tensor.device:
            x_tensor = x_tensor.to(log_min.device)
        
        log_values = x_tensor * (log_max - log_min) + log_min
        original_values = torch.exp(log_values) - self.epsilon
        return original_values
    
    def log_plus_one_denormalize(self, x, c):
        """
        Apply inverse of log(y+1) transformation: exp(x) - 1
        
        Args:
            x: Normalized values
            c: Channel index
        
        Returns:
            Original values
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float() if not x.is_floating_point() else x
        
        # Apply exp(x) - 1 to get original values
        original_values = torch.exp(x_tensor) - 1.0
        return original_values
    
    def _compute_normalization_params(self):
        """Compute normalization parameters using training data."""
        if not self.train_windows:
            print("No training windows found for computing normalization parameters")
            return
        
        all_county_features = []
        print("Collecting features from all training graphs for global normalization...")
        for input_seq, _ in self.train_windows:
            for _, _, file_path in input_seq:
                try:
                    if file_path.endswith('.pt'):
                        pyg_data = torch.load(file_path, weights_only=False)
                    elif file_path.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            pyg_data = pickle.load(f)
                    else:
                        continue
                    
                    if hasattr(pyg_data, 'x_dict') and 'county' in pyg_data.x_dict:
                        county_features = pyg_data.x_dict['county']
                        all_county_features.append(county_features)
                except Exception as e:
                    print(f"Error loading features from {file_path}: {e}")
                    continue
        
        if not all_county_features:
            print("No county features found in training data")
            return
        
        try:
            all_features = torch.cat(all_county_features, dim=0)
            print(f"Collected features from {len(all_county_features)} graphs, total shape: {all_features.shape}")
            
            self.norm_list = []
            
            if self.norm_mode == 'minmax':
                for c in range(all_features.shape[1]):
                    c_min = all_features[:, c].min()
                    c_max = all_features[:, c].max()
                    self.norm_list.append({'min': c_min, 'max': c_max})
                    print(f'channel {c}, min: {c_min}, max: {c_max}')
            elif self.norm_mode == 'z_score':
                for c in range(all_features.shape[1]):
                    c_mean = all_features[:, c].mean()
                    c_std = all_features[:, c].std()
                    if c_std == 0:
                        c_std = torch.tensor(1.0)
                    self.norm_list.append({'mean': c_mean, 'std': c_std})
                    print(f'channel {c}, mean: {c_mean}, std: {c_std}')
            elif self.norm_mode == 'log_minmax':
                for c in range(all_features.shape[1]):
                    log_transformed = torch.log(all_features[:, c] + self.epsilon)
                    log_min = log_transformed.min()
                    log_max = log_transformed.max()
                    self.norm_list.append({'log_min': log_min, 'log_max': log_max})
                    zeros_pct = (all_features[:, c] == 0).float().mean() * 100
                    print(f'channel {c}, raw_min: {all_features[:, c].min()}, raw_max: {all_features[:, c].max()}, '
                          f'zeros: {zeros_pct:.1f}%, log_min: {log_min}, log_max: {log_max}')
            elif self.norm_mode == 'log_plus_one':
                # 对于log_plus_one模式，我们不需要保存任何参数，因为变换是固定的
                for c in range(all_features.shape[1]):
                    self.norm_list.append({})  # 空字典作为占位符
                    zeros_pct = (all_features[:, c] == 0).float().mean() * 100
                    print(f'channel {c}, raw_min: {all_features[:, c].min()}, raw_max: {all_features[:, c].max()}, '
                          f'zeros: {zeros_pct:.1f}%, using log(y+1) transform')
            
            print(f"Computed normalization parameters for {len(self.norm_list)} channels")
        except Exception as e:
            print(f"Error computing global normalization parameters: {e}")
    
    def extract_county_targets(self, pyg_data):
        """Extract infection counts for all counties from PyG data."""
        if 'county' not in pyg_data.x_dict:
            return [], torch.tensor([]), torch.tensor([])
        
        county_features = pyg_data.x_dict['county']
        
        infection_counts = county_features[:, 0]  # First column is infection_count
        abundance_counts = county_features[:, 1]  # Second column is abundance_count
        
        county_ids = list(range(len(infection_counts)))
        
        return county_ids, infection_counts, abundance_counts
    
    def __len__(self):
        """Return the number of sequences."""
        if self.is_train:
            return len(self.train_windows)
        else:
            return len(self.test_windows)
    
    def _extract_temporal_ranges(self):
        """Extract unique years and weeks for one-hot encoding."""
        years = set()
        weeks = set()
        
        for year, week, _ in self.time_points:
            years.add(year)
            weeks.add(week)
        
        self.unique_years = sorted(list(years))
        self.unique_weeks = sorted(list(weeks))
        
        print(f"Extracted {len(self.unique_years)} unique years and {len(self.unique_weeks)} unique weeks")
    
    def encode_temporal_info(self, year, week):
        """
        Create one-hot encodings for temporal information (year and week).
        
        Args:
            year: Year value
            week: Week value
            
        Returns:
            tuple: (year_encoding, week_encoding)
        """
        # Encode year
        year_encoding = torch.zeros(len(self.unique_years))
        if year in self.unique_years:
            year_idx = self.unique_years.index(year)
            year_encoding[year_idx] = 1.0
        
        # Encode week
        week_encoding = torch.zeros(len(self.unique_weeks))
        if week in self.unique_weeks:
            week_idx = self.unique_weeks.index(week)
            week_encoding[week_idx] = 1.0
        
        return year_encoding, week_encoding
    
    def __getitem__(self, idx):
        """
        Get a sequence of input graphs and target values.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            tuple: (input_graphs, target_values, county_ids, temporal_info)
                - input_graphs: A list of HeteroData objects, one for each time step
                - target_values: A list of tensors with target values for each prediction time step
                - county_ids: List of county IDs
                - temporal_info: A list of (year_encoding, week_encoding) tuples for each time step
        """
        if self.is_train:
            input_seq, target_seq = self.train_windows[idx]
        else:
            input_seq, target_seq = self.test_windows[idx]
        
        input_graphs = []
        temporal_info = []
        
        for year, week, file_path in input_seq:
            pyg_graph = self.load_graph(file_path)
            
            if 'county' in pyg_graph.x_dict:
                county_features = pyg_graph.x_dict['county']
                normalized_features = self.channel_wise_normalize(county_features)
                new_graph = HeteroData()
                for edge_type, edge_index in pyg_graph.edge_index_dict.items():
                    new_graph[edge_type].edge_index = edge_index
                for node_type, features in pyg_graph.x_dict.items():
                    if node_type == 'county':
                        new_graph[node_type].x = normalized_features
                    else:
                        new_graph[node_type].x = features
                pyg_graph = new_graph
            
            year_encoding, week_encoding = self.encode_temporal_info(year, week)
            temporal_info.append((year_encoding, week_encoding))
            
            input_graphs.append(pyg_graph)
        
        target_values = []
        target_temporal_info = []
        county_ids = None
        
        time_features_list = []
        for year, week, file_path in target_seq:
            pyg_graph = self.load_graph(file_path)
            
            if 'county' in pyg_graph.x_dict:
                county_features = pyg_graph.x_dict['county']
                normalized_features = self.channel_wise_normalize(county_features)
                ids = list(range(len(normalized_features)))
                if self.dataset == 'japan' or self.dataset == 'state':
                    counts = normalized_features[:, 0]
                else:
                    counts = normalized_features[:, 0]
                    abundances = normalized_features[:, 1]
            else:
                ids = []
                counts = torch.tensor([])
                abundances = torch.tensor([])
            
            if county_ids is None:
                county_ids = ids
            
            year_encoding, week_encoding = self.encode_temporal_info(year, week)
            target_temporal_info.append((year_encoding, week_encoding))
            if self.dataset == 'japan' or self.dataset == 'state':
                time_features_list.append((counts))
            else:
                # time_features_list.append((counts, abundances))
                time_features_list.append((counts))
                
            
        
        if time_features_list:
            for t in range(len(time_features_list)):
                if self.dataset == 'japan':
                    counts = time_features_list[t]
                else:
                    # counts, abundances = time_features_list[t]
                    counts = time_features_list[t]
                target_values.append(torch.stack([counts], dim=-1))

        return input_graphs, target_values, county_ids, (temporal_info, target_temporal_info)
