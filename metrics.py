import torch
import numpy as np
from Model_metrics import ModelEvaluator


def calculate_metrics(args, model, data_loader, device, dataset, pred_horizon=4, mode='avian', set='test'):
    """
    Calculates various performance metrics for the model's predictions.
    """
    model.eval()
    
    with torch.no_grad():
        model_evaluator = ModelEvaluator(args)
        forecast, groundtruth = [], []
        for batch in data_loader:
            batched_graphs, batched_targets, _, batched_temporal_info = batch
            
            device_graphs = []
            for time_graphs in batched_graphs:
                device_graphs.append([g.to(device) for g in time_graphs])
                
            device_targets = []
            for time_targets in batched_targets:
                device_targets.append([t.to(device) for t in time_targets])
                
            input_temporals, _ = batched_temporal_info
            device_input_temporals = []
            for seq_temporal in input_temporals:
                device_input_temporals.append(
                    [(y.to(device), w.to(device)) for y, w in seq_temporal]
                )
            
            batch_outputs = model(device_graphs, device_input_temporals)
            y_preds = torch.stack(batch_outputs, dim=0)
            
            reshaped_targets = []
            batch_size = len(device_targets[0])
            num_time_steps = len(device_targets)
            for batch_idx in range(batch_size):
                batch_targets_over_time = []
                for time_idx in range(num_time_steps):
                    batch_targets_over_time.append(device_targets[time_idx][batch_idx])
                
                batch_tensor = torch.stack(batch_targets_over_time, dim=0) 
                reshaped_targets.append(batch_tensor)
            
            targets = torch.stack(reshaped_targets, dim=0) 

            forecast_denorm = dataset.channel_wise_denormalize(y_preds.cpu().detach().numpy(), size=0)
            groundtruth_denorm = dataset.channel_wise_denormalize(targets.cpu().detach().numpy(), size=0)

            forecast.append(forecast_denorm)
            groundtruth.append(groundtruth_denorm)
                        
        forecast = np.concatenate(forecast, axis=0)
        groundtruth = np.concatenate(groundtruth, axis=0)
        
        multistep_metrics, avg_metrics = model_evaluator.evaluate_numeric(
            y_pred=forecast, 
            y_true=groundtruth, 
            mode=set
        )
        return multistep_metrics, avg_metrics