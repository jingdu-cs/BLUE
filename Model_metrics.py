import time
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import matplotlib.pyplot as plt
import seaborn as sns


def mask_data(x:np.array, H:int, W:int, mask):
    assert (len(x.shape) == 4)|(len(x.shape) == 5)
    if len(x.shape) == 4:   # graph
        assert x.shape[-2]==H*W
        x = x.reshape(x.shape[0], x.shape[1], H, W, x.shape[-1])
    else:   # len=5
        if (x.shape[-2]==H)&(x.shape[-1]==W):   # grid
            x = x.transpose((0, 1, 3, 4, 2))    # switch to channel last
        else:
            assert (x.shape[2]==H)&(x.shape[3]==W)
            pass

    if mask != []:
        mask_count = 0
        x_masked = list()
        for h in range(H):
            for w in range(W):
                if (h, w) in mask:
                    mask_count += 1
                    continue
                x_masked.append(x[:, :, h, w, :])

        x_masked = np.array(x_masked).transpose((1, 2, 0, 3))
    else:
        x_masked = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])       # unmasked
    return x_masked



class ModelEvaluator(object):
    def __init__(self, params:dict, precision=4, epsilon=1e-5):
        self.params = params if isinstance(params, dict) else vars(params)
        
        if not isinstance(self.params, dict):
            raise TypeError(f"params should be a dictionary or an object convertible to a dictionary, got {type(params)}")

        self.precision = precision
        self.epsilon = epsilon

    @staticmethod
    def plot_value_heatmap(data_array: np.array, plot_title: str, filename_path: str, cmap: str = "viridis"):

        if data_array.ndim == 1: # Reshape if 1D (e.g. [num_locations])
            data_array = data_array.reshape(1, -1)
        
        plt.figure(figsize=(max(10, data_array.shape[1] // 100), max(4, data_array.shape[0] // 2))) # Adjust size
        sns.heatmap(data_array, cmap=cmap, annot=False, cbar=True)
        plt.title(plot_title)
        plt.xlabel("Locations (e.g., Counties)")
        plt.ylabel("Samples in Batch / Other Dimension")
        
        plot_dir = os.path.dirname(filename_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        
        plt.savefig(filename_path)
        plt.close()
        print(f"Heatmap saved to {filename_path}")

    def evaluate_numeric(self, y_pred: np.array, y_true: np.array, mode:str, mask:list=None):
        assert y_pred.shape == y_true.shape
        file_path = self.params['model_dir'] + f'/{self.params["dataset"]}-{self.params["model_type"]}-{mode}-metrics.csv'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as cf:
            print(' '.join(['*' * 10, f'Evaluation on {mode} set started at', time.ctime(), '*' * 10]))
            cf.write(' '.join(['*' * 10, f'Evaluation on {mode} set started at', time.ctime(), '*' * 10]))
            cf.write(f'*****, Evaluation starts, {mode}, {time.ctime()}, ***** \n')
            for param in self.params.keys():
                cf.write(f'{param}: {self.params[param]},')
            cf.write('\n')

        run_plots_dir = os.path.join(self.params['model_dir'], 'evaluation_plots', mode)
        os.makedirs(run_plots_dir, exist_ok=True)

        y_pred_masked, y_true_masked = y_pred, y_true
        multistep_metrics = list()
        for step in range(self.params['pred_horizon']):
            print(f'Evaluating step {step}:')
            step_metrics = self.one_step_eval_num(
                y_pred_masked[:, step,...], 
                y_true_masked[:, step,...],
                current_step_idx=step,
                run_plots_dir=run_plots_dir,
                mode_name=mode
            )
            multistep_metrics.append(step_metrics)
        horizon_avg = list()
        avg_metrics = {}
        for measure in list(step_metrics.keys()):
            step_measures = list()
            for step in range(self.params['pred_horizon']):
                step_measures.append(multistep_metrics[step][measure])
            horizon_avg.append(np.mean(step_measures))
            avg_metrics[measure.lower()] = np.mean(step_measures)
            print(f'Horizon avg. {measure}: {horizon_avg[-1]:9.4f}')

        with open(self.params['model_dir'] + f'/{self.params["dataset"]}-{self.params["model_type"]}-{mode}-metrics.csv', 'a') as cf:
            col_names = [' '] + list(step_metrics.keys())
            cf.write(','.join(col_names) + '\n')
            for step in range(self.params['pred_horizon']):
                row_items = [f'Step {step}'] + list(multistep_metrics[step].values())
                cf.write(','.join([str(item) for item in row_items]) + '\n')
            row_items = [f'Horizon avg.'] + horizon_avg
            cf.write(','.join([str(item) for item in row_items]) + '\n')
            cf.write(f'*****, Evaluation ends, {mode}, {time.ctime()}, ***** \n \n')
            print(' '.join(['*' * 10, f'Evaluation on {mode} set ended at', time.ctime(), '*' * 10]))

        return multistep_metrics, avg_metrics

    def one_step_eval_num(self, y_pred_step: np.array, y_true_step: np.array, current_step_idx: int, run_plots_dir: str, mode_name: str):
        assert y_pred_step.shape == y_true_step.shape

        MSEs, RMSEs, MAEs, MAPEs, F1s, Precisions, Recalls, FPRs, Pearsons, Spearmans = [], [], [], [], [], [], [], [], [], []
        print('y_pred_step.shape', y_pred_step.shape)
        for c in range(y_pred_step.shape[-1]):
            if c == 0:
                y_pred_step_c = y_pred_step[...,c]
                y_true_step_c = y_true_step[...,c]

                if hasattr(y_pred_step_c, 'detach') and hasattr(y_pred_step_c, 'cpu') and hasattr(y_pred_step_c, 'numpy'):
                    y_pred_numpy_c = y_pred_step_c.detach().cpu().numpy()
                else:
                    y_pred_numpy_c = np.asarray(y_pred_step_c)
                
                if hasattr(y_true_step_c, 'detach') and hasattr(y_true_step_c, 'cpu') and hasattr(y_true_step_c, 'numpy'):
                    y_true_numpy_c = y_true_step_c.detach().cpu().numpy()
                else:
                    y_true_numpy_c = np.asarray(y_true_step_c)

                pred_heatmap_filename = os.path.join(run_plots_dir, f"step{current_step_idx}_channel{c}_pred_heatmap.png")
                true_heatmap_filename = os.path.join(run_plots_dir, f"step{current_step_idx}_channel{c}_true_heatmap.png")
                
                pred_title = f"Predicted Values Heatmap (Mode: {mode_name}, Step: {current_step_idx}, Channel: {c})"
                true_title = f"True Values Heatmap (Mode: {mode_name}, Step: {current_step_idx}, Channel: {c})"

                self.plot_value_heatmap(y_pred_numpy_c, pred_title, pred_heatmap_filename)
                self.plot_value_heatmap(y_true_numpy_c, true_title, true_heatmap_filename)

                MSE_c = self.MSE(y_pred_numpy_c, y_true_numpy_c)
                RMSE_c = self.RMSE(y_pred_numpy_c, y_true_numpy_c)
                MAE_c = self.MAE(y_pred_numpy_c, y_true_numpy_c)
                MAPE_c = self.MAPE(y_pred_numpy_c, y_true_numpy_c)
                F1_c = self.F1_score(y_pred_numpy_c, y_true_numpy_c)
                Precision_c = self.precision_score(y_pred_numpy_c, y_true_numpy_c)
                Recall_c = self.recall_score(y_pred_numpy_c, y_true_numpy_c)
                FPR_c = self.false_positive_rate(y_pred_numpy_c, y_true_numpy_c)
                Pearson_c = self.pearson_corr(y_pred_numpy_c, y_true_numpy_c)
                Spearman_c = self.spearman_corr(y_pred_numpy_c, y_true_numpy_c)
                print(f'MSE: {MSE_c:11.4f}, RMSE: {RMSE_c:9.4f}, MAE: {MAE_c:9.4f}, MAPE: {MAPE_c:9.4%}, F1: {F1_c:9.4f}, Precision: {Precision_c:9.4f}, Recall: {Recall_c:9.4f}, FPR: {FPR_c:9.4f}, Pearson: {Pearson_c:9.4f}, Spearman: {Spearman_c:9.4f}')
                MSEs.append(MSE_c)
                RMSEs.append(RMSE_c)
                MAEs.append(MAE_c)
                MAPEs.append(MAPE_c)
                F1s.append(F1_c)
                Precisions.append(Precision_c)
                Recalls.append(Recall_c)
                FPRs.append(FPR_c)
                Pearsons.append(Pearson_c)
                Spearmans.append(Spearman_c)
            else:
                continue

        step_metrics = dict()
        step_metrics['mse'] = np.mean(MSEs)
        step_metrics['rmse'] = np.mean(RMSEs)
        step_metrics['mae'] = np.mean(MAEs)
        step_metrics['mape'] = np.nanmean(MAPEs)
        step_metrics['f1'] = np.mean(F1s)
        step_metrics['precision'] = np.mean(Precisions)
        step_metrics['recall'] = np.mean(Recalls)
        step_metrics['fpr'] = np.mean(FPRs)
        step_metrics['pearson'] = np.mean(Pearsons)
        step_metrics['spearman'] = np.mean(Spearmans)
        
        print('Overall: \n'
              f'   MSE: {step_metrics["mse"]:10.4f}, RMSE: {step_metrics["rmse"]:9.4f} \n'
              f'   MAE: {step_metrics["mae"]:10.4f}, MAPE: {step_metrics["mape"]:9.4%} \n'
              f'   F1: {step_metrics["f1"]:10.4f}, Precision: {step_metrics["precision"]:9.4f} \n'
              f'   Recall: {step_metrics["recall"]:9.4f}, FPR: {step_metrics["fpr"]:9.4f} \n'
              f'   Pearson: {step_metrics["pearson"]:9.4f}, Spearman: {step_metrics["spearman"]:9.4f} \n')
        
        return step_metrics


    @staticmethod
    def MSE(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        return np.mean(np.square(y_pred - y_true))

    @staticmethod
    def RMSE(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    @staticmethod
    def MAE(y_pred: np.array, y_true: np.array):        
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def MAPE(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        greater_than_ = y_true > 10
        y_pred, y_true = y_pred[greater_than_], y_true[greater_than_]
        return np.mean(np.abs(y_pred - y_true) / np.abs(y_true))

    @staticmethod
    def F1_score(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred) # Ensure numpy array for F1 logic
        y_true = np.asarray(y_true) # Ensure numpy array for F1 logic
        y_pred_binary = (y_pred > 0).astype(np.int32)
        y_true_binary = (y_true > 0).astype(np.int32)
        
        true_positives = np.sum(y_pred_binary * y_true_binary)
        false_positives = np.sum(y_pred_binary * (1 - y_true_binary))
        false_negatives = np.sum((1 - y_pred_binary) * y_true_binary)
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return f1

    @staticmethod
    def precision_score(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_pred_binary = (y_pred > 0).astype(np.int32)
        y_true_binary = (y_true > 0).astype(np.int32)
        
        true_positives = np.sum(y_pred_binary * y_true_binary)
        false_positives = np.sum(y_pred_binary * (1 - y_true_binary))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        
        return precision

    @staticmethod
    def recall_score(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_pred_binary = (y_pred > 0).astype(np.int32)
        y_true_binary = (y_true > 0).astype(np.int32)
        
        true_positives = np.sum(y_pred_binary * y_true_binary)
        false_negatives = np.sum((1 - y_pred_binary) * y_true_binary)
        
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        return recall

    @staticmethod
    def false_positive_rate(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_pred_binary = (y_pred > 0).astype(np.int32)
        y_true_binary = (y_true > 0).astype(np.int32)
        
        false_positives = np.sum(y_pred_binary * (1 - y_true_binary))
        true_negatives = np.sum((1 - y_pred_binary) * (1 - y_true_binary))
        
        fpr = false_positives / (false_positives + true_negatives + 1e-10)
        
        return fpr

    @staticmethod
    def pearson_corr(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        
        correlation, _ = pearsonr(y_pred_flat, y_true_flat)
        
        if np.isnan(correlation):
            return 0.0
            
        return correlation

    @staticmethod
    def spearman_corr(y_pred: np.array, y_true: np.array):
        if hasattr(y_pred, 'detach') and hasattr(y_pred, 'cpu') and hasattr(y_pred, 'numpy'):
            y_pred = y_pred.detach().cpu().numpy()
        if hasattr(y_true, 'detach') and hasattr(y_true, 'cpu') and hasattr(y_true, 'numpy'):
            y_true = y_true.detach().cpu().numpy()
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        
        correlation, _ = spearmanr(y_pred_flat, y_true_flat)
        
        if np.isnan(correlation):
            return 0.0
            
        return correlation
