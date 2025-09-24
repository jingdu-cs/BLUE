# BLUE: Bi-layer Heterogeneous Graph Fusion Network for Avian Influenza Forecasting

Accurate forecasting of avian influenza outbreaks within wild bird populations requires models that account for complex, multi-scale transmission patterns driven by various factors. Spatio-temporal GNN-based models have recently gained traction for infection forecasting due to their ability to capture relations and flow between spatial regions, but most existing frameworks rely solely on spatial regions and their connections. This overlooks valuable genetic information at the case level, such as cases in one region being genetically descended from strains in another, which is essential for understanding how infectious diseases spread through epidemiological linkages beyond geography.We address this gap with **BLUE**, a **B**i-**L**ayer heterogeneous graph f**U***sion n**E**twork designed to integrate genetic, spatial, and ecological data for accurate outbreak forecasting.
The framework 1) builds heterogeneous graphs from multiple information sources and multiple layers,2) smooths across relation types, 3) performs fusion while retaining structural patterns, and 4) predicts future outbreaks via an autoregressive graph sequence model that captures transmission dynamics over time. To facilitate further research, we introduce **Avian-US** dataset, the dataset for avian influenza outbreak forecasting in the United States, incorporating genetic, spatial, and ecological data across locations. BLUE achieves superior performance over existing baselines, highlighting the value of incorporating multi-layer information into infectious disease forecasting.



## 🗂️ Repository structure

```
├── HeteroGraphNetwork.py         # BLUE implementation
├── spectral_loss.py              # Spectral Alignment loss implementation
├── laplacian.py                  # Laplacian Matrix implementation
├── MRF.py                        # Markov Random Field smoothing module
├── simple_graph_dataset.py       # Windowed time‑series dataset loader
├── metrics.py                    # Prediction process
├── Model_metrics.py              # MAE / RMSE / PCC /SCC / F1 Score evaluation metrics
├── spectral_simple_main.py       # Train / Val / Eval entry‑point
└── requirements.txt              # environments
```


## ⚙️ Installation

```bash
# 1. Clone the repo

# 2. Create environment (CUDA 11.8 example)
$ conda create -n blue python=3.10
$ conda activate blue

# 3. Install PyTorch + PyG (adjust CUDA version if needed)
$ pip install -r requirements.txt  # numpy, pandas, scikit‑learn, tqdm, tensorboard, pytorch, torch_geometric, scatter, etc.
```

---

## 📄 Dataset preparation

BLUE expects **weekly graphs** that already combine all raw data into PyG `HeteroData` pickle files
Each file **must** contain:

* Node types: `"county"`, `"case"`
* Edge types: `(county, spatial, county)`, `(case, genetic, case)`, `(case, assignment, county)`
* Node feature tensors named `x`
* County‑level attributes [`infected count`, 'abundance']
* Case-level attributes ['importance']


## 🚀 Quick start

```bash
python spectral_simple_main.py \
    --dataset avian \
    --batch_size 4 \
    --window_size 4 \
    --pred_horizon 4 \
    --hidden_dim 16 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --epochs 100 \
    --spectral_gamma 0.9 \
    --loss_type 'infection_weighted' \
    --use_eigenvalue_constraint True \
    --eigenvalue_loss_type cosine_similarity \
    --spectral_k 10 \
    --infection_zero_weight 1.0 \
    --infection_low_weight 8.0 \
    --infection_med_weight 15.0 \
    --infection_high_weight 25.0 \
    --infection_low_threshold 1.0 \
    --infection_med_threshold 5.0 \
    --infection_high_threshold 20.0
```

| Flag             | Meaning                           | Default |
| ---------------- | --------------------------------- | ------- |
| --window_size    | $w$ historic weeks fed to encoder | 4       |
| --pred_horizon   | $h$ weeks to forecast             | 4       |
| --num_mrf        | MRF smoothing layers              | 1       |
| --spectral_gamma | weight of spectral approximation  | 0.9     |

`spectral_simple_main.py --help` prints the full list.

During training the script will output fold‑wise **MAE / RMSE / F1 / Pearson / Spearman** and save the best model to `./save_results/`.

---

## 📊 Evaluation metrics

The following metrics are computed (see `metrics.py`):

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **Pearson** - Pearson correlations
* **Spearman** - Spearman correlations
* **F1 Score** - F1 Score

We adapted parts of the implementation (data split and evaluation metrics) from the open-source GitHub repository [EAST-Net] (https://github.com/underdoc-wang/EAST-Net), modifying it to fit the specific requirements of our study.


