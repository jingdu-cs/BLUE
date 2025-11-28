# Genomic-Informed Heterogeneous Graph Learning for Spatiotemporal Avian Influenza Outbreak Forecasting

Accurate forecasting of Avian Influenza Virus (AIV) outbreaks within wild bird populations necessitates models that explicitly account for complex, multi-scale transmission patterns driven by diverse factors. While conventional spatiotemporal epidemic models are robust for human-centric diseases, they rely on **spatial homophily** and **diffusive transmission** between geographic regions. This simplification is incomplete for AIV as it neglects valuable genetic information critical for capturing dynamics like high-frequency reassortment and lineage turnover at the case level (e.g., genetic descent across regions). These epidemiological linkages beyond geography are essential for understanding AIV spread. To address these limitations, we systematically formulate the AIV forecasting problem and propose **BLUE**(**B**i-**L**ayer genomic-aware heterogeneous graph f**U**sion pipelin**E**). This pipeline integrates genetic, spatial, and ecological data to achieve highly accurate outbreak forecasting. It 1) defines a multi-layered graph structure incorporating information from diverse sources and multiple layers (case and location), 2) applies cross-relation smoothing to smooth information flow across edge types, 3) performs graph fusion that preserves critical structural patterns backed by theoretical spectral guarantees, and 4) forecasts future outbreaks using an autoregressive graph sequence model to capture transmission dynamics. To support research, we release the Avian-US dataset, which provides comprehensive genetic, spatial, and ecological data on US avian influenza outbreaks. BLUE demonstrates superior performance over existing baselines, highlighting the efficacy of integrating multi-layer information for infectious disease forecasting.



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


