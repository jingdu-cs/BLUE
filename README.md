# BLUE: Bi-layer Heterogeneous Graph Fusion Network for Avian Influenza Forecasting

Accurate forecasting of avian influenza outbreaks within wild bird populations requires models that account for complex, multi-scale transmission patterns driven by various factors. Spatio-temporal models have recently gained traction for infection forecasting due to their ability to capture relational dynamics, but most existing frameworks rely solely on spatial connections at the county level. This overlooks valuable genetic information at the case level, which is essential for understanding how avian influenza spreads through epidemiological linkages beyond geography. We address this gap with \textit{BLUE}, a \textbf{B}i-\textbf{L}ayer heterogeneous graph f\textbf{U}sion n\textbf{E}twork designed to integrate genomic and spatial data for accurate outbreak forecasting. The framework 1) builds heterogeneous graphs from multiple information sources and multiple layers 2) smooths across relation types, 3) performs fusion at node and edge levels while retaining structural patterns, and  4) predicts future outbreaks via an autoregressive graph sequence model that captures transmission dynamics over time.  To facilitate further research, we introduce \textbf{Aivan-NA} dataset, the public dataset for avian influenza outbreak forecasting in the United States, incorporating genetic, spatial, and ecological data across 3227 counties. BLUE achieves superior performance over existing baselines, highlighting the value of incorporating multi-modal information into infectious disease forecasting.



## 🗂️ Repository structure

```
├── HeteroGraphNetwork.py   # BLUE implementation
├── MRF.py                  # Markov Random Field smoothing module
├── simple_graph_dataset.py # Windowed time‑series dataset loader
├── metrics.py              # Prediction process
├── Model_metrics.py        # MAE / RMSE / PCC /F1 Score evaluation metrics
├── spectral_simple_main.py # Train / Val / Eval entry‑point
└── requirements.txt        # environments
```


## ⚙️ Installation

```bash
# 1. Clone the repo
$ git clone https://github.com/<your‑handle>/FusionGNN.git
$ cd FusionGNN

# 2. Create environment (CUDA 11.8 example)
$ conda create -n fusiongnn python=3.10
$ conda activate fusiongnn

# 3. Install PyTorch + PyG (adjust CUDA version if needed)
$ pip install -r requirements.txt  # numpy, pandas, scikit‑learn, tqdm, tensorboard, pytorch, torch_geometric, scatter, etc.
```

---

## 📄 Dataset preparation

FusionGNN expects **weekly graphs** that already combine all raw data into PyG `HeteroData` pickle files
Each file **must** contain:

* Node types: `"county"`, `"case"`
* Edge types: `(county, spatial, county)`, `(case, genetic, case)`, `(case, assignment, county)`
* Node feature tensors named `x`
* A node‑level infection count attribute `count` (target to predict)


## 🚀 Quick start

```bash
python spectral_simple_main.py \
  --dataset 'avian' \
  --data_dir './processed_graphs' \
  --model_type 'FusionGNN' \
  --window_size 4 \
  --pred_horizon 4 \
  --hidden_dim 8 \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-5 \
  --num_mrf 1 \
  --spectral_gamma 0.1
```

| Flag             | Meaning                           | Default |
| ---------------- | --------------------------------- | ------- |
| --window_size    | $w$ historic weeks fed to encoder | 4       |
| --pred_horizon   | $h$ weeks to forecast             | 4       |
| --num_mrf        | MRF smoothing layers              | 1       |
| --spectral_gamma | weight of spectral approximation  | 0.1     |

`spectral_simple_main.py --help` prints the full list.

During training the script will output fold‑wise **MAE / RMSE / MAPE / F1 / Pearson** and save the best model to `./save_results/`.

---

## 📊 Evaluation metrics

The following metrics are computed (see `metrics.py`):

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **MAPE** – Mean Absolute Percentage Error
* **Pearson** - Pearson correlations
* **F1 Score** - F1 Score

We adapted parts of the implementation (data split and evaluation metrics) from the open-source GitHub repository [EAST-Net] (https://github.com/underdoc-wang/EAST-Net), modifying it to fit the specific requirements of our study.


