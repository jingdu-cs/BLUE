# BLUE: Bi-layer Heterogeneous Graph Fusion Network for Avian Influenza Forecasting

Accurate forecasting of avian influenza outbreaks within wild bird populations requires models that account for complex, multi-scale transmission patterns driven by various factors. Spatio-temporal models have recently gained traction for infection forecasting due to their ability to capture relational dynamics, but most existing frameworks rely solely on spatial connections at the county level. This overlooks valuable genetic information at the case level, which is essential for understanding how avian influenza spreads through epidemiological linkages beyond geography. We address this gap with \textit{BLUE}, a \textbf{B}i-\textbf{L}ayer heterogeneous graph f\textbf{U}sion n\textbf{E}twork designed to integrate genomic and spatial data for accurate outbreak forecasting. The framework 1) builds heterogeneous graphs from multiple information sources and multiple layers 2) smooths across relation types, 3) performs fusion at node and edge levels while retaining structural patterns, and  4) predicts future outbreaks via an autoregressive graph sequence model that captures transmission dynamics over time.  To facilitate further research, we introduce \textbf{Aivan-NA} dataset, the public dataset for avian influenza outbreak forecasting in the United States, incorporating genetic, spatial, and ecological data across 3227 counties. BLUE achieves superior performance over existing baselines, highlighting the value of incorporating multi-modal information into infectious disease forecasting.



## 🗂️ Repository structure

```
├── HeteroGraphNetwork.py   # Core GNN layers + fusion gates
├── FullHeteroGNN.py        # Wrapper that assembles encoder/decoder + MRF
├── MRF.py                  # Markov Random Field smoothing module
├── simple_graph_dataset.py # Windowed time‑series dataset loader
├── metrics.py              # MAE / RMSE / PCC utilities
├── Model_metrics.py        # Extra post‑hoc evaluation helpers
├── spactral_simple_main.py # Train / eval entry‑point
└── requirements.txt        # (create with conda‑env export)
```

> **Tip:** Each file starts with extensive docstrings – read them for a deeper dive into implementation details.

---

## ⚙️ Installation

```bash
# 1. Clone the repo
$ git clone https://github.com/<your‑handle>/FusionGNN.git
$ cd FusionGNN

# 2. Create environment (CUDA 11.8 example)
$ conda create -n fusiongnn python=3.10 -y
$ conda activate fusiongnn

# 3. Install PyTorch + PyG (adjust CUDA version if needed)
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch, re; print(re.sub('[+]', '', torch.__version__.split('+')[0]))")+cu118.html

# 4. Other Python deps
$ pip install -r requirements.txt  # numpy, pandas, scikit‑learn, tqdm, tensorboard, etc.
```

---

## 📄 Dataset preparation

FusionGNN expects **weekly graphs** that already combine all raw data into PyG `HeteroData` pickle files:

```text
processed_graphs/
    2017_W01.pkl  # HeteroData with nodes: {county, case} & edges: {(county, spatial, county), ...}
    2017_W02.pkl
    ...
```

Each file **must** contain:

* Node types: `"county"`, `"case"`
* Edge types: `(county, spatial, county)`, `(case, genetic, case)`, `(case, assignment, county)`
* Node feature tensors named `x`
* A node‑level infection count attribute `count` (target to predict)

If you do not yet have these tensors, check out the separate *data‑preprocessing* repo or adapt your own pipeline to output the above format.

---

## 🚀 Quick start

```bash
python spactral_simple_main.py \
  --dataset avian \
  --data_dir ./processed_graphs \
  --model_type full \
  --window_size 4 \
  --pred_horizon 4 \
  --hidden_dim 128 \
  --epochs 200 \
  --batch_size 16 \
  --lr 1e-3 \
  --use_mrf \
  --spectral_reg
```

| Flag             | Meaning                           | Default |
| ---------------- | --------------------------------- | ------- |
| `--window_size`  | $w$ historic weeks fed to encoder | `4`     |
| `--pred_horizon` | $h$ weeks to forecast             | `4`     |
| `--use_mrf`      | Enable MRF smoothing              | *False* |
| `--spectral_reg` | Add Laplacian loss                | *False* |
| `--k_folds`      | Cross‑validation folds            | `5`     |

`spactral_simple_main.py --help` prints the full list.

During training the script will output fold‑wise **MAE / RMSE / MAPE** and save the best model to `./checkpoints/`.

---

## 📊 Evaluation metrics

The following metrics are computed (see `metrics.py`):

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **MAPE** – Mean Absolute Percentage Error
* **R²** – Coefficient of determination (optional)

After training, aggregate scores across folds are stored in `results.csv`.

---

## 🔬 Research usage

If you use FusionGNN in academic work **please cite**:

```bibtex
@misc{FusionGNN2025,
  author       = {Jing Du and collaborators},
  title        = {FusionGNN: Bi‑layer Heterogeneous Graph Fusion for Epidemic Forecasting},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/<your‑handle>/FusionGNN}
}
```

---

## 🤝 Contributing

1. Fork the repo & create your branch: `git checkout -b feature/awesome`
2. Commit your changes following **conventional‑commit** style.
3. Ensure `pre‑commit` passes `ruff`, `black`, `isort`, and `pytest`.
4. Push to the branch and open a Pull Request.

We welcome bug fixes, new datasets, and model improvements!

---

## 📄 License

FusionGNN is released under the **MIT License**.  See the [LICENSE](LICENSE) file for details.

---

## 🌱 Acknowledgements

* Built with [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
* Spatial and genetic distance matrices courtesy of the *Avian Influenza Genomics Project*.
* This work was supported by the Macquarie University **AI in Health** initiative.

---

Feel free to open an issue if you encounter any problem or have a feature request.
