# Money Mule Detection in Financial Transaction Networks

> A Graph Neural Network-based Anti-Money Laundering (AML) system using
> Ensemble GNNs, Quantum-Inspired Optimization, and Explainability layers
> for detecting suspicious money mule networks in financial transaction graphs.

***

## 📌 Project Status

🚧 **Active Development** — Setting up environment and base pipeline. (More changes have been made will update after my shit is done)

***

## 🎯 Project Overview

Money mule networks are a critical component of modern money laundering operations. Traditional rule-based AML systems fail to detect the complex, evolving patterns used by criminal networks.

This project builds a **hybrid detection framework** that combines:

- **Graph Neural Networks (GNNs)** — to model account-transaction relationships as a directed weighted graph
- **Ensemble Learning** — combining GCN, GAT, and GraphSAGE for robust, high-recall fraud detection
- **Quantum-Inspired Optimization** — QUBO-based subgraph candidate ranking for intelligent suspicious account prioritization
- **Explainability (SHAP/LIME/Captum)** — compliance-ready decision transparency for each flagged account

***

## 🏗️ System Architecture

```
Financial Transaction Data (Elliptic / IBM AML)
               ↓
     [ Graph Construction ]
      NetworkX + PyTorch Geometric
               ↓
     [ GNN Ensemble Layer ]
      GCN  +  GAT  +  GraphSAGE
               ↓
  [ Quantum-Inspired Optimizer ]
   QUBO-based subgraph risk ranking
               ↓
     [ Explainability Layer ]
      SHAP  +  Captum  +  PyVis
               ↓
     [ Detection Output ]
  Risk scores + Visualization Dashboard
```

***

## 📂 Project Structure

```
money-mule-detection/
├── data/
│   ├── raw/                    # Elliptic + IBM AML raw datasets (not committed)
│   └── processed/              # Cleaned, preprocessed outputs
├── notebooks/                  # EDA and experiment notebooks
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── cleaning_v2.py
│   │   └── labels.py          # Data loading, cleaning, feature engineering
│   ├── graph/
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── account_features.py
│   │   ├── account_features_v2.py
│   │   ├── account_features_v3.py
│   │   ├── finalize_features.py
│   │   ├── profile_features_v2.py
│   │   ├── profile_features_v3.py
│   │   └── verify_final_features.py                # Graph construction (NetworkX + PyG)
│   └── models/
│       ├── __init__.py
│       ├── architectures.py
│       ├── prepare_data.py
│       ├── gcn_baseline.py
│       ├── train_gnn.py
│       ├── train_gnn_sampled.py
│       ├── ensemble_predictions.py
│       ├── ensemble_voting.py
│       ├── tuned_voting.py
│       ├── stacking.py
│       ├── threshold_selection.py
│       ├── phase_6_experiment_summary.md
│       └── phase_6_experiment_summary_values.csv                 # GCN, GAT, GraphSAGE, GIN model definitions
│   ├── ensemble/               # Soft voting, stacking logic
│   ├── explainability/         # SHAP, LIME, Captum integration
│   ├── quantum/                # Quantum-inspired scoring layer
│   └── utils/                  # Logging, helpers, config
├── app/
│   ├── backend/                # FastAPI endpoints
│   └── frontend/               # Streamlit dashboard
├── results/                    # Saved models, metrics, plots
├── requirements.txt
├── README.md
```

***

## 🧪 Datasets Used

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Elliptic Bitcoin Dataset | [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set) | 203K nodes, 234K edges | GNN model benchmarking |

> ⚠️ Datasets are **NOT committed** to this repository due to size and licensing constraints.
> See [`data/README.md`](data/README.md) for download and setup instructions.

***

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11.x |
| Graph ML | PyTorch Geometric | 2.5.x |
| Deep Learning | PyTorch | 2.3.0 |
| Graph Analysis | NetworkX | 3.3 |
| GNN Models | GCN, GAT, GraphSAGE, GIN | via PyG |
| Quantum-Inspired | PennyLane, Qiskit | 0.36.x / 1.1.x |
| Explainability | SHAP, LIME, Captum | 0.45.x / 0.2.x / 0.7.x |
| Visualization | Plotly, PyVis, Seaborn | 5.22.x / 0.3.x / 0.13.x |
| Backend | FastAPI + Uvicorn | 0.111.x / 0.30.x |
| Frontend | Streamlit | 1.35.x |
| ML Baselines | Scikit-learn, XGBoost | 1.4.x / 2.0.x |

***

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/money-mule-detection.git
cd money-mule-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate — Mac/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch and PyTorch Geometric (separately)

```bash
# CPU version (works on all machines)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# GPU version (if you have an NVIDIA GPU with CUDA 12.1)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.5.3
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

### 5. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in any required configuration values
```

### 6. Download Datasets

See [`data/README.md`](data/README.md) for full dataset download and placement instructions.

***

## 🚀 Running the Project

```bash
# Run Streamlit dashboard (frontend)
streamlit run app/frontend/dashboard.py

# Run FastAPI backend
uvicorn app.backend.main:app --reload

# Open Jupyter notebooks for experiments
jupyter notebook notebooks/
```

***

## 📊 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision (illicit class) | Minimize false positive alerts |
| Recall (illicit class) | Catch all real money mules |
| F1-Score | Balance precision and recall |
| ROC-AUC | Threshold-independent performance |
| PR-AUC | Best metric for imbalanced datasets |
| Macro F1 | Overall multi-class performance |
| SHAP Feature Importance | Explainability and compliance |

***

## 🔬 Reference Papers

1. Ferretti et al. — *Graph-Based AML Detection in Financial Networks* **(Base Paper)**
2. Haider, Noreen, Salman et al. — *Towards Quantum-Ready Blockchain Fraud Detection via Ensemble Graph Neural Networks*, IEEE BCCA 2025
3. Braine, Egger, Glick, Woerner — *Quantum Algorithms for Mixed Binary Optimization Applied to Transaction Settlement*, IEEE TQE 2021
4. Rashid & Hayat — *AMLGaurd: Graph-Based Money Laundering Detection in Financial Networks*, IEEE ECCE 2025
5. Rao et al. — *Quantum Algorithms for Solving Large-Scale Optimization Problems: Challenges and Breakthroughs*, IEEE ICoICI 2025
6. Hadinata et al. — *Generating Synthetic Anomaly Graph Network Dataset for AML Prediction Using GAN*, 2025
7. Kansal et al. — *Clean Code Against Dirty Cash: A Survey on Anti-Money Laundering Techniques*, 2025

***

## 🗺️ Development Roadmap

- [x] Repository setup and environment configuration
- [x] Requirements and tech stack finalized
- [x] Reference papers reviewed and mapped
- [x] **Phase 1** — Elliptic dataset pipeline + GNN models (GCN, GAT, GraphSAGE)
- [x] **Phase 2** — Ensemble layer (soft voting + stacking)
- [x] **Phase 3** — Explainability layer (SHAP + Captum + PyVis)
- [x] **Phase 4** — Quantum-inspired optimization scoring layer
- [x] **Phase 5** — Dashboard + FastAPI integration
- [x] **Phase 6** — Ensemble                ← NEXT
- [ ] **Phase 7** — Evaluation              ⏳
- [ ] **Phase 8** — Explainability          ⏳
- [ ] **Phase 9** — Quantum-inspired        ⏳
- [ ] **Phase 10** — Integration            ⏳

***

## 👥 Team

| Name | Role |
|---|---|
| [Team Member 1] | Graph Construction + GNN Models |
| [Team Member 2] | Quantum-Inspired Layer + Backend |
| [Team Member 3] | Frontend + Explainability |

> 📢 **Project Guide:** Dr. Siva Sundara Pandian
> 🏫 **Institution:** ACS College of Engineering

***

## 🤝 Contributing

This repository is currently **team-only**. If you are a collaborator:

1. Never push directly to `main`
2. Create a branch for your feature: `git checkout -b feature/your-feature-name`
3. Write clear, descriptive commit messages
4. Raise a **Pull Request** and get at least one review before merging
5. Keep your branch up to date with `main` before raising a PR

***

## 📄 License

This project is developed for **academic purposes** as part of a university major project.
All dataset usage complies with respective dataset licenses. (kraggle)

***

<div align="center">
  <sub>Built with 🔬 for financial security research | ACS College of Engineering, 2026</sub>
</div>
