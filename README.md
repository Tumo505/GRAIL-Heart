# GRAIL-Heart: Graph-based Reconstruction of Artificial Intercellular Links

<p align="center">
  <img src="https://bioicons.com/icons/cc-by-3.0/Human_physiology/Servier/heart.svg" width="100" alt="heart icon"/>
</p>

<p align="center">
  <strong>A Graph Neural Network framework for causal ligand-receptor analysis in cardiac spatial transcriptomics</strong>
</p>

<p align="center">
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6137179">📄 Preprint</a> •
  <a href="https://tumo505.github.io/GRAIL-Heart/outputs/cytoscape/index.html">🕸️ Network Explorer</a> •
  <a href="#web-application">🖥️ Web App</a> •
  <a href="docs/METHODOLOGY.md">📖 Methods</a> •
  <a href="docs/RESULTS.md">📊 Results</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/>
  <img src="https://img.shields.io/badge/status-published-brightgreen" alt="Status"/>
</p>

---

## Highlights

- **Forward + Inverse modelling** — predict active L-R interactions *and* identify which ones causally drive cell fate
- **Expression-gated causal scoring** — 90th-percentile aggregation avoids score collapse common in naïve approaches
- **Mechanosensitive pathway enrichment** — YAP/TAZ (p = 6.7×10⁻²⁰), BMP (p = 2.2×10⁻²³), TGF-β (p = 3.3×10⁻¹¹)
- **OmniPath integration** — 22,234 curated L-R pairs from CellPhoneDB, CellChat, ICELLNET and 10+ databases
- **Interactive web application** — upload your own spatial transcriptomics data, run inference, visualise results
- **12.3M parameter model** — 4-layer GAT, 4-head attention, 256-d hidden, trained on Heart Cell Atlas v2

## Overview

GRAIL-Heart combines deep graph neural networks with inverse modelling to move beyond correlative L-R co-expression and identify **causal** cell-cell communication signals in the heart.

| Modelling Mode | Input | Output | Question Answered |
|----------------|-------|--------|-------------------|
| **Forward** | Expression + Spatial | L-R predictions | *Which L-R interactions are active?* |
| **Inverse** | Observed cell fates | Causal L-R scores | *Which signals drove this differentiation?* |

The model is trained on the **Heart Cell Atlas v2** — 42,654 cells across six Visium spatial transcriptomics regions (Apex, Left Atrium, Left Ventricle, Right Atrium, Right Ventricle, Septum).

## Performance

### Final Model (Epoch 154 / 200)

| Metric | Value |
|--------|-------|
| Reconstruction R² | **0.988** |
| Pearson Correlation | **0.997** |
| L-R AUROC | **0.876** |
| L-R AUPRC | **1.000** |
| Cell Type Accuracy | **0.875** |
| F1 Score | **0.933** |
| Parameters | 12,242,493 (12.3 M) |

### Leave-One-Region-Out Cross-Validation (6-fold)

| Metric | Mean ± Std | Best Region |
|--------|-----------|-------------|
| L-R AUROC | 0.762 ± 0.183 | RV (0.970) |
| L-R AUPRC | 0.969 ± 0.028 | RV (0.999) |
| Recon R² | 0.879 ± 0.110 | LV (0.975) |
| Pearson Correlation | 0.991 ± 0.005 | LV (0.997) |
| Accuracy | 0.910 ± 0.049 | RV (0.990) |
| F1 Score | 0.951 ± 0.029 | RV (0.995) |

<details>
<summary><b>Per-Region Breakdown</b></summary>

| Region | AUROC | AUPRC | R² | Pearson | Accuracy | F1 |
|--------|:-----:|:-----:|:--:|:-------:|:--------:|:--:|
| AX (Apex) | 0.933 | 0.988 | 0.947 | 0.992 | 0.885 | 0.936 |
| LA (Left Atrium) | 0.568 | 0.941 | 0.720 | 0.984 | 0.922 | 0.959 |
| LV (Left Ventricle) | 0.844 | 0.958 | **0.975** | **0.997** | 0.827 | 0.899 |
| RA (Right Atrium) | 0.472 | 0.927 | 0.727 | 0.985 | 0.916 | 0.956 |
| RV (Right Ventricle) | **0.970** | **0.999** | 0.963 | 0.995 | **0.990** | **0.995** |
| SP (Septum) | 0.788 | 0.999 | 0.942 | 0.994 | 0.920 | 0.958 |

</details>

### Benchmark Comparison

| Method | L-R AUROC | L-R AUPRC | Recon R² | Parameters |
|--------|-----------|-----------|----------|------------|
| **GRAIL-Heart** | **0.977** | **1.000** | **0.876** | 12.3 M |
| GraphSAGE | 0.841 | 0.999 | −0.079 | 1.6 M |
| MLP | 0.839 | 0.999 | −0.103 | 2.5 M |
| GCN | 0.807 | 0.998 | −0.109 | 1.4 M |
| Single-Task GAT | 0.804 | 0.998 | −0.886 | 1.4 M |
| CellPhoneDB | 0.624 | 0.994 | — | — |
| CellChat | 0.429 | 0.993 | — | — |

### External Validation (5 Independent Datasets)

The trained model was applied **without re-training** to 5 external datasets spanning two species and multiple conditions:

| Dataset | Source | Cells | Species | Condition | Top Causal L-R | Score |
|---------|--------|------:|---------|-----------|----------------|:-----:|
| Kawasaki LCWE | GSE178799 | 1,195 | Mouse | Disease | LGALS1→COL1A2 | 0.993 |
| Kawasaki PBS | GSE178799 | 2,030 | Mouse | Control | TTN→MYBPC3 | 0.988 |
| E18 Mouse 10k | 10x Genomics | 8,028 | Mouse | Developmental | APP→APLP2 | 0.385 |
| E18 Mouse 1k | 10x Genomics | 668 | Mouse | Developmental | MDK→NCL | 0.380 |
| HCA Apex | Heart Cell Atlas v2 | 6,497 | Human | Healthy | B2M→CALR | 0.757 |

**Key generalisability findings:**
- **L-R consistency:** Spearman ρ = 0.953 (E18 pair), 0.828 (Kawasaki pair), 0.44–0.68 (cross-species) — all p < 1e-8
- **TGF-β replicates everywhere** — significant (p < 1e-4) in all 5 external datasets
- **Disease rewiring** — Kawasaki vasculitis upregulates HMOX1→CXCL10 (log₂FC = +16.3), downregulates complement CFD→C3 (p = 1.1×10⁻²⁵)
- **Integrin/FAK** significant in spatial data (HCA AX, p = 1.5e-3)

<details>
<summary><b>L-R Consistency Matrix (Spearman ρ)</b></summary>

| | Kawasaki LCWE | Kawasaki PBS | E18 10k | E18 1k | HCA AX |
|---|:---:|:---:|:---:|:---:|:---:|
| Kawasaki LCWE | 1.000 | **0.828** | 0.597 | 0.644 | 0.684 |
| Kawasaki PBS | 0.828 | 1.000 | 0.674 | 0.685 | 0.623 |
| E18 10k | 0.597 | 0.674 | 1.000 | **0.953** | 0.439 |
| E18 1k | 0.644 | 0.685 | 0.953 | 1.000 | 0.582 |
| HCA AX | 0.684 | 0.623 | 0.439 | 0.582 | 1.000 |

</details>

## Key Biological Findings

### Top Causal L-R Interactions by Region

| Region | Top Causal L-R | Causal Score | Pathway |
|--------|----------------|:------------:|---------|
| AX (Apex) | TIMP1 → MMP2 | 0.827 | ECM Regulator |
| LA (Left Atrium) | SERPING1 → C1S | 0.812 | Complement Regulator |
| LV (Left Ventricle) | CFD → C3 | 0.804 | Complement |
| RA (Right Atrium) | TIMP2 → MMP2 | 0.796 | ECM Regulator |
| RV (Right Ventricle) | TIMP1 → MMP2 | 0.819 | ECM Regulator |
| SP (Septum) | THBS1 → FN1 | 0.781 | ECM |

### Mechanosensitive Pathway Enrichment

| Pathway | AX | LA | **LV** | RA | RV | SP |
|---------|:--:|:--:|:------:|:--:|:--:|:--:|
| **YAP/TAZ** | 1.7e-11 | 6.9e-14 | **6.7e-20** | 7.8e-18 | 1.4e-13 | 4.8e-11 |
| **BMP** | 5.0e-13 | 4.2e-16 | **2.2e-23** | 6.3e-22 | 7.0e-16 | 2.6e-14 |
| **TGF-β** | 5.7e-07 | 4.5e-08 | **3.3e-11** | 3.7e-10 | 6.5e-08 | 2.3e-07 |

The Left Ventricle shows the strongest mechanosensitive signature, consistent with its role as the primary contractile chamber.

## Architecture

```
Gene Expression [N × 2000]
        │
        ▼
Gene Encoder [512 → 256]  ─────┐
                                ├── Multi-Modal Fusion [384 → 256]
Spatial Encoder [2D → 64]  ────┤
Cell Type Embed [64]  ─────────┘
        │
        ▼
Edge-Type-Aware GAT (4 layers × 4 heads × 256d)
        │
        ▼
Jumping Knowledge Concatenation
        │
   ┌────┼────┬──────────┬─────────────┐
   ▼    ▼    ▼          ▼             ▼
 L-R  Recon  Cell     Fate       Pathway
 Head  Head  Type    Predictor   Activation
              Head  (Inverse)   (Mechano)
```

## Project Structure

```
GRAIL-Heart/
├── src/grail_heart/          # Core Python package
│   ├── data/                 #   Data loading, graph construction, OmniPath L-R
│   ├── models/               #   GNN architecture, inverse modelling, encoders
│   ├── training/             #   Trainer, multi-task losses, metrics
│   ├── evaluation/           #   Evaluation utilities
│   └── visualization/        #   Spatial network visualisation
├── app/                      # Streamlit web application (6 tabs)
├── configs/                  # YAML configurations (default, cv, ablation)
├── data/                     # Datasets — Heart Cell Atlas v2 Visium h5ad files
├── outputs/                  # Checkpoints, figures, tables, inverse analysis
├── docs/                     # Methodology, results, paper draft, manuscript guide
├── train.py                  # Standard training
├── train_cv.py               # LORO cross-validation
├── enhanced_inference.py     # Enhanced inference pipeline
├── inverse_inference.py      # Inverse modelling analysis
├── benchmark_comparison.py   # Baseline benchmarks
├── ablation_study.py         # Architecture ablation
├── validate_external.py      # External validation & generalisability
├── Dockerfile                # Docker image
└── docker-compose.yml        # Docker Compose deployment
```

## Installation

### Quick Install

```bash
pip install grail-heart
```

### From Source

```bash
git clone https://github.com/Tumo505/GRAIL-Heart.git
cd GRAIL-Heart
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -e ".[all]"
```

## Usage

### Training

```bash
# Standard training (200 epochs, inverse modelling enabled)
python train.py --config configs/default.yaml

# Leave-One-Region-Out cross-validation
python train_cv.py --config configs/cv.yaml
```

### Inference

```bash
# Enhanced inference (forward + inverse, all 6 regions)
python enhanced_inference.py

# Inverse modelling with mechanosensitive pathway analysis
python inverse_inference.py

# External validation on independent datasets
python validate_external.py --config configs/validation.yaml
```

### Web Application

```bash
# Local
cd app && streamlit run app.py

# Docker
docker compose up -d    # → http://localhost:8501
```

Accepts `.h5ad`, `.h5`, `.csv`, `.tsv`. Ensembl ID → HGNC symbol resolution is automatic.

### Python API

```python
import torch
from grail_heart.models import GRAILHeart

ckpt = torch.load("outputs/checkpoints/best.pt", map_location="cpu")
model = GRAILHeart(**ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    out = model(data)

# Forward: out["lr_scores"], out["reconstruction"]
# Inverse: out["causal_scores"], out["fate_logits"], out["pathway_activation"]
```

## Configuration

Key training parameters (`configs/default.yaml`):

```yaml
model:
  hidden_dim: 256
  n_gat_layers: 4
  n_heads: 4
  n_edge_types: 2          # spatial + L-R
  dropout: 0.2
  use_inverse_modelling: true
  n_pathways: 20
  n_mechano_pathways: 8

training:
  n_epochs: 200
  learning_rate: 0.0001
  weight_decay: 0.01
  mixed_precision: true
  grad_clip: 1.0
```

## Docker Deployment

```bash
# Build and start
docker compose up -d --build

# Access at http://localhost:8501
```


## Acknowledgements

- [Heart Cell Atlas v2](https://www.heartcellatlas.org/) — spatial transcriptomics data
- [OmniPath](https://omnipathdb.org/) — CellPhoneDB, CellChat, ICELLNET L-R databases
- [MSigDB](https://www.gsea-msigdb.org/) — Hallmark pathway gene sets
- [PyTorch Geometric](https://pyg.org/), [Scanpy](https://scanpy.readthedocs.io/), [AnnData](https://anndata.readthedocs.io/), [Bioicons](https://bioicons.com/)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
