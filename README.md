# GRAIL-Heart: Graph-based Reconstruction of Artificial Intercellular Links

A Graph Neural Network framework for analyzing cell-cell communication in cardiac spatial transcriptomics data.

## Overview

GRAIL-Heart is a deep learning model designed to discover and analyze ligand-receptor (L-R) interactions in spatial transcriptomics datasets, with a focus on cardiac tissue. The framework integrates:

- **Gene expression encoding** with neural networks
- **Spatial information** through positional embeddings
- **Graph attention mechanisms** for neighborhood analysis
- **Multi-task learning** for simultaneous prediction of L-R interactions, gene expression reconstruction, and cell type classification

The model is trained on the Heart Cell Atlas v2, comprising spatial transcriptomics data from six distinct cardiac regions (Apex, Left Atrium, Left Ventricle, Right Atrium, Right Ventricle, and Septum).

## Key Features

- Multi-task learning framework balancing L-R prediction, reconstruction, and classification
- Edge-type aware Graph Attention Networks with spatial and L-R edge types
- **OmniPath L-R database integration** (22,000+ curated pairs from CellPhoneDB, CellChat, ICELLNET)
- **Leave-One-Region-Out (LORO) cross-validation** for robust generalization assessment
- Comprehensive spatial visualization of cell-cell communication networks
- Mixed precision training for efficient GPU utilization
- Complete inference pipeline with cross-region analysis

## Cross-Validation Results

GRAIL-Heart was evaluated using 6-fold Leave-One-Region-Out cross-validation:

| Metric | Mean | ± Std | Interpretation |
|--------|------|-------|----------------|
| **Reconstruction R²** | 0.885 | 0.105 | Excellent gene expression reconstruction |
| **Pearson Correlation** | 0.991 | 0.005 | Near-perfect correlation |
| **L-R AUROC** | 0.745 | 0.221 | Good L-R prediction |
| **L-R AUPRC** | 0.970 | 0.030 | Excellent precision-recall |
| **Accuracy** | 0.925 | 0.061 | Very high classification accuracy |
| **F1 Score** | 0.958 | 0.036 | Excellent balance |

### Per-Region Performance

| Region | R² | AUROC | AUPRC |
|--------|-----|-------|-------|
| RV (Right Ventricle) | 0.962 | **0.985** | 0.9998 |
| LA (Left Atrium) | 0.952 | 0.896 | 0.987 |
| AX (Apex) | 0.966 | 0.882 | 0.977 |
| LV (Left Ventricle) | 0.956 | 0.826 | 0.948 |
| RA (Right Atrium) | 0.737 | 0.423 | 0.915 |
| SP (Septum) | 0.736 | 0.456 | 0.995 |

## Project Structure

```
GRAIL-Heart/
├── src/grail_heart/                  # Main package
│   ├── data/                         # Data loading and preprocessing
│   │   ├── datasets.py              # Spatial transcriptomics dataset loader
│   │   ├── graph_builder.py         # Spatial graph construction (kNN, radius, Delaunay)
│   │   ├── lr_database.py           # Standard L-R database
│   │   ├── expanded_lr_database.py  # Extended database with 500+ pairs
│   │   └── cellchat_database.py     # OmniPath integration (22,000+ pairs)
│   ├── models/                       # Neural network architectures
│   │   ├── encoders.py              # Gene and spatial encoders
│   │   ├── gat_layers.py            # Graph Attention layers with edge type awareness
│   │   ├── grail_heart.py           # Main GRAIL-Heart model
│   │   ├── predictors.py            # Prediction heads for L-R, reconstruction, classification
│   │   └── reconstruction.py        # Gene expression decoders
│   ├── training/                     # Training utilities
│   │   ├── losses.py                # Multi-task loss functions
│   │   ├── metrics.py               # Evaluation metrics
│   │   ├── trainer.py               # Training loop and checkpointing
│   │   └── contrastive.py           # Contrastive learning modules
│   ├── utils/                        # Utility functions
│   └── visualization/                # Spatial visualization tools
│       └── spatial_viz.py           # Network and L-R visualization
├── configs/
│   ├── default.yaml                  # Default configuration file
│   └── cv.yaml                       # Cross-validation configuration
├── data/                             # Data directory (not in repo)
│   └── HeartCellAtlasv2/            # Heart Cell Atlas v2 datasets
├── outputs/                          # Training outputs
│   ├── checkpoints/                 # Model checkpoints
│   ├── logs/                        # TensorBoard logs
│   ├── analysis/                    # Network analysis outputs
│   └── enhanced_analysis/           # Enhanced inference results
├── docs/                             # Documentation
│   ├── METHODOLOGY.md               # Detailed methods
│   └── RESULTS.md                   # Results and findings
├── notebooks/                        # Jupyter notebooks
├── train.py                          # Standard training script
├── train_cv.py                       # Cross-validation training script
├── enhanced_inference.py             # Enhanced inference pipeline
├── evaluate_test.py                  # Model evaluation script
├── check_checkpoint.py               # Checkpoint inspection utility
└── README.md                         # This file
```

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Tumo505/GRAIL-Heart.git
cd GRAIL-Heart
```

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision  # Install PyTorch with CUDA support
pip install torch-geometric
pip install omnipath           # For L-R database
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- PyTorch Geometric
- OmniPath (L-R database access)
- Scanpy (single-cell analysis)
- AnnData (data container)
- Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- PyYAML (configuration)
- TensorBoard (logging)

## Data Preparation

### Download Heart Cell Atlas v2

Download the Visium spatial transcriptomics files from the Heart Cell Atlas:

```bash
# Create data directory
mkdir -p data/HeartCellAtlasv2

# Download datasets (approximately 30GB total)
# Place .h5ad files in data/HeartCellAtlasv2/
# Expected files:
# - visium-OCT_AX_raw.h5ad  (Apex)
# - visium-OCT_LA_raw.h5ad  (Left Atrium)
# - visium-OCT_LV_raw.h5ad  (Left Ventricle)
# - visium-OCT_RA_raw.h5ad  (Right Atrium)
# - visium-OCT_RV_raw.h5ad  (Right Ventricle)
# - visium-OCT_SP_raw.h5ad  (Septum)
```

### Data Format

Input data should be in AnnData format (.h5ad):
- `adata.X`: Expression matrix [n_cells × n_genes]
- `adata.obsm['spatial']`: Spatial coordinates [n_cells × 2]
- `adata.obs`: Cell metadata (including cell types if available)
- `adata.var`: Gene names and features

The framework will automatically:
- Select top 2,000 highly variable genes
- Normalize library sizes
- Apply log1p transformation
- Filter cells (min 200 genes) and genes (min 3 cells)

## Usage

### Training with Cross-Validation (Recommended)

Run Leave-One-Region-Out cross-validation for robust evaluation:

```bash
# Full 6-fold CV
python train_cv.py --config configs/cv.yaml

# Quick test (specific folds, fewer epochs)
python train_cv.py --config configs/cv.yaml --n_epochs 50 --folds "AX,LA,LV"

# Run specific regions only
python train_cv.py --config configs/cv.yaml --folds "RV,SP"
```

### Standard Training

Train the GRAIL-Heart model with default configuration:

```bash
python train.py --config configs/default.yaml
```

Train with custom data directory:

```bash
python train.py --config configs/default.yaml --data_dir /path/to/data
```

### Configuration Parameters

Key parameters in `configs/cv.yaml`:

```yaml
model:
  hidden_dim: 256           # Embedding dimension
  n_gat_layers: 3          # Number of GAT layers
  n_heads: 8               # Attention heads
  n_edge_types: 2          # Spatial + L-R edges
  encoder_dims: [512, 256] # Gene encoder hidden dims
  dropout: 0.1             # Dropout rate
  decoder_type: residual   # Expression decoder type

data:
  max_lr_pairs: 5000       # Limit L-R pairs (memory optimization)

training:
  n_epochs: 100            # Training epochs
  learning_rate: 0.0001    # Adam learning rate
  weight_decay: 0.01       # L2 regularization
  batch_size: 1            # Full graph per batch
  grad_clip: 1.0           # Gradient clipping threshold
  mixed_precision: true    # Use AMP training

loss:
  lr_weight: 1.0           # L-R prediction weight
  recon_weight: 1.0        # Reconstruction weight
  cell_type_weight: 1.0    # Cell type classification weight
  contrastive_weight: 0.5  # Contrastive learning weight
```

### Inference

Run enhanced inference with OmniPath L-R database:

```bash
python enhanced_inference.py \
  --checkpoint outputs/checkpoints/best.pt \
  --data_dir data/HeartCellAtlasv2 \
  --output_dir outputs/enhanced_analysis
```

This generates:
- L-R interaction scores (CSV tables)
- Spatial network visualizations (PNG figures)
- Interaction networks (JSON files)
- Cross-region comparison analysis

### Evaluation

Evaluate model on test set:

```bash
python evaluate_test.py
```

## Ligand-Receptor Database

### OmniPath Integration

GRAIL-Heart integrates with OmniPath, providing access to multiple curated L-R databases:

| Source Database | Description |
|-----------------|-------------|
| CellPhoneDB | Comprehensive L-R interactions |
| CellChat | Cell-cell communication database |
| ICELLNET | Intercellular communication |
| Ramilowski2015 | Literature-curated interactions |
| And more... | 10+ integrated sources |

**Database Statistics:**
- Raw interactions: 115,064
- Unique L-R pairs: 22,234
- Unique ligands: 2,284
- Unique receptors: 2,637
- Pathway categories: 16

### Usage

```python
from grail_heart.data.cellchat_database import get_omnipath_lr_database

# Load full database
lr_pairs = get_omnipath_lr_database()
print(f"Loaded {len(lr_pairs)} L-R pairs")

# With caching
lr_pairs = get_omnipath_lr_database(cache_path="data/lr_database_cache.csv")
```

## Model Architecture

### Overview

```
Input (Gene expression + Spatial coordinates)
    |
    v
Gene Expression Encoder [512 -> 256]
    |
    +-----------> Spatial Position Encoder [2D -> 64D]
    |
    v
Multi-Modal Encoder (concatenate + project)
    |
    v
Graph Attention Network Stack (3 layers, 8 heads)
    |
    v
Jumping Knowledge Concatenation
    |
    +---------> L-R Interaction Head (Bilinear)
    +---------> Gene Expression Decoder (Residual)
    +---------> Cell Type Classifier
    +---------> Signaling Network Predictor
```

### Multi-Task Learning

Total loss function:

```
L_total = w_lr * L_lr + w_recon * L_recon + w_ct * L_ct + w_contr * L_contr

where:
- L_lr: Focal binary cross-entropy for L-R prediction
- L_recon: Combined MSE + Cosine + Correlation loss
- L_ct: Cross-entropy for cell type classification
- L_contr: InfoNCE contrastive learning loss
```

## Outputs

### Cross-Validation Outputs

```
outputs/cv_TIMESTAMP/
├── config.yaml              # Configuration used
├── cv_results.yaml          # Aggregated CV metrics
├── cv_results.json          # JSON format results
├── fold_0_AX/               # Fold 0 (held out AX)
│   ├── checkpoints/
│   │   └── best.pt
│   └── val_metrics.yaml
├── fold_1_LA/               # Fold 1 (held out LA)
├── ...
└── fold_5_SP/               # Fold 5 (held out SP)
```

### Inference Outputs

```
outputs/enhanced_analysis/
├── tables/
│   ├── AX_lr_scores.csv
│   ├── LA_lr_scores.csv
│   ├── ... (one per region)
│   └── cross_region_comparison.csv
├── figures/
│   ├── AX_spatial_network.png
│   ├── AX_lr_heatmap.png
│   ├── AX_pathway_activity.png
│   ├── ... (multiple per region)
│   ├── cross_region_lr_heatmap.png
│   ├── region_comparison_panels.png
│   └── network_summary_dashboard.png
└── networks/
    ├── AX_network.json
    └── ... (one per region)
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**: Comprehensive description of methods, model architecture, cross-validation strategy, and L-R database
- **[docs/RESULTS.md](docs/RESULTS.md)**: Detailed results, CV metrics, biological findings, and figures

## Troubleshooting

### GPU Memory Issues

If you encounter out-of-memory errors:

1. Use CV config with limited L-R pairs:
   ```bash
   python train_cv.py --config configs/cv.yaml  # Uses 5000 L-R pairs
   ```

2. Reduce L-R pairs further in `configs/cv.yaml`:
   ```yaml
   data:
     max_lr_pairs: 2000  # Smaller model
   ```

3. Reduce model hidden dimension or number of GAT layers

### Missing Data Files

Ensure all required .h5ad files are in `data/HeartCellAtlasv2/` directory with correct naming:
- visium-OCT_AX_raw.h5ad
- visium-OCT_LA_raw.h5ad
- visium-OCT_LV_raw.h5ad
- visium-OCT_RA_raw.h5ad
- visium-OCT_RV_raw.h5ad
- visium-OCT_SP_raw.h5ad

### CUDA Errors

If CUDA is not detected:

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU training (slower)
# Edit configs/cv.yaml: hardware.device: cpu
```

## Acknowledgments

- Heart Cell Atlas v2 dataset
- OmniPath database (CellPhoneDB, CellChat, ICELLNET)
- PyTorch Geometric framework
- Scanpy and AnnData communities

## Citation

If you use GRAIL-Heart in your research, please cite:

```bibtex
@software{grail_heart,
  author = {Tumo Kgabeng, Lulu Wang, Harry Ngwangwa, Thanyani Pandelani},
  title = {GRAIL-Heart: Graph-based Reconstruction of Artificial Intercellular Links},
  year = {2026},
  url = {https://github.com/Tumo505/GRAIL-Heart}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

*Last updated: January 2026*
