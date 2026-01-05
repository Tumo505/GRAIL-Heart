# GRAIL-Heart: Methodology

## Graph-based Reconstruction of Artificial Intercellular Links for Cardiac Spatial Transcriptomics

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources and Preprocessing](#data-sources-and-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Graph Construction](#graph-construction)
5. [Ligand-Receptor Database](#ligand-receptor-database)
6. [Multi-Task Learning Framework](#multi-task-learning-framework)
7. [Loss Functions](#loss-functions)
8. [Training Strategy](#training-strategy)
9. [Cross-Validation](#cross-validation)
10. [Inference Pipeline](#inference-pipeline)
11. [Inverse Modelling](#inverse-modelling)

---

## Overview

GRAIL-Heart is a graph neural network (GNN) framework designed specifically for analyzing cardiac spatial transcriptomics data. The model integrates:

- **Gene expression encoding**: Deep neural networks to transform high-dimensional gene expression into compact representations
- **Spatial information**: Positional encoding to capture tissue architecture
- **Graph attention mechanisms**: Multi-head attention over cellular neighborhoods
- **Multi-task prediction**: Simultaneous learning of ligand-receptor interactions, gene expression reconstruction, and cell type classification

The framework enables comprehensive analysis of cell-cell communication patterns across different cardiac regions, providing insights into the molecular mechanisms underlying cardiac physiology and pathology.

---

## Data Sources and Preprocessing

### Primary Dataset: Heart Cell Atlas v2

We utilized spatial transcriptomics data from the Heart Cell Atlas v2, comprising Visium datasets from six distinct cardiac regions:

| Region | Abbreviation | Cells | Description |
|--------|--------------|-------|-------------|
| Apex | AX | 6,497 | Ventricular apex |
| Left Atrium | LA | 5,822 | Left atrial tissue |
| Left Ventricle | LV | 9,626 | Left ventricular wall |
| Right Atrium | RA | 7,027 | Right atrial tissue |
| Right Ventricle | RV | 5,039 | Right ventricular wall |
| Septum | SP | 8,643 | Interventricular septum |
| **Total** | | **42,654** | |

### Data Preprocessing Pipeline

1. **Quality Control Filtering**
   - Minimum cells per gene: 3
   - Minimum genes per cell: 200

2. **Gene Selection**
   - Top 2,000 highly variable genes selected for analysis
   - Selection based on variance-stabilized expression values

3. **Normalization**
   - Library size normalization
   - Log1p transformation (`log(x + 1)`)

4. **Configuration Parameters**
   ```yaml
   data:
     n_top_genes: 2000
     normalize: true
     log_transform: true
     min_cells: 3
     min_genes: 200
   ```

---

## Model Architecture

### Gene Expression Encoder

The gene expression encoder transforms raw gene counts into latent representations:

```
Input: Gene expression [N cells × 2000 genes]
    ↓
Linear(2000 → 512) + BatchNorm + GELU + Dropout(0.1)
    ↓
Linear(512 → 256) + BatchNorm + GELU + Dropout(0.1)
    ↓
Linear(256 → 256)
    ↓
Output: Cell embeddings [N × 256]
```

**Key Design Choices:**
- GELU activation for smooth gradients
- Batch normalization for training stability
- Dropout (10%) for regularization
- Xavier initialization for weight matrices

### Multi-Modal Encoder

When spatial coordinates are available, we employ a multi-modal encoder that fuses:

1. **Gene expression features** (256-dim)
2. **Spatial position encoding** (64-dim, sinusoidal)
3. **Cell type embeddings** (64-dim, when available)

Fusion is performed via concatenation followed by a projection layer.

### Graph Attention Network (GAT)

The core of GRAIL-Heart is an edge-type-aware Graph Attention Network:

**Architecture Parameters:**
| Parameter | Value |
|-----------|-------|
| Hidden dimension | 256 |
| Number of GAT layers | 3 |
| Attention heads | 8 |
| Edge types | 2 (spatial, L-R) |
| Dropout | 0.1 |

**Edge-Type Aware Attention:**

For edge type $t$, the attention coefficient between nodes $i$ and $j$ is:

$$\alpha_{ij}^{(t)} = \text{softmax}_j \left( \frac{(W_Q^{(t)} h_i)^T (W_K^{(t)} h_j)}{\sqrt{d_k}} \right)$$

Where:
- $h_i, h_j$ are node embeddings
- $W_Q^{(t)}, W_K^{(t)}$ are type-specific query and key projections
- $d_k$ is the key dimension (hidden_dim / n_heads)

**Jumping Knowledge (JK) Connection:**

We employ concatenation-based jumping knowledge to aggregate representations from all GAT layers:

$$h_i^{final} = [h_i^{(1)} || h_i^{(2)} || h_i^{(3)}]$$

### Prediction Heads

#### 1. Ligand-Receptor Interaction Predictor

Bilinear attention for interaction scoring between cell pairs:

$$s_{ij} = h_i^T W h_j + b$$

Where $W$ is a learned bilinear weight matrix.

#### 2. Gene Expression Decoder (Residual)

Multi-layer decoder with residual connections for expression reconstruction:

```
Input: Cell embeddings [N × 256]
    ↓
Linear(256 → 256) + BatchNorm + GELU + Dropout
    ↓
Linear(256 → 512) + BatchNorm + GELU + Dropout
    ↓
Linear(512 → 2000) + Softplus
    ↓
Output: Reconstructed expression [N × 2000]
```

#### 3. Cell Type Classifier

Two-layer MLP for cell type prediction:

```
Linear(256 → 256) + GELU + Dropout(0.1) + Linear(256 → n_types)
```

#### 4. Signaling Network Predictor

Attention-based adjacency matrix prediction with signaling type classification:
- **Adjacency matrix**: Softmax-normalized attention scores
- **Signaling types**: Autocrine, paracrine, endocrine (3-class classification)

---

## Graph Construction

### Spatial Graph Building

We construct graphs from spatial coordinates using k-nearest neighbors (kNN):

**Algorithm:**
1. Build KD-Tree from spatial coordinates
2. Query k=6 nearest neighbors for each spot
3. Create bidirectional edges
4. Compute edge weights based on Euclidean distance:

$$w_{ij} = \exp\left(-\frac{||pos_i - pos_j||_2^2}{\sigma^2}\right)$$

**Graph Construction Options:**
| Method | Description |
|--------|-------------|
| kNN | k-nearest neighbors (default, k=6) |
| Radius | All neighbors within radius threshold |
| Delaunay | Delaunay triangulation edges |

### Edge Types

The graph contains two edge types:
1. **Spatial edges (type=0)**: Based on physical proximity
2. **L-R edges (type=1)**: Based on ligand-receptor co-expression

---

## Ligand-Receptor Database

### OmniPath Integration

GRAIL-Heart integrates with **OmniPath**, a comprehensive resource that aggregates multiple curated ligand-receptor databases:

| Source Database | Description | Contribution |
|-----------------|-------------|--------------|
| CellPhoneDB | Literature-curated L-R interactions | Major source |
| CellChat | Cell-cell communication database | Major source |
| ICELLNET | Intercellular communication | Major source |
| Ramilowski2015 | Systematic L-R curation | Supporting |
| HPMR | Human Plasma Membrane Receptome | Supporting |
| Guide2Pharma | IUPHAR/BPS pharmacology | Supporting |
| 10+ others | Various curated sources | Supporting |

### Database Statistics

| Metric | Value |
|--------|-------|
| Raw interactions downloaded | 115,064 |
| **Unique L-R pairs** | **22,234** |
| Unique ligands | 2,284 |
| Unique receptors | 2,637 |
| Pathway categories | 16 |

### Implementation

```python
from grail_heart.data.cellchat_database import get_omnipath_lr_database

# Load full database from OmniPath
lr_pairs = get_omnipath_lr_database(
    min_curation_effort=0,  # Include all curated pairs
    min_sources=1,          # At least 1 source database
    cache_path="data/lr_database_cache.csv"
)

# Each pair contains:
# - ligand: Gene symbol of ligand
# - receptor: Gene symbol of receptor
# - pathway: Pathway category
# - function: Biological function
# - sources: Source databases
# - n_sources: Number of supporting databases
# - is_stimulation: Activating interaction
# - is_inhibition: Inhibiting interaction
```

### Memory-Efficient Mode

For GPUs with limited memory, the database can be subsampled:

```yaml
# In configs/cv.yaml
data:
  max_lr_pairs: 5000  # Use top 5000 pairs by curation confidence
```

This prioritizes pairs with more supporting source databases.

### L-R Score Computation

For each L-R pair, we compute interaction scores:

$$score_{L-R}(i,j) = \sigma(h_i^T W_{LR} h_j) \cdot \sqrt{expr_L(i) \cdot expr_R(j)}$$

Where:
- $h_i, h_j$ are learned cell embeddings
- $W_{LR}$ is the bilinear weight matrix
- $expr_L(i)$ is ligand expression in cell $i$
- $expr_R(j)$ is receptor expression in cell $j$

---

## Multi-Task Learning Framework

GRAIL-Heart performs simultaneous optimization of multiple objectives:

### Task 1: Ligand-Receptor Interaction Prediction

**Objective:** Predict whether cell pairs engage in L-R interactions

**Loss:** Focal Binary Cross-Entropy
$$\mathcal{L}_{LR} = -\alpha(1-p_t)^\gamma \log(p_t)$$

### Task 2: Gene Expression Reconstruction

**Objective:** Reconstruct original expression from learned embeddings

**Loss:** Combined loss (MSE + Cosine + Correlation)
$$\mathcal{L}_{recon} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{cosine} + \lambda_2 \mathcal{L}_{corr}$$

### Task 3: Cell Type Classification

**Objective:** Classify cells into known cell types

**Loss:** Cross-Entropy with label smoothing

### Task 4: Contrastive Learning

**Objective:** Learn discriminative embeddings for similar/dissimilar cells

**Loss:** InfoNCE contrastive loss

---

## Inverse Modelling Framework

### Motivation

The abstract states:
> "This inverse modelling framework will also elucidate mechanosensitive pathways that modulate early-stage cardiac tissue patterning, linking molecular signalling to the formation of soft contractile structures."

While the **forward modelling** (described above) predicts L-R interactions from expression data, **inverse modelling** answers the causal question:

**"Which L-R signals are responsible for driving cell differentiation?"**

### Forward vs Inverse Modelling

| Aspect | Forward Modelling | Inverse Modelling |
|--------|-------------------|-------------------|
| **Input** | Expression + Spatial | Observed Fate/Phenotype |
| **Output** | L-R predictions | Causal L-R signals |
| **Question** | "What L-R interactions are active?" | "What signals CAUSED this?" |
| **Use case** | Prediction | Causal inference |

### Inverse Modelling Components

GRAIL-Heart implements four inverse modelling components:

#### 1. Cell Fate Prediction Head

Maps L-R interaction patterns to cell differentiation states:

```
L-R Scores → L-R Encoder → Neighborhood Aggregation → Fate Classifier
                                    ↓
                            Fate Trajectory Encoder
                                    ↓
                         Differentiation Scorer
```

**Outputs:**
- `fate_logits`: Discrete fate predictions [N, n_fates]
- `fate_trajectory`: Continuous differentiation embedding [N, 64]
- `differentiation_score`: Scalar differentiation progress [N, 1]

#### 2. Counterfactual Reasoner

Identifies **causal** L-R interactions through counterfactual analysis:

**Key Question:** "If we removed this L-R interaction, would the cell fate change?"

**Approach:**
1. Compute baseline fate predictions
2. Apply integrated gradients attribution
3. Identify L-R edges whose removal changes fate

```python
# Gradient-based causal attribution
∂fate/∂lr_score → causal_importance
```

**Output:** `causal_lr_scores` [E] - Causal importance of each L-R edge

#### 3. L-R to Target Gene Decoder

Predicts downstream gene expression changes from L-R interactions:

$$L-R \rightarrow Pathway \; Activation \rightarrow Target \; Gene \; Effects$$

**Architecture:**
```
L-R Features → lr_to_pathway → pathway_gene_matrix → Gene Effects
                              ↓
               Learned Pathway-Gene Associations
```

**Pathways Modeled:**
- ECM/Adhesion
- NOTCH signaling
- TGF-β pathway
- Wnt signaling
- Chemokine signaling
- Growth factor signaling
- And more (20 total)

#### 4. Mechanosensitive Pathway Module

Models cardiac mechanobiology as described in the abstract:

**Mechanosensitive Pathways:**
| Pathway | Biological Role |
|---------|-----------------|
| YAP/TAZ | Hippo signaling, stiffness sensing |
| Integrin-FAK | Cell-ECM mechanotransduction |
| Piezo1/2 | Mechanosensitive ion channels |
| TGF-β | Mechanical stress response |
| Wnt | Cardiac development |
| Notch | Cell-cell signaling |
| BMP | Cardiac patterning |
| FGF | Growth factor signaling |

**Computation:**
```python
# L-R features with spatial context
lr_features = [z_src, z_dst_spatial, lr_score]

# Map to mechanosensitive pathway activation
mechano_activation = sigmoid(lr_to_mechano(lr_features))

# Compute pathway crosstalk
pathway_crosstalk = pathway_interaction(mechano_activation)

# Effect on differentiation
diff_effect = pathway_to_diff(mechano_activation)
```

### Inverse Inference Pipeline

The complete inverse inference pipeline:

```python
# 1. Run forward model
outputs = model(data, run_inverse=True)

# Forward outputs
lr_scores = outputs['lr_scores']           # Which L-R are active
reconstruction = outputs['reconstruction']  # Reconstructed expression

# Inverse outputs
fate_logits = outputs['fate_logits']       # Predicted cell fates
causal_lr = outputs['causal_lr_scores']    # Causal L-R importance
mechano = outputs['mechano_pathway_activation']  # Mechanobiology

# 2. True inverse inference (fate → signals)
inverse_results = model.infer_causal_signals(
    observed_fate=observed_fate_labels,
    observed_expression=expression_matrix,
    z=embeddings,
    edge_index=edges,
)
inferred_lr = inverse_results['inferred_lr_importance']
```

### Inverse Modelling Loss Functions

Training the inverse modelling components uses additional losses:

$$\mathcal{L}_{inverse} = w_{fate}\mathcal{L}_{fate} + w_{causal}\mathcal{L}_{causal} + w_{diff}\mathcal{L}_{diff} + w_{gene}\mathcal{L}_{gene}$$

| Loss | Weight | Description |
|------|--------|-------------|
| $\mathcal{L}_{fate}$ | 0.5 | Cross-entropy for fate classification |
| $\mathcal{L}_{causal}$ | 0.3 | Sparsity regularization for causal scores |
| $\mathcal{L}_{diff}$ | 0.2 | MSE for differentiation scores |
| $\mathcal{L}_{gene}$ | 0.3 | Target gene prediction accuracy |

### Differentiation Staging

For the differentiation loss ($\mathcal{L}_{diff}$), GRAIL-Heart automatically computes differentiation staging using a hierarchical approach:

#### 1. Pseudotime Analysis (Primary)
If pseudotime has been pre-computed in the dataset (`dpt_pseudotime` in `obs`), or if neighbors are available, GRAIL-Heart computes diffusion pseudotime:

```python
# Automatic computation in datasets.py
sc.tl.diffmap(adata)
sc.tl.dpt(adata)
```

#### 2. Cell Type-Based Ordering (Fallback)
When pseudotime is unavailable, GRAIL-Heart uses a cardiac-specific differentiation hierarchy:

| Stage | Cell Types | Score |
|-------|-----------|-------|
| Progenitor | progenitor, stem | 0.1 |
| Mesenchymal | mesenchymal, fibroblast | 0.2-0.3 |
| Supporting | endothelial, immune, macrophage | 0.3-0.4 |
| Smooth Muscle | SMC, pericyte | 0.5 |
| Atrial CM | atrial, aCM | 0.7 |
| Ventricular CM | ventricular, vCM, cardiomyocyte | 0.85-0.9 |
| Conduction | purkinje, nodal | 1.0 |

This hierarchy reflects the developmental trajectory from cardiac progenitors to terminally differentiated cells.

### Enabling Inverse Modelling

```python
from grail_heart.models import GRAILHeart

model = GRAILHeart(
    n_genes=2000,
    hidden_dim=256,
    n_cell_types=10,
    # Enable inverse modelling
    use_inverse_modelling=True,
    n_fates=10,           # Number of fate categories
    n_pathways=20,        # Number of signaling pathways
    n_mechano_pathways=8, # Number of mechanosensitive pathways
)
```

### Running Inverse Analysis

```bash
# Run enhanced inference (recommended)
python enhanced_inference.py

# Or run inverse inference with full mechanobiology analysis
python inverse_inference.py
```

#### Inverse Inference Pipeline

The `inverse_inference.py` script performs comprehensive inverse modelling analysis including:
1. **Mechanosensitive pathway analysis** - p-value testing for 8 key pathways
2. **Causal L-R interaction scoring** - identifying causally important cell-cell signals
3. **Cell fate prediction** - differentiation staging via diffusion pseudotime
4. **Network visualization** - causal network graphs per region

**Output Structure:**
```
outputs/inverse_analysis/
├── inverse_modelling_summary.json    # Master summary with all metrics
├── mechanobiology/
│   ├── AX_mechano_pathways.json      # Per-region mechanosensitive pathway p-values
│   ├── LA_mechano_pathways.json
│   ├── LV_mechano_pathways.json
│   ├── RA_mechano_pathways.json
│   ├── RV_mechano_pathways.json
│   └── SP_mechano_pathways.json
├── causal_analysis/
│   ├── AX_causal_lr.csv              # Per-region causal L-R interaction scores
│   ├── LA_causal_lr.csv
│   ├── LV_causal_lr.csv
│   ├── RA_causal_lr.csv
│   ├── RV_causal_lr.csv
│   └── SP_causal_lr.csv
├── fate_prediction/
│   ├── AX_fate_analysis.json         # Cell fate differentiation metrics
│   ├── LA_fate_analysis.json
│   ├── LV_fate_analysis.json
│   ├── RA_fate_analysis.json
│   ├── RV_fate_analysis.json
│   └── SP_fate_analysis.json
└── figures/
    ├── mechano_pathway_heatmap.png   # Cross-region mechanobiology heatmap
    ├── AX_causal_network.png         # Causal network visualization
    ├── LA_causal_network.png
    ├── LV_causal_network.png
    ├── RA_causal_network.png
    ├── RV_causal_network.png
    └── SP_causal_network.png
```

#### Enhanced Inference Outputs

The `enhanced_inference.py` script generates comprehensive analysis to `outputs/enhanced_analysis/`:

```
outputs/enhanced_analysis/
├── analysis_report.txt      # Summary statistics and top causal interactions
├── tables/
│   ├── cross_region_comparison.csv   # L-R scores across all regions
│   ├── {region}_lr_scores.csv        # Per-region L-R rankings
├── figures/
│   ├── cross_region_lr_heatmap.png   # Cross-region comparison
│   ├── region_comparison_panels.png  # Multi-panel overview
│   ├── network_summary_dashboard.png # Summary dashboard
│   ├── {region}_spatial_network.png  # Spatial network layout
│   ├── {region}_lr_heatmap.png       # L-R heatmap
│   ├── {region}_pathway_activity.png # Pathway activity bars
│   └── {region}_{ligand}_{receptor}.png  # Top interaction visualizations (56 total)
├── causal_analysis/         # NEW: Inverse modelling outputs
│   └── {region}_causal_edges.csv     # Per-edge causal scores
└── networks/                # Network files (JSON/GraphML)
```

#### Causal Analysis Output Format

Each `{region}_causal_edges.csv` contains:
| Column | Description |
|--------|-------------|
| `edge_idx` | Edge index in the spatial graph |
| `causal_score` | Score [0,1] indicating causal importance for cell fate |

High causal scores (>0.8) indicate L-R interactions that are causally responsible for driving cell differentiation, not just co-expressed.

#### Inverse Modelling Causal Analysis Output Format

From `outputs/inverse_analysis/causal_analysis/{region}_causal_lr.csv`:
| Column | Description |
|--------|-------------|
| `edge_idx` | Edge index in the spatial graph |
| `source_cell` | Source cell identifier |
| `target_cell` | Target cell identifier |
| `causal_score` | Score [0,1] indicating causal importance for cell fate |
| `lr_score` | L-R interaction probability from model |
| `region` | Cardiac region identifier |

#### Mechanosensitive Pathway Analysis

The inverse modelling tests 8 mechanosensitive signaling pathways:
| Pathway | Description | Key Genes |
|---------|-------------|-----------|
| YAP_TAZ | Hippo pathway mechanotransduction | YAP1, WWTR1, TEAD1-4 |
| Integrin_FAK | Focal adhesion signaling | ITGA5, ITGAV, PTK2 |
| Piezo | Mechanosensitive ion channels | PIEZO1, PIEZO2 |
| TGF_beta | Fibrosis/remodeling | TGFB1-3, TGFBR1-2 |
| Wnt | Development/regeneration | WNT1-11, FZD1-10 |
| Notch | Cell fate determination | NOTCH1-4, JAG1-2, DLL1-4 |
| BMP | Bone morphogenetic protein | BMP2-7, BMPR1A/B |
| FGF | Fibroblast growth factor | FGF1-23, FGFR1-4 |

P-values indicate pathway enrichment significance (stored in `outputs/inverse_analysis/mechanobiology/{region}_mechano_pathways.json`).

#### Key Metrics from Enhanced Inference

| Metric | Description |
|--------|-------------|
| `mean_score` | Average L-R interaction score across cells |
| `max_score` | Maximum interaction score (strongest signal) |
| `total_score` | Sum of all interaction scores |
| `n_interactions` | Number of cell pairs with this interaction |
| `pct_edges` | Percentage of edges showing interaction |

---

## Loss Functions

### Total Loss

$$\mathcal{L}_{total} = w_{LR} \mathcal{L}_{LR} + w_{recon} \mathcal{L}_{recon} + w_{ct} \mathcal{L}_{cell\_type} + w_{contr} \mathcal{L}_{contrastive}$$

### Loss Weights

| Loss Component | Weight |
|----------------|--------|
| L-R Prediction | 1.0 |
| Reconstruction | 1.0 |
| Cell Type | 1.0 |
| Contrastive | 0.5 |
| Signaling | 0.1 |
| KL Divergence | 0.001 |

### Reconstruction Loss (Combined)

```python
combined_loss = (
    mse_loss(pred, target) +                    # MSE component
    0.3 * (1 - cosine_similarity(pred, target)) +  # Cosine component  
    0.3 * (1 - pearson_correlation(pred, target))  # Correlation component
)
```

---

## Training Strategy

### Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.01 |
| Batch size | 1 (full graph) |
| Epochs | 100 (CV) / 200 (standard) |
| Gradient clipping | 1.0 |

### Learning Rate Schedule

**Cosine annealing** with warmup:
- Warmup epochs: 5-10
- Minimum LR: 1e-6
- Cosine decay over remaining epochs

### Early Stopping

| Parameter | Value |
|-----------|-------|
| Patience | 15-25 epochs |
| Monitor metric | val_total_loss |
| Mode | minimize |

### Mixed Precision Training

- Enabled FP16 training via PyTorch AMP
- Gradient scaling for numerical stability

### Checkpointing

- Save every 5-10 epochs
- Save best model based on validation loss
- Save final model state

---

## Cross-Validation

### Leave-One-Region-Out (LORO) Cross-Validation

To robustly assess model generalization, we implement Leave-One-Region-Out cross-validation across the 6 cardiac regions:

| Fold | Validation Region | Training Regions |
|------|------------------|------------------|
| 1 | AX (Apex) | LA, LV, RA, RV, SP |
| 2 | LA (Left Atrium) | AX, LV, RA, RV, SP |
| 3 | LV (Left Ventricle) | AX, LA, RA, RV, SP |
| 4 | RA (Right Atrium) | AX, LA, LV, RV, SP |
| 5 | RV (Right Ventricle) | AX, LA, LV, RA, SP |
| 6 | SP (Septum) | AX, LA, LV, RA, RV |

### Implementation

```bash
# Full 6-fold CV
python train_cv.py --config configs/cv.yaml

# Run specific folds
python train_cv.py --config configs/cv.yaml --folds "AX,LA,LV"

# Customize epochs
python train_cv.py --config configs/cv.yaml --n_epochs 50
```

### Metrics Aggregation

For each fold, we compute:
- Reconstruction: R², Pearson correlation, RMSE, MAE
- L-R Prediction: AUROC, AUPRC, Accuracy, F1
- Cell Type: Accuracy, F1

Final metrics are reported as **mean ± std** across all 6 folds.

### Why LORO Cross-Validation?

1. **Limited Data**: Only 6 spatial graphs available
2. **Region Heterogeneity**: Each cardiac region has distinct biology
3. **True Generalization**: Tests if model works on unseen regions
4. **Publication Standard**: Required for scientific claims

---

## Inference Pipeline

### Enhanced Inference Process

1. **Load Trained Model**
   - Load checkpoint from `outputs/checkpoints/best.pt`
   - Initialize model with matching architecture

2. **Load L-R Database**
   - Initialize OmniPath database (22,234 pairs)
   - Filter to expressed genes in dataset

3. **Per-Region Processing**
   ```python
   for region in ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']:
       # Load h5ad file
       adata = load_region_data(region)
       
       # Build spatial graph
       graph = build_graph(adata.X, adata.obsm['spatial'])
       
       # Run model inference
       embeddings, lr_scores, recon = model(graph)
       
       # Compute L-R interaction scores
       scores = compute_lr_scores(embeddings, lr_database)
       
       # Generate visualizations
       visualize_spatial_network(adata, scores)
   ```

4. **Cross-Region Analysis**
   - Compare L-R patterns across cardiac regions
   - Identify region-specific signaling
   - Generate comprehensive reports

### Output Files

| Output Type | Location |
|-------------|----------|
| Network JSON | `outputs/analysis/{region}_network.json` |
| L-R Scores CSV | `outputs/enhanced_analysis/tables/{region}_lr_scores.csv` |
| Spatial Figures | `outputs/enhanced_analysis/figures/` |
| Cross-Region Comparison | `outputs/enhanced_analysis/tables/cross_region_comparison.csv` |

---

## Implementation Details

### Software Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- OmniPath
- Scanpy
- AnnData
- NumPy, Pandas, Matplotlib, Seaborn

### Hardware Configuration

- GPU: CUDA-enabled (recommended)
- Mixed precision: FP16/FP32
- Memory: 12GB+ GPU RAM recommended
- With 5000 L-R pairs: ~335M parameters
- With 22,234 L-R pairs: ~1.47B parameters

### Reproducibility

- Random seed: 42
- Deterministic CUDA operations enabled
- LORO cross-validation for unbiased evaluation

---

## References

1. Heart Cell Atlas v2 - Spatial Transcriptomics
2. Graph Attention Networks (Veličković et al., 2018)
3. OmniPath - Integrated L-R Database (Türei et al., 2021)
4. CellChat - Cell-Cell Communication Analysis (Jin et al., 2021)
5. CellPhoneDB - Ligand-Receptor Database (Efremova et al., 2020)

---
