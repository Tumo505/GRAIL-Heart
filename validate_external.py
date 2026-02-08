"""
External Validation & Generalisability Assessment for GRAIL-Heart

This script validates GRAIL-Heart's inverse modelling predictions on
external cardiac spatial transcriptomics datasets to assess
generalisability beyond the Heart Cell Atlas v2 training data.

All datasets must have true Visium spatial coordinates — the GNN
architecture, positional encoding, and causal L-R scores depend on
physical proximity.  Datasets without spatial coordinates are rejected.

Validation strategies:
1. L-R Consistency — Do top causal L-R pairs replicate across datasets?
2. Pathway Enrichment — Does TGF-β, YAP/TAZ, BMP enrichment replicate?
3. Disease vs Control — Kawasaki LCWE vs PBS + Kuppe MI vs healthy
4. MI Zone Comparison — Ischemic vs fibrotic vs remote zone L-R profiles
5. Cross-species — Mouse Visium (Kawasaki) vs human Visium (HCA, Kuppe)

Datasets (all Visium spatial):
- Kuppe et al. (Nature 2022): Human myocardial infarction (5 samples)
- Li et al. (2022): Mouse Kawasaki disease model (2 samples)
- Heart Cell Atlas v2: Human healthy apex (1 internal control)

Usage:
    python validate_external.py --config configs/validation.yaml
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from scipy.stats import fisher_exact, spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from grail_heart.models import GRAILHeart
from grail_heart.data.cellchat_database import (
    get_omnipath_lr_database,
    annotate_cardiac_pathways,
)
from grail_heart.data.expanded_lr_database import (
    filter_to_expressed_genes,
    compute_lr_scores,
    get_lr_genes,
)


# ─── MSigDB Hallmark Mechanosensitive Pathway Definitions ───────────
MECHANO_PATHWAYS = {
    'YAP_TAZ': [
        'WWTR1', 'YAP1', 'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4',
        'LATS1', 'LATS2', 'MST1', 'MST2', 'SAV1', 'MOB1A', 'MOB1B',
        'AMOT', 'AMOTL1', 'AMOTL2', 'NF2', 'PTPN14', 'RASSF1',
        'CCN1', 'CCN2', 'ANKRD1', 'CTGF', 'AXL', 'BIRC5',
    ],
    'BMP': [
        'BMP2', 'BMP4', 'BMP5', 'BMP6', 'BMP7', 'BMP10',
        'BMPR1A', 'BMPR1B', 'BMPR2', 'ACVR1', 'ACVRL1',
        'SMAD1', 'SMAD5', 'SMAD9', 'SMAD4', 'ID1', 'ID2', 'ID3',
        'NOG', 'GREM1', 'CHRD', 'FST', 'BAMBI',
    ],
    'TGF_BETA': [
        'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2', 'TGFBR3',
        'SMAD2', 'SMAD3', 'SMAD4', 'SMAD6', 'SMAD7',
        'SERPINE1', 'THBS1', 'FN1', 'COL1A1', 'COL3A1',
        'ACTA2', 'LTBP1', 'LTBP2', 'DCN', 'BGN',
    ],
    'NOTCH': [
        'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
        'DLL1', 'DLL3', 'DLL4', 'JAG1', 'JAG2',
        'HES1', 'HEY1', 'HEY2', 'HEYL', 'RBPJ', 'MAML1',
    ],
    'INTEGRIN_FAK': [
        'ITGA1', 'ITGA2', 'ITGA3', 'ITGA4', 'ITGA5', 'ITGA6',
        'ITGAV', 'ITGB1', 'ITGB3', 'ITGB5',
        'PTK2', 'PXN', 'VCL', 'TLN1', 'TLN2', 'ILK',
    ],
    'PIEZO': [
        'PIEZO1', 'PIEZO2', 'TRPV4', 'TRPC1', 'TRPC6',
        'KCNK2', 'KCNK4', 'PKD1', 'PKD2',
    ],
    'WNT': [
        'WNT1', 'WNT2', 'WNT3A', 'WNT5A', 'WNT5B', 'WNT11',
        'FZD1', 'FZD2', 'FZD7', 'LRP5', 'LRP6',
        'CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'TCF7', 'LEF1',
    ],
    'FGF': [
        'FGF1', 'FGF2', 'FGF7', 'FGF9', 'FGF10', 'FGF18',
        'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4',
        'FRS2', 'GRB2', 'SOS1', 'GAB1', 'SPRY1', 'SPRY2',
    ],
}


class ExternalValidator:
    """
    Validate GRAIL-Heart on external cardiac datasets.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ['figures', 'tables', 'reports']:
            (self.output_dir / sub).mkdir(exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.lr_database = None
        self.has_inverse = False

        # Results storage
        self.dataset_results: Dict[str, Dict] = {}
        self.reference_lr: Optional[pd.DataFrame] = None  # HCA v2 reference

        # Visium barcode → array coordinate lookup (lazy-loaded)
        self._visium_bc_lookup: Optional[Dict[str, Tuple[int, int]]] = None

    # ────────────────────────────────────────────────
    # Model loading (reuses EnhancedInference logic)
    # ────────────────────────────────────────────────
    def load_model(self) -> None:
        """Load trained GRAIL-Heart model from checkpoint."""
        ckpt_path = self.config['model']['checkpoint']
        print(f"Loading model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        model_cfg = config.get('model', {})

        # Infer architecture from state_dict
        n_genes = state_dict['gene_encoder.encoder.0.weight'].shape[1]
        hidden_dim = state_dict.get('gat.jk_proj.weight', torch.zeros(256, 1)).shape[0]
        gat_layers = set(
            int(k.split('.')[2]) for k in state_dict
            if k.startswith('gat.layers.') and k.split('.')[2].isdigit()
        )
        n_gat_layers = len(gat_layers) if gat_layers else 4
        att_key = 'gat.layers.0.gat.att_src'
        if att_key in state_dict:
            n_edge_types, n_heads, _ = state_dict[att_key].shape
        else:
            n_edge_types, n_heads = 2, 4
        ct_key = 'multimodal_encoder.cell_type_encoder.embedding.weight'
        n_cell_types = state_dict[ct_key].shape[0] if ct_key in state_dict else 10
        enc0 = state_dict.get('gene_encoder.encoder.0.weight')
        enc1 = state_dict.get('gene_encoder.encoder.4.weight')
        encoder_dims = [enc0.shape[0], enc1.shape[0]] if enc0 is not None and enc1 is not None else [512, 256]

        lr_proj_keys = [k for k in state_dict if 'lr_projections' in k and k.endswith('.weight')]
        n_lr_pairs = max(int(k.split('.')[-2]) for k in lr_proj_keys) + 1 if lr_proj_keys else None

        self.has_inverse = any('inverse_module' in k for k in state_dict)
        n_pathways, n_mechano_pathways = 20, 8
        if self.has_inverse:
            pw_key = 'inverse_module.lr_target_decoder.pathway_gene_matrix'
            if pw_key in state_dict:
                n_pathways = state_dict[pw_key].shape[0]
            mech_key = 'inverse_module.mechano_module.pathway_gene_mask'
            if mech_key in state_dict:
                n_mechano_pathways = state_dict[mech_key].shape[0]

        fate_key = 'inverse_module.cell_fate_head.fate_classifier.4.weight'
        n_fates = state_dict[fate_key].shape[0] if fate_key in state_dict else n_cell_types

        self.model = GRAILHeart(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            n_gat_layers=n_gat_layers,
            n_heads=n_heads,
            n_cell_types=n_cell_types,
            n_edge_types=n_edge_types,
            encoder_dims=encoder_dims,
            dropout=model_cfg.get('dropout', 0.2),
            use_spatial=model_cfg.get('use_spatial', True),
            use_variational=model_cfg.get('use_variational', False),
            tasks=model_cfg.get('tasks', ['lr', 'reconstruction', 'cell_type']),
            n_lr_pairs=n_lr_pairs,
            use_inverse_modelling=self.has_inverse,
            n_fates=n_fates,
            n_pathways=n_pathways,
            n_mechano_pathways=n_mechano_pathways,
            pathway_gene_mask=None,
            mechano_gene_mask=None,
            mechano_pathway_names=None,
            decoder_type=model_cfg.get('decoder_type', 'residual'),
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"  Inverse modelling: {'ENABLED' if self.has_inverse else 'DISABLED'}")
        print(f"  Expected n_genes: {n_genes}")

    def load_lr_database(self) -> None:
        """Load OmniPath L-R database."""
        cache = Path('data/lr_database_cache.csv')
        self.lr_database = get_omnipath_lr_database(cache_path=cache)
        self.lr_database = annotate_cardiac_pathways(self.lr_database)
        print(f"Loaded {len(self.lr_database)} L-R pairs from OmniPath")

    # ────────────────────────────────────────────────
    # Data loading & preprocessing
    # ────────────────────────────────────────────────
    def _load_dataset(self, ds_key: str) -> Optional[ad.AnnData]:
        """Load and preprocess a single external dataset."""
        ds_cfg = self.config['datasets'][ds_key]
        path = Path(ds_cfg['path'])

        if not path.exists():
            print(f"  [!] File not found: {path}")
            return None

        print(f"  Loading {ds_cfg['name']} from {path}")

        # Load based on format
        fmt = ds_cfg.get('format', 'h5ad')
        if fmt == 'h5ad':
            adata = sc.read_h5ad(path)
        elif fmt == 'h5_10x':
            adata = sc.read_10x_h5(path)
        elif fmt == 'directory':
            adata = sc.read_10x_mtx(path)
        else:
            print(f"  [!] Unknown format: {fmt}")
            return None

        print(f"  Raw: {adata.n_obs} cells x {adata.n_vars} genes")

        # Make var names unique
        adata.var_names_make_unique()

        # Convert Ensembl IDs to gene symbols if needed
        if adata.var_names[0].startswith('ENSG') or adata.var_names[0].startswith('ENSMUSG'):
            if 'gene_symbols' in adata.var.columns:
                adata.var_names = adata.var['gene_symbols'].values
                adata.var_names_make_unique()
            elif 'SYMBOL' in adata.var.columns:
                adata.var_names = adata.var['SYMBOL'].values
                adata.var_names_make_unique()

        # Mouse → Human gene symbol conversion (uppercase)
        if ds_cfg.get('species') == 'mouse':
            adata.var['original_name'] = adata.var_names.copy()
            adata.var_names = [g.upper() if isinstance(g, str) else g for g in adata.var_names]
            adata.var_names_make_unique()

        # Preprocessing
        pp = self.config.get('preprocessing', {})
        sc.pp.filter_cells(adata, min_genes=pp.get('min_genes', 200))
        sc.pp.filter_genes(adata, min_cells=pp.get('min_cells', 3))

        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Select HVGs
        n_hvg = pp.get('n_top_genes', 2000)
        if adata.n_vars > n_hvg:
            try:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=n_hvg,
                    flavor='seurat_v3', layer='counts',
                )
            except Exception:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat')
            adata = adata[:, adata.var.highly_variable].copy()

        # Save log-norm before z-scoring
        if hasattr(adata.X, 'toarray'):
            adata.layers['log_norm'] = adata.X.toarray().copy()
        else:
            adata.layers['log_norm'] = np.array(adata.X).copy()

        # Scale
        sc.pp.scale(adata, max_value=10)
        print(f"  After preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")

        return adata

    def _load_visium_barcode_lookup(self) -> Dict[str, Tuple[int, int]]:
        """Load Visium V1 barcode → (array_row, array_col) lookup table.

        Uses the reference tissue_positions_list.csv downloaded alongside the
        10x Genomics V1_Human_Heart dataset.  All Visium V1 slides share the
        same barcode–position mapping (4,992 spots on a hexagonal grid), so a
        single reference file is sufficient.

        Returns:
            dict mapping barcode strings to (row, col) tuples.
        """
        if self._visium_bc_lookup is not None:
            return self._visium_bc_lookup

        ref_path = Path('data/V1_Human_Heart/spatial/tissue_positions_list.csv')
        if not ref_path.exists():
            # Attempt to download via scanpy
            print("  Downloading Visium V1 reference coordinates...")
            try:
                sc.datasets.visium_sge(sample_id='V1_Human_Heart')
            except Exception as e:
                print(f"  [!] Failed to download reference: {e}")
                self._visium_bc_lookup = {}
                return self._visium_bc_lookup
            if not ref_path.exists():
                self._visium_bc_lookup = {}
                return self._visium_bc_lookup

        tpos = pd.read_csv(
            ref_path, header=None,
            names=['barcode', 'in_tissue', 'array_row', 'array_col',
                   'pxl_row', 'pxl_col'],
        )
        self._visium_bc_lookup = dict(
            zip(tpos['barcode'], zip(tpos['array_row'], tpos['array_col']))
        )
        print(f"  Loaded Visium V1 barcode lookup ({len(self._visium_bc_lookup)} barcodes)")
        return self._visium_bc_lookup

    def _reconstruct_visium_coords(self, adata: ad.AnnData) -> Optional[np.ndarray]:
        """Reconstruct Visium spatial coordinates from barcode identities.

        10x Visium V1 slides have a fixed hexagonal grid: each barcode maps
        to a deterministic (array_row, array_col) position.  We convert the
        array coordinates to micrometre-scale spatial coordinates using the
        known Visium geometry (100 µm centre-to-centre, hexagonal packing).

        These are the TRUE physical positions — NOT pseudo-coordinates.
        """
        lookup = self._load_visium_barcode_lookup()
        if not lookup:
            return None

        barcodes = adata.obs_names.tolist()
        matched = [bc for bc in barcodes if bc in lookup]
        match_rate = len(matched) / len(barcodes) if barcodes else 0

        if match_rate < 0.5:
            print(f"  [!] Only {match_rate:.0%} barcodes match Visium V1 -- not a Visium V1 slide")
            return None

        # Assign array coordinates and convert to spatial (µm)
        # Visium geometry: 100 µm centre-to-centre, hexagonal packing
        # row spacing = 100 µm, col spacing = 100 µm
        # Odd rows are offset by half a column width
        SPOT_SPACING = 100.0  # µm
        coords = np.zeros((len(barcodes), 2), dtype=np.float64)
        for i, bc in enumerate(barcodes):
            if bc in lookup:
                row, col = lookup[bc]
                # Convert hex grid to Cartesian (µm)
                x = col * SPOT_SPACING * 0.5  # columns are spaced 2-apart in array
                y = row * SPOT_SPACING * (np.sqrt(3) / 2)  # hex row spacing
                coords[i] = [x, y]
            else:
                # Place unmatched barcodes at origin (will be few if any)
                coords[i] = [0, 0]

        # Also store in adata for downstream use
        adata.obsm['spatial'] = coords

        print(f"  Reconstructed Visium coordinates: {len(matched)}/{len(barcodes)} "
              f"barcodes mapped ({match_rate:.1%})")
        print(f"  Spatial extent: {np.ptp(coords[:,0]):.0f} x {np.ptp(coords[:,1]):.0f} um")
        return coords

    def _get_spatial_coords(self, adata: ad.AnnData) -> Optional[np.ndarray]:
        """Get real spatial coordinates. Returns None if not available.

        GRAIL-Heart is a spatial transcriptomics framework — the GNN graph,
        positional encoding, and causal L-R scores all require genuine
        spatial proximity.  Using pseudo-coordinates (PCA/UMAP) would make
        the kNN graph reflect expression similarity rather than physical
        adjacency, invalidating the spatial assumptions of the model.

        For 10x Visium data where spatial coordinates are not embedded in
        the .h5 file (e.g. CZ CELLxGENE downloads), we reconstruct true
        array positions from the deterministic barcode–position mapping.
        """
        # Try Visium / spatial coords already in the object
        for key in ['spatial', 'X_spatial']:
            if key in adata.obsm:
                coords = np.array(adata.obsm[key])
                if coords.shape[1] >= 2:
                    print(f"  Using spatial coordinates from obsm['{key}']")
                    return coords[:, :2]

        # Try to reconstruct from Visium V1 barcode identities
        coords = self._reconstruct_visium_coords(adata)
        if coords is not None:
            return coords

        # No spatial coordinates → skip this dataset
        print("  [!] No spatial coordinates found -- skipping dataset")
        print("    (GRAIL-Heart requires true Visium spatial coords;")
        print("     pseudo-coordinates from PCA/UMAP would violate")
        print("     the spatial proximity assumptions of the GNN)")
        return None

    def _build_knn_graph(self, coords: np.ndarray, k: int = 15) -> np.ndarray:
        """Build k-NN graph from coordinates."""
        adj = kneighbors_graph(coords, n_neighbors=min(k, coords.shape[0] - 1), mode='connectivity')
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        coo = adj.tocoo()
        return np.vstack([coo.row, coo.col])

    # ────────────────────────────────────────────────
    # Model inference
    # ────────────────────────────────────────────────
    def _run_inference(
        self,
        expression: np.ndarray,
        coords: np.ndarray,
        edge_index: np.ndarray,
        cell_types: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run forward + inverse model inference."""
        if self.model is None:
            return {}

        # Pad or truncate genes to match model expected input
        expected_genes = self.model.n_genes
        n_cells, n_genes = expression.shape

        if n_genes < expected_genes:
            padded = np.zeros((n_cells, expected_genes), dtype=np.float32)
            padded[:, :n_genes] = expression
            expression = padded
        elif n_genes > expected_genes:
            expression = expression[:, :expected_genes]

        x = torch.tensor(expression, dtype=torch.float32)
        edge_idx = torch.tensor(edge_index, dtype=torch.long)
        pos = torch.tensor(coords, dtype=torch.float32)

        # Normalize positions to [0, 1]
        pos = (pos - pos.min(dim=0)[0]) / (pos.max(dim=0)[0] - pos.min(dim=0)[0] + 1e-8)

        if cell_types is None:
            cell_type_tensor = torch.zeros(n_cells, dtype=torch.long)
        else:
            cell_type_tensor = torch.tensor(cell_types, dtype=torch.long)

        data = Data(
            x=x, edge_index=edge_idx, pos=pos,
            y=cell_type_tensor, num_nodes=n_cells,
        )
        data = data.to(self.device)

        outputs = {}
        with torch.no_grad():
            try:
                run_inverse = self.has_inverse and self.config['model'].get('use_inverse', True)
                model_out = self.model(data, run_inverse=run_inverse)

                if 'lr_scores' in model_out:
                    outputs['forward_lr_scores'] = model_out['lr_scores'].cpu()
                if 'node_embeddings' in model_out:
                    outputs['node_embeddings'] = model_out['node_embeddings'].cpu()

                if run_inverse:
                    for key in ['causal_lr_scores', 'fate_logits', 'differentiation_score',
                                'pathway_activation', 'mechano_pathway_activation']:
                        if key in model_out:
                            outputs[key] = model_out[key].cpu()

                    if hasattr(self.model, 'infer_causal_signals') and self.model.inverse_module is not None:
                        try:
                            inv = self.model.infer_causal_signals(data)
                            if 'inferred_lr_importance' in inv:
                                outputs['inferred_lr_importance'] = inv['inferred_lr_importance'].cpu()
                        except Exception:
                            pass

            except Exception as e:
                print(f"  [!] Inference error: {e}")

        return outputs

    def _aggregate_causal_scores(
        self,
        causal_per_edge: np.ndarray,
        expression_unscaled: np.ndarray,
        gene_names: List[str],
        edge_index: np.ndarray,
        lr_df: pd.DataFrame,
        percentile: float = 90.0,
        threshold: float = 0.5,
    ) -> List[float]:
        """Expression-gated P90 aggregation of per-edge causal scores."""
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        src, dst = edge_index[0], edge_index[1]
        scores = []

        for _, row in lr_df.iterrows():
            lig_idx = gene_to_idx.get(row['ligand'])
            rec_idx = gene_to_idx.get(row['receptor'])
            if lig_idx is None or rec_idx is None:
                scores.append(0.0)
                continue

            lig_expr = expression_unscaled[src, lig_idx]
            rec_expr = expression_unscaled[dst, rec_idx]
            active = (lig_expr > threshold) & (rec_expr > threshold)

            if active.sum() < 3:
                scores.append(0.0)
                continue

            pct = float(np.percentile(causal_per_edge[active], percentile))
            frac = active.sum() / len(causal_per_edge)
            scores.append(pct * np.sqrt(frac))

        return scores

    # ────────────────────────────────────────────────
    # Process a single dataset
    # ────────────────────────────────────────────────
    def process_dataset(self, ds_key: str) -> Optional[Dict]:
        """Full inference pipeline on one external dataset."""
        ds_cfg = self.config['datasets'][ds_key]
        print(f"\n{'='*60}")
        print(f"Validating: {ds_cfg['name']} ({ds_key})")
        print(f"  Species: {ds_cfg.get('species', '?')}, Condition: {ds_cfg.get('condition', '?')}")
        print('='*60)

        adata = self._load_dataset(ds_key)
        if adata is None:
            return None

        gene_names = adata.var_names.tolist()
        coords = self._get_spatial_coords(adata)
        if coords is None:
            return None

        k = self.config.get('preprocessing', {}).get('k_neighbors', 15)
        edge_index = self._build_knn_graph(coords, k=k)
        print(f"  Graph: {edge_index.shape[1]} edges (k={k})")

        # Expression matrices
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = np.array(adata.X)

        expression_unscaled = np.array(adata.layers.get('log_norm', expression))

        # Filter L-R database
        expressed_lr = filter_to_expressed_genes(self.lr_database, gene_names)
        print(f"  Matched L-R pairs: {len(expressed_lr)}")
        if len(expressed_lr) == 0:
            print("  [!] No L-R pairs found -- skipping")
            return None

        # Expression-based L-R scores
        expr_scores = compute_lr_scores(
            expression=expression, gene_names=gene_names,
            edge_index=edge_index, lr_df=expressed_lr, method='product',
        )

        # Model inference
        model_out = self._run_inference(expression, coords, edge_index)

        # Add causal scores
        lr_scores = expr_scores.copy()
        if model_out and 'causal_lr_scores' in model_out:
            causal = model_out['causal_lr_scores'].numpy()
            causal_per_pair = self._aggregate_causal_scores(
                causal, expression_unscaled, gene_names, edge_index, lr_scores,
            )
            lr_scores['causal_score'] = causal_per_pair
            print(f"  Causal scores: {sum(1 for s in causal_per_pair if s > 0)} non-zero / {len(causal_per_pair)}")

        # Fate scores
        fate_stats = {}
        if model_out and 'fate_logits' in model_out:
            fate_probs = torch.softmax(model_out['fate_logits'], dim=-1)
            max_fate = fate_probs.max(dim=-1)[0].numpy()
            fate_stats = {
                'mean': float(np.mean(max_fate)),
                'std': float(np.std(max_fate)),
                'min': float(np.min(max_fate)),
                'max': float(np.max(max_fate)),
                'range': float(np.max(max_fate) - np.min(max_fate)),
            }
            print(f"  Fate scores: {fate_stats['mean']:.4f} +/- {fate_stats['std']:.4f} "
                  f"(range {fate_stats['min']:.4f}-{fate_stats['max']:.4f})")

        # Pathway enrichment
        enrichment = self._pathway_enrichment(lr_scores, gene_names)

        # Store result
        result = {
            'dataset': ds_key,
            'name': ds_cfg['name'],
            'species': ds_cfg.get('species'),
            'condition': ds_cfg.get('condition'),
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'n_edges': edge_index.shape[1],
            'n_lr_matched': len(expressed_lr),
            'n_lr_detected': len(lr_scores),
            'lr_scores': lr_scores,
            'fate_stats': fate_stats,
            'enrichment': enrichment,
            'model_outputs': model_out,
            'gene_names': gene_names,
        }
        self.dataset_results[ds_key] = result

        # Save per-dataset results
        lr_scores.to_csv(self.output_dir / 'tables' / f'{ds_key}_lr_scores.csv', index=False)

        return result

    # ────────────────────────────────────────────────
    # Pathway enrichment (Fisher's exact test)
    # ────────────────────────────────────────────────
    def _pathway_enrichment(
        self,
        lr_scores: pd.DataFrame,
        gene_names: List[str],
        top_k: int = 50,
    ) -> Dict[str, Dict]:
        """Fisher's exact test for mechanosensitive pathway enrichment."""
        results = {}

        # Get top-k causal L-R genes
        score_col = 'causal_score' if 'causal_score' in lr_scores.columns else 'mean_score'
        top_lr = lr_scores.nlargest(top_k, score_col)
        top_genes = set(top_lr['ligand'].tolist() + top_lr['receptor'].tolist())
        background = set(gene_names)

        for pathway_name, pathway_genes in MECHANO_PATHWAYS.items():
            pathway_set = set(pathway_genes) & background
            if len(pathway_set) == 0:
                results[pathway_name] = {'p_value': 1.0, 'odds_ratio': 0, 'overlap': 0, 'pathway_size': 0}
                continue

            a = len(top_genes & pathway_set)
            b = len(top_genes - pathway_set)
            c = len(pathway_set - top_genes)
            d = len(background - top_genes - pathway_set)

            try:
                odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            except Exception:
                odds, p = 0, 1.0

            results[pathway_name] = {
                'p_value': float(p),
                'odds_ratio': float(odds) if not np.isinf(odds) else 999.0,
                'overlap': a,
                'pathway_size': len(pathway_set),
                'top_genes_in_pathway': sorted(top_genes & pathway_set),
            }

        return results

    # ────────────────────────────────────────────────
    # Validation analyses
    # ────────────────────────────────────────────────
    def analyse_lr_consistency(self) -> pd.DataFrame:
        """Compare top L-R rankings across all datasets."""
        print("\n" + "="*60)
        print("L-R Consistency Analysis")
        print("="*60)

        top_k = self.config.get('analyses', {}).get('lr_consistency', {}).get('top_k', 50)
        score_col = 'causal_score'

        all_rankings = {}
        for ds_key, result in self.dataset_results.items():
            lr = result['lr_scores']
            col = score_col if score_col in lr.columns else 'mean_score'
            top = lr.nlargest(top_k, col)
            top_pairs = set(zip(top['ligand'], top['receptor']))
            all_rankings[ds_key] = top_pairs

        # Compute pairwise Jaccard similarity
        keys = list(all_rankings.keys())
        n = len(keys)
        jaccard_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                a, b = all_rankings[keys[i]], all_rankings[keys[j]]
                inter = len(a & b)
                union = len(a | b)
                jaccard_matrix[i, j] = inter / union if union > 0 else 0

        jaccard_df = pd.DataFrame(jaccard_matrix, index=keys, columns=keys)
        jaccard_df.to_csv(self.output_dir / 'tables' / 'lr_consistency_jaccard.csv')

        print(f"\nJaccard similarity (top-{top_k} L-R pairs):")
        print(jaccard_df.round(3).to_string())

        # Spearman rank correlation on common L-R pairs
        print("\nSpearman rank correlations (common pairs):")
        spearman_results = []
        for i in range(n):
            for j in range(i+1, n):
                lr_i = self.dataset_results[keys[i]]['lr_scores']
                lr_j = self.dataset_results[keys[j]]['lr_scores']
                col_i = score_col if score_col in lr_i.columns else 'mean_score'
                col_j = score_col if score_col in lr_j.columns else 'mean_score'

                # Merge on L-R pair
                merged = lr_i[['ligand', 'receptor', col_i]].merge(
                    lr_j[['ligand', 'receptor', col_j]],
                    on=['ligand', 'receptor'], how='inner', suffixes=('_a', '_b')
                )
                if len(merged) >= 10:
                    rho, p = spearmanr(merged.iloc[:, 2], merged.iloc[:, 3])
                    print(f"  {keys[i]} vs {keys[j]}: rho={rho:.3f} (p={p:.2e}, n={len(merged)})")
                    spearman_results.append({
                        'dataset_a': keys[i], 'dataset_b': keys[j],
                        'rho': rho, 'p_value': p, 'n_pairs': len(merged),
                    })

        if spearman_results:
            pd.DataFrame(spearman_results).to_csv(
                self.output_dir / 'tables' / 'lr_consistency_spearman.csv', index=False
            )

        return jaccard_df

    def analyse_disease_vs_control(self) -> Optional[Dict]:
        """Compare disease vs control L-R profiles (Kawasaki LCWE vs PBS)."""
        print("\n" + "="*60)
        print("Disease vs Control Comparison")
        print("="*60)

        pairs = self.config.get('analyses', {}).get('disease_comparison', {}).get('pairs', [])
        results = {}

        for disease_key, control_key in pairs:
            if disease_key not in self.dataset_results or control_key not in self.dataset_results:
                print(f"  [!] Skipping {disease_key} vs {control_key} (not processed)")
                continue

            disease = self.dataset_results[disease_key]['lr_scores']
            control = self.dataset_results[control_key]['lr_scores']
            score_col = 'causal_score' if 'causal_score' in disease.columns else 'mean_score'

            # Merge
            merged = disease[['ligand', 'receptor', 'pathway', score_col]].merge(
                control[['ligand', 'receptor', score_col]],
                on=['ligand', 'receptor'], how='inner', suffixes=('_disease', '_control')
            )

            if len(merged) == 0:
                print(f"  [!] No common L-R pairs between {disease_key} and {control_key}")
                continue

            # Compute differential scores
            col_d = f'{score_col}_disease'
            col_c = f'{score_col}_control'
            merged['log2fc'] = np.log2((merged[col_d] + 1e-6) / (merged[col_c] + 1e-6))
            merged = merged.sort_values('log2fc', ascending=False)

            # Statistical test
            stat, p = mannwhitneyu(merged[col_d], merged[col_c], alternative='two-sided')

            print(f"\n  {disease_key} vs {control_key}:")
            print(f"    Common L-R pairs: {len(merged)}")
            print(f"    Mann-Whitney U p-value: {p:.2e}")
            print(f"    Top upregulated in disease:")
            for _, row in merged.head(5).iterrows():
                print(f"      {row['ligand']}->{row['receptor']}: log2FC={row['log2fc']:.2f} ({row['pathway']})")
            print(f"    Top downregulated in disease:")
            for _, row in merged.tail(5).iterrows():
                print(f"      {row['ligand']}->{row['receptor']}: log2FC={row['log2fc']:.2f} ({row['pathway']})")

            merged.to_csv(
                self.output_dir / 'tables' / f'disease_vs_control_{disease_key}_{control_key}.csv',
                index=False,
            )
            results[f'{disease_key}_vs_{control_key}'] = {                'n_pairs': len(merged), 'mwu_p': float(p),
                'top_up': merged.head(5)[['ligand', 'receptor', 'log2fc']].to_dict('records'),
                'top_down': merged.tail(5)[['ligand', 'receptor', 'log2fc']].to_dict('records'),
            }

        return results

    def analyse_mi_zone_comparison(self) -> Optional[Dict]:
        """Compare inverse modelling outputs across MI zones (Kuppe et al.)."""
        mi_cfg = self.config.get('analyses', {}).get('mi_zone_comparison', {})
        if not mi_cfg.get('enabled', False):
            return None

        print("\n" + "="*60)
        print("MI Zone Comparison (Kuppe et al., Nature 2022)")
        print("="*60)

        zones = mi_cfg.get('zones', {})
        control_keys = mi_cfg.get('controls', [])

        # Collect zone data
        zone_data = {}
        for zone_name, ds_key in zones.items():
            if ds_key in self.dataset_results:
                zone_data[zone_name] = self.dataset_results[ds_key]

        control_data = {k: self.dataset_results[k] for k in control_keys if k in self.dataset_results}

        if not zone_data:
            print("  No MI zone data available")
            return None

        results = {'zone_profiles': {}, 'pathway_by_zone': {}, 'fate_by_zone': {}}

        # ── 1. Per-zone pathway enrichment comparison ──
        print("\n  Pathway enrichment by MI zone:")
        sig_threshold = 0.05
        for zone_name, data in zone_data.items():
            enrich = data.get('enrichment', {})
            sig_pathways = [
                pw for pw, s in enrich.items()
                if s['p_value'] * len(MECHANO_PATHWAYS) < sig_threshold
            ]
            p_vals = {pw: s['p_value'] for pw, s in enrich.items()}
            results['pathway_by_zone'][zone_name] = p_vals
            print(f"    {zone_name}: {', '.join(sig_pathways) if sig_pathways else 'None significant'}")
            for pw in ['TGF_BETA', 'YAP_TAZ', 'BMP', 'INTEGRIN_FAK']:
                if pw in p_vals:
                    p = p_vals[pw] * len(MECHANO_PATHWAYS)
                    marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    print(f"      {pw}: p={p:.2e} {marker}")

        # ── 2. Per-zone fate score comparison ──
        print("\n  Cell fate scores by MI zone:")
        for zone_name, data in zone_data.items():
            fate = data.get('fate_stats', {})
            if fate:
                results['fate_by_zone'][zone_name] = fate
                print(f"    {zone_name}: {fate['mean']:.4f} +/- {fate['std']:.4f} "
                      f"(range {fate['min']:.4f}-{fate['max']:.4f})")

        # Control comparison
        for ctrl_key, ctrl_data in control_data.items():
            fate = ctrl_data.get('fate_stats', {})
            if fate:
                results['fate_by_zone'][f'control ({ctrl_key})'] = fate
                print(f"    control ({ctrl_key}): {fate['mean']:.4f} +/- {fate['std']:.4f}")

        # ── 3. TGF-β score gradient across zones ──
        print("\n  TGF-beta L-R score gradient (fibrotic > ischemic > remote ~ control):")
        tgf_scores = {}
        for zone_name, data in zone_data.items():
            lr = data['lr_scores']
            if 'pathway' in lr.columns and 'causal_score' in lr.columns:
                tgf_lr = lr[lr['pathway'].str.contains('TGF', case=False, na=False)]
                if len(tgf_lr) > 0:
                    tgf_scores[zone_name] = float(tgf_lr['causal_score'].mean())

        for ctrl_key, ctrl_data in control_data.items():
            lr = ctrl_data['lr_scores']
            if 'pathway' in lr.columns and 'causal_score' in lr.columns:
                tgf_lr = lr[lr['pathway'].str.contains('TGF', case=False, na=False)]
                if len(tgf_lr) > 0:
                    tgf_scores[f'control ({ctrl_key})'] = float(tgf_lr['causal_score'].mean())

        for name, score in sorted(tgf_scores.items(), key=lambda x: -x[1]):
            print(f"    {name}: mean TGF-beta causal score = {score:.4f}")
        results['tgf_gradient'] = tgf_scores

        # ── 4. Zone-specific top L-R pairs ──
        print("\n  Zone-specific top L-R (unique to each zone vs control):")
        control_top_pairs = set()
        for ctrl_key, ctrl_data in control_data.items():
            lr = ctrl_data['lr_scores']
            col = 'causal_score' if 'causal_score' in lr.columns else 'mean_score'
            top30 = lr.nlargest(30, col)
            control_top_pairs.update(zip(top30['ligand'], top30['receptor']))

        for zone_name, data in zone_data.items():
            lr = data['lr_scores']
            col = 'causal_score' if 'causal_score' in lr.columns else 'mean_score'
            top30 = lr.nlargest(30, col)
            zone_pairs = set(zip(top30['ligand'], top30['receptor']))
            unique = zone_pairs - control_top_pairs
            print(f"    {zone_name}: {len(unique)} unique pairs (of top 30)")
            if unique:
                for lig, rec in list(unique)[:3]:
                    row = lr[(lr['ligand']==lig) & (lr['receptor']==rec)]
                    pw = row['pathway'].values[0] if len(row) > 0 else '?'
                    print(f"      {lig} -> {rec} ({pw})")
            results['zone_profiles'][zone_name] = {
                'n_unique_top30': len(unique),
                'examples': [(l, r) for l, r in list(unique)[:5]],
            }

        # Save
        zone_summary = pd.DataFrame([
            {'zone': z, 'fate_mean': results['fate_by_zone'].get(z, {}).get('mean', np.nan),
             'fate_std': results['fate_by_zone'].get(z, {}).get('std', np.nan),
             'tgf_score': results['tgf_gradient'].get(z, np.nan),
             'n_unique_top30': results['zone_profiles'].get(z, {}).get('n_unique_top30', 0)}
            for z in list(zone_data.keys()) + [f'control ({k})' for k in control_data]
        ])
        zone_summary.to_csv(self.output_dir / 'tables' / 'mi_zone_comparison.csv', index=False)

        return results

    def analyse_pathway_enrichment(self) -> pd.DataFrame:
        """Summarise pathway enrichment across all datasets."""
        print("\n" + "="*60)
        print("Cross-Dataset Pathway Enrichment")
        print("="*60)

        rows = []
        for ds_key, result in self.dataset_results.items():
            enrich = result.get('enrichment', {})
            for pathway, stats in enrich.items():
                rows.append({
                    'dataset': ds_key,
                    'condition': self.config['datasets'][ds_key].get('condition', '?'),
                    'species': self.config['datasets'][ds_key].get('species', '?'),
                    'pathway': pathway,
                    'p_value': stats['p_value'],
                    'odds_ratio': stats['odds_ratio'],
                    'overlap': stats['overlap'],
                    'pathway_size': stats['pathway_size'],
                })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        # Apply Bonferroni correction per dataset
        df['bonferroni_p'] = df['p_value'] * len(MECHANO_PATHWAYS)
        df['bonferroni_p'] = df['bonferroni_p'].clip(upper=1.0)
        df['significant'] = df['bonferroni_p'] < 0.05

        # Pivot for display
        pivot = df.pivot_table(
            index='pathway', columns='dataset', values='p_value', aggfunc='first',
        )
        print("\nPathway enrichment p-values:")
        print(pivot.applymap(lambda x: f"{x:.2e}" if x < 0.05 else f"{x:.2f}").to_string())

        df.to_csv(self.output_dir / 'tables' / 'pathway_enrichment_all.csv', index=False)
        pivot.to_csv(self.output_dir / 'tables' / 'pathway_enrichment_pivot.csv')

        return df

    # ────────────────────────────────────────────────
    # Figures
    # ────────────────────────────────────────────────
    def _plot_enrichment_heatmap(self, enrichment_df: pd.DataFrame) -> None:
        """Heatmap of pathway enrichment across datasets."""
        if len(enrichment_df) == 0:
            return

        pivot = enrichment_df.pivot_table(
            index='pathway', columns='dataset', values='p_value', aggfunc='first'
        )
        log_p = -np.log10(pivot.fillna(1.0))

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), 6))
        sns.heatmap(
            log_p, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, vmin=0, vmax=25,
        )
        ax.set_title('Mechanosensitive Pathway Enrichment (-log₁₀ p)', fontsize=13)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'pathway_enrichment_heatmap.png', dpi=300)
        plt.close()

    def _plot_lr_consistency_heatmap(self, jaccard_df: pd.DataFrame) -> None:
        """Heatmap of L-R consistency Jaccard indices."""
        if len(jaccard_df) < 2:
            return

        fig, ax = plt.subplots(figsize=(max(6, len(jaccard_df) * 1.2), max(5, len(jaccard_df))))
        sns.heatmap(
            jaccard_df, annot=True, fmt='.2f', cmap='Blues',
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
        )
        ax.set_title('L-R Pair Consistency (Jaccard Similarity)', fontsize=13)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'lr_consistency_heatmap.png', dpi=300)
        plt.close()

    def _plot_fate_comparison(self) -> None:
        """Bar chart of fate score distributions across datasets."""
        fate_data = []
        for ds_key, result in self.dataset_results.items():
            stats = result.get('fate_stats', {})
            if stats:
                fate_data.append({
                    'dataset': ds_key,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'condition': self.config['datasets'][ds_key].get('condition', '?'),
                })

        if not fate_data:
            return

        df = pd.DataFrame(fate_data)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
        colors = {'healthy': '#4CAF50', 'disease': '#F44336',
                  'control': '#9E9E9E', 'myocardial_infarction': '#D32F2F', '?': '#757575'}
        bar_colors = [colors.get(row['condition'], '#757575') for _, row in df.iterrows()]

        bars = ax.bar(df['dataset'], df['mean'], yerr=df['std'], capsize=5,
                      color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Cell Fate Score (max probability)', fontsize=11)
        ax.set_title('Cell Fate Scores Across Datasets', fontsize=13)
        ax.set_xticklabels(df['dataset'], rotation=45, ha='right')

        # Add legend
        from matplotlib.patches import Patch
        legend_items = [Patch(facecolor=c, label=l) for l, c in colors.items() if l in df['condition'].values]
        ax.legend(handles=legend_items, loc='upper right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'fate_comparison.png', dpi=300)
        plt.close()

    def _plot_disease_volcano(self) -> None:
        """Volcano plot for disease vs control comparisons."""
        pairs = self.config.get('analyses', {}).get('disease_comparison', {}).get('pairs', [])
        for disease_key, control_key in pairs:
            csv_path = self.output_dir / 'tables' / f'disease_vs_control_{disease_key}_{control_key}.csv'
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if 'log2fc' not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['red' if abs(fc) > 1 else 'gray' for fc in df['log2fc']]
            ax.scatter(df['log2fc'], range(len(df)), c=colors, alpha=0.6, s=10)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
            ax.axvline(1, color='red', linestyle=':', alpha=0.5)
            ax.axvline(-1, color='blue', linestyle=':', alpha=0.5)
            ax.set_xlabel('log₂ Fold Change (Disease / Control)')
            ax.set_ylabel('L-R Pair Rank')
            ax.set_title(f'{disease_key} vs {control_key}: Differential L-R Scores')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'figures' / f'volcano_{disease_key}_{control_key}.png', dpi=300)
            plt.close()

    def _plot_mi_zone_comparison(self) -> None:
        """Radar/bar chart of pathway enrichment across MI zones."""
        mi_cfg = self.config.get('analyses', {}).get('mi_zone_comparison', {})
        zones = mi_cfg.get('zones', {})
        control_keys = mi_cfg.get('controls', [])

        # Collect -log10(p) for each zone × pathway
        zone_keys = {z: k for z, k in zones.items() if k in self.dataset_results}
        if not zone_keys:
            return

        pathways = list(MECHANO_PATHWAYS.keys())
        data_rows = []

        for zone_name, ds_key in zone_keys.items():
            enrich = self.dataset_results[ds_key].get('enrichment', {})
            for pw in pathways:
                p = enrich.get(pw, {}).get('p_value', 1.0)
                data_rows.append({'zone': zone_name, 'pathway': pw, 'neg_log10_p': -np.log10(max(p, 1e-30))})

        for ctrl_key in control_keys:
            if ctrl_key not in self.dataset_results:
                continue
            enrich = self.dataset_results[ctrl_key].get('enrichment', {})
            for pw in pathways:
                p = enrich.get(pw, {}).get('p_value', 1.0)
                data_rows.append({'zone': f'ctrl', 'pathway': pw, 'neg_log10_p': -np.log10(max(p, 1e-30))})
            break  # Use first control only for clarity

        df = pd.DataFrame(data_rows)
        if len(df) == 0:
            return

        pivot = df.pivot_table(index='pathway', columns='zone', values='neg_log10_p', aggfunc='first')

        fig, ax = plt.subplots(figsize=(10, 6))
        zone_colors = {'ischemic': '#D32F2F', 'fibrotic': '#FF9800', 'remote': '#2196F3', 'ctrl': '#4CAF50'}
        pivot.plot(kind='bar', ax=ax, color=[zone_colors.get(c, '#757575') for c in pivot.columns],
                   edgecolor='black', linewidth=0.5)
        ax.axhline(-np.log10(0.05 / len(pathways)), color='red', linestyle='--', alpha=0.7,
                   label=f'Bonferroni threshold (p=0.05/{len(pathways)})')
        ax.set_ylabel('-log$_{10}$(p-value)', fontsize=11)
        ax.set_xlabel('')
        ax.set_title('Mechanosensitive Pathway Enrichment by MI Zone (Kuppe et al.)', fontsize=13)
        ax.legend(title='Zone', loc='upper right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'mi_zone_pathway_enrichment.png', dpi=300)
        plt.close()

        # TGF-β gradient bar chart
        tgf_csv = self.output_dir / 'tables' / 'mi_zone_comparison.csv'
        if tgf_csv.exists():
            zone_df = pd.read_csv(tgf_csv)
            zone_df = zone_df.dropna(subset=['tgf_score'])
            if len(zone_df) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors_list = []
                for z in zone_df['zone']:
                    if 'ischemic' in z:
                        colors_list.append('#D32F2F')
                    elif 'fibrotic' in z:
                        colors_list.append('#FF9800')
                    elif 'remote' in z:
                        colors_list.append('#2196F3')
                    else:
                        colors_list.append('#4CAF50')
                ax.bar(zone_df['zone'], zone_df['tgf_score'], color=colors_list,
                       edgecolor='black', linewidth=0.5)
                ax.set_ylabel('Mean TGF-$\\beta$ Causal Score', fontsize=11)
                ax.set_title('TGF-$\\beta$ L-R Score Gradient Across MI Zones', fontsize=13)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'figures' / 'tgf_beta_gradient.png', dpi=300)
                plt.close()

    # ────────────────────────────────────────────────
    # Report generation
    # ────────────────────────────────────────────────
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        lines = []
        lines.append("="*70)
        lines.append("GRAIL-Heart External Validation Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)
        lines.append("")

        # Summary table
        lines.append("Dataset Summary:")
        lines.append("-"*70)
        lines.append(f"{'Dataset':<20} {'Cells':>7} {'Genes':>7} {'L-R':>6} {'Species':<8} {'Condition':<12}")
        lines.append("-"*70)
        for ds_key, result in self.dataset_results.items():
            lines.append(
                f"{ds_key:<20} {result['n_cells']:>7,} {result['n_genes']:>7,} "
                f"{result['n_lr_detected']:>6} {result.get('species', '?'):<8} "
                f"{result.get('condition', '?'):<12}"
            )
        lines.append("")

        # Fate scores
        lines.append("Cell Fate Scores:")
        lines.append("-"*70)
        for ds_key, result in self.dataset_results.items():
            fate = result.get('fate_stats', {})
            if fate:
                lines.append(
                    f"  {ds_key}: {fate['mean']:.4f} +/- {fate['std']:.4f} "
                    f"(range {fate['min']:.4f}-{fate['max']:.4f})"
                )
        lines.append("")

        # Pathway enrichment summary
        lines.append("Mechanosensitive Pathway Enrichment (Bonferroni p < 0.05):")
        lines.append("-"*70)
        for ds_key, result in self.dataset_results.items():
            enrich = result.get('enrichment', {})
            sig = [pw for pw, s in enrich.items() if s['p_value'] * len(MECHANO_PATHWAYS) < 0.05]
            lines.append(f"  {ds_key}: {', '.join(sig) if sig else 'None'}")
        lines.append("")

        # MI zone comparison
        mi_cfg = self.config.get('analyses', {}).get('mi_zone_comparison', {})
        if mi_cfg.get('enabled', False):
            lines.append("MI Zone Comparison (Kuppe et al., Nature 2022):")
            lines.append("-"*70)
            zone_csv = self.output_dir / 'tables' / 'mi_zone_comparison.csv'
            if zone_csv.exists():
                zone_df = pd.read_csv(zone_csv)
                for _, row in zone_df.iterrows():
                    tgf = f"TGF-beta={row.get('tgf_score', 'N/A'):.4f}" if pd.notna(row.get('tgf_score')) else ''
                    fate = f"fate={row.get('fate_mean', 'N/A'):.4f}" if pd.notna(row.get('fate_mean')) else ''
                    lines.append(f"  {row['zone']:<25} {fate}  {tgf}")
            lines.append("")

        # Top L-R per dataset
        lines.append("Top 5 Causal L-R Interactions Per Dataset:")
        lines.append("-"*70)
        for ds_key, result in self.dataset_results.items():
            lr = result['lr_scores']
            col = 'causal_score' if 'causal_score' in lr.columns else 'mean_score'
            top5 = lr.nlargest(5, col)
            zone = self.config['datasets'][ds_key].get('zone', '')
            cond = result.get('condition', '?')
            label = f"{cond}/{zone}" if zone else cond
            lines.append(f"\n  {ds_key} ({label}):")
            for _, row in top5.iterrows():
                lines.append(f"    {row['ligand']:>10} -> {row['receptor']:<10} "
                             f"score={row[col]:.3f}  pathway={row['pathway']}")

        report = "\n".join(lines)

        # Save
        report_path = self.output_dir / 'reports' / 'validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # Also save structured JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
        }
        for ds_key, result in self.dataset_results.items():
            summary['datasets'][ds_key] = {
                'name': result['name'],
                'n_cells': result['n_cells'],
                'n_genes': result['n_genes'],
                'n_lr_detected': result['n_lr_detected'],
                'species': result.get('species'),
                'condition': result.get('condition'),
                'fate_stats': result.get('fate_stats', {}),
                'enrichment': {
                    pw: {k: v for k, v in stats.items() if k != 'top_genes_in_pathway'}
                    for pw, stats in result.get('enrichment', {}).items()
                },
            }

        with open(self.output_dir / 'reports' / 'validation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(report)
        return report

    # ────────────────────────────────────────────────
    # Main orchestration
    # ────────────────────────────────────────────────
    def run_all(self) -> None:
        """Run complete validation pipeline."""
        print("="*70)
        print("GRAIL-Heart External Validation Pipeline")
        print("="*70)

        # Load model and database
        self.load_model()
        self.load_lr_database()

        # Process each dataset
        for ds_key in self.config['datasets']:
            try:
                self.process_dataset(ds_key)
            except Exception as e:
                print(f"  [X] Error processing {ds_key}: {e}")
                import traceback; traceback.print_exc()

        if not self.dataset_results:
            print("\n[!] No datasets successfully processed!")
            return

        # Run analyses
        jaccard_df = self.analyse_lr_consistency()
        self.analyse_disease_vs_control()
        self.analyse_mi_zone_comparison()
        enrichment_df = self.analyse_pathway_enrichment()

        # Create figures
        print("\nGenerating figures...")
        self._plot_enrichment_heatmap(enrichment_df)
        self._plot_lr_consistency_heatmap(jaccard_df)
        self._plot_fate_comparison()
        self._plot_disease_volcano()
        self._plot_mi_zone_comparison()

        # Report
        self.generate_report()

        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("  tables/  -- CSV results per dataset & analysis")
        print("  figures/ -- Heatmaps, volcano plots, fate comparison")
        print("  reports/ -- Summary report & JSON")
        print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GRAIL-Heart External Validation')
    parser.add_argument('--config', type=str, default='configs/validation.yaml',
                        help='Path to validation config YAML')
    args = parser.parse_args()

    validator = ExternalValidator(args.config)
    validator.run_all()


if __name__ == '__main__':
    main()
