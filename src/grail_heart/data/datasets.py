"""
Spatial Transcriptomics Dataset Loading for GRAIL-Heart

This module handles loading and preprocessing of various spatial transcriptomics
data formats including Visium (10x Genomics), h5ad (AnnData), and legacy ST formats.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import anndata as ad
from scipy import sparse as sp_sparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


def compute_cytotrace_scores(adata: ad.AnnData) -> np.ndarray:
    """
    Compute CytoTRACE-style differentiation scores from raw/normalised counts.

    CytoTRACE (Gulati et al., Science 2020) uses the number of detectably
    expressed genes per cell (gene counts) as a proxy for differentiation
    potential.  More expressed genes → higher stemness → lower differentiation
    score (closer to 0).

    Algorithm
    ---------
    1. Gene counts  = number of genes with expression > 0 per cell.
    2. Top-200 genes most positively correlated with gene counts across cells
       are selected (these are "stemness" marker genes).
    3. A stemness signature score is the mean expression of those 200 genes.
    4. The final differentiation score is 1 − normalise(stemness_score),
       so progenitors ≈ 0 and terminally differentiated cells ≈ 1.

    Parameters
    ----------
    adata : AnnData
        Should contain raw or log-normalised counts in ``adata.X``.

    Returns
    -------
    np.ndarray of shape ``(n_cells,)`` with values in ``[0, 1]``.
    """
    X = adata.X
    if sp_sparse.issparse(X):
        X_dense = np.asarray(X.todense())
    else:
        X_dense = np.asarray(X)

    n_cells, n_genes = X_dense.shape

    # 1. Gene counts per cell (number of expressed genes)
    gene_counts = (X_dense > 0).sum(axis=1).astype(np.float64)

    # 2. Correlate each gene's expression with gene_counts across cells
    # Pearson r — vectorised
    gc_centered = gene_counts - gene_counts.mean()
    gc_std = gc_centered.std() + 1e-12

    X_centered = X_dense - X_dense.mean(axis=0, keepdims=True)
    X_std = X_centered.std(axis=0, keepdims=True) + 1e-12

    correlations = (gc_centered[:, None] * X_centered).mean(axis=0) / (gc_std * X_std.squeeze())

    # 3. Top-200 genes most correlated with gene counts (stemness genes)
    n_top = min(200, n_genes)
    top_gene_idx = np.argsort(correlations)[-n_top:]

    # 4. Stemness score = mean expression of top stemness genes
    stemness = X_dense[:, top_gene_idx].mean(axis=1)

    # 5. Normalise to [0, 1] and invert so that high stemness → low score
    s_min, s_max = stemness.min(), stemness.max()
    if s_max - s_min < 1e-12:
        return np.full(n_cells, 0.5, dtype=np.float32)

    diff_score = 1.0 - (stemness - s_min) / (s_max - s_min)
    return diff_score.astype(np.float32)


class SpatialTranscriptomicsDataset(Dataset):
    """
    Dataset class for spatial transcriptomics data.
    
    Loads Visium or other spatial transcriptomics data and prepares it
    for graph neural network training.
    
    Args:
        data_path: Path to the data file (h5ad, h5, or directory)
        gene_list: Optional list of genes to subset
        normalize: Whether to normalize gene expression
        log_transform: Whether to log-transform expression values
        scale: Whether to scale to unit variance
        min_cells: Minimum cells expressing a gene to keep it
        min_genes: Minimum genes expressed in a cell to keep it
        n_top_genes: Number of highly variable genes to select (None = all)
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        gene_list: Optional[List[str]] = None,
        normalize: bool = True,
        log_transform: bool = True,
        scale: bool = False,
        min_cells: int = 3,
        min_genes: int = 200,
        n_top_genes: Optional[int] = 2000,
        spatial_key: str = 'spatial',
    ):
        self.data_path = Path(data_path)
        self.gene_list = gene_list
        self.normalize = normalize
        self.log_transform = log_transform
        self.scale = scale
        self.min_cells = min_cells
        self.min_genes = min_genes
        self.n_top_genes = n_top_genes
        self.spatial_key = spatial_key
        
        # Load and preprocess data
        self.adata = self._load_data()
        self._preprocess()
        
        # Extract arrays for PyTorch
        self.expression = self._get_expression_matrix()
        self.spatial_coords = self._get_spatial_coordinates()
        self.cell_types = self._get_cell_types()
        self.differentiation_stage = self._compute_differentiation_stage()
        
        # Prefer gene symbols over Ensembl IDs if available
        if 'SYMBOL' in self.adata.var.columns:
            self.gene_names = self.adata.var['SYMBOL'].tolist()
        elif 'gene_symbols' in self.adata.var.columns:
            self.gene_names = self.adata.var['gene_symbols'].tolist()
        elif 'gene_name' in self.adata.var.columns:
            self.gene_names = self.adata.var['gene_name'].tolist()
        else:
            self.gene_names = self.adata.var_names.tolist()
        
        self.cell_ids = self.adata.obs_names.tolist()
        
    def _load_data(self) -> ad.AnnData:
        """Load spatial transcriptomics data from various formats."""
        
        if self.data_path.suffix == '.h5ad':
            # AnnData format
            adata = sc.read_h5ad(self.data_path)
            
        elif self.data_path.suffix == '.h5':
            # 10x Genomics h5 format
            adata = sc.read_10x_h5(self.data_path)
            
        elif self.data_path.is_dir():
            # Directory with spaceranger output
            adata = self._load_visium_directory()
            
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")
            
        print(f"Loaded data: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
    
    def _load_visium_directory(self) -> ad.AnnData:
        """Load Visium data from spaceranger output directory."""
        
        # Check for filtered or raw feature matrix
        h5_file = None
        for name in ['filtered_feature_bc_matrix.h5', 'raw_feature_bc_matrix.h5']:
            path = self.data_path / name
            if path.exists():
                h5_file = path
                break
                
        if h5_file is None:
            raise FileNotFoundError(f"No h5 matrix found in {self.data_path}")
            
        # Load expression data
        adata = sc.read_10x_h5(h5_file)
        
        # Load spatial coordinates if available
        spatial_dir = self.data_path / 'spatial'
        if spatial_dir.exists():
            # Load tissue positions
            positions_file = spatial_dir / 'tissue_positions.csv'
            if not positions_file.exists():
                positions_file = spatial_dir / 'tissue_positions_list.csv'
                
            if positions_file.exists():
                positions = pd.read_csv(positions_file, header=None if 'list' in str(positions_file) else 0)
                
                # Handle different formats
                if positions.shape[1] == 6:
                    positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
                    positions = positions.set_index('barcode')
                    
                # Filter to cells in the data
                common_barcodes = adata.obs_names.intersection(positions.index)
                adata = adata[common_barcodes].copy()
                positions = positions.loc[common_barcodes]
                
                # Store spatial coordinates
                adata.obsm[self.spatial_key] = positions[['pxl_col', 'pxl_row']].values
                
            # Load scale factors
            scalefactors_file = spatial_dir / 'scalefactors_json.json'
            if scalefactors_file.exists():
                import json
                with open(scalefactors_file) as f:
                    adata.uns['spatial'] = {'scalefactors': json.load(f)}
                    
        return adata
    
    def _preprocess(self):
        """Preprocess the data: filtering, normalization, HVG selection."""
        
        # Make variable names unique
        self.adata.var_names_make_unique()
        
        # Basic filtering
        sc.pp.filter_cells(self.adata, min_genes=self.min_genes)
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)
        
        # Store raw counts
        self.adata.layers['counts'] = self.adata.X.copy()
        
        # Normalize
        if self.normalize:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            
        # Log transform
        if self.log_transform:
            sc.pp.log1p(self.adata)
            
        # Select highly variable genes
        if self.n_top_genes and self.n_top_genes < self.adata.n_vars:
            sc.pp.highly_variable_genes(
                self.adata, 
                n_top_genes=self.n_top_genes,
                flavor='seurat_v3' if 'counts' in self.adata.layers else 'seurat'
            )
            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
            
        # Subset to gene list if provided
        if self.gene_list:
            valid_genes = [g for g in self.gene_list if g in self.adata.var_names]
            if len(valid_genes) < len(self.gene_list):
                print(f"Warning: {len(self.gene_list) - len(valid_genes)} genes not found")
            self.adata = self.adata[:, valid_genes].copy()
            
        # Scale
        if self.scale:
            sc.pp.scale(self.adata, max_value=10)
            
        print(f"After preprocessing: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        
    def _get_expression_matrix(self) -> torch.Tensor:
        """Extract expression matrix as torch tensor."""
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return torch.tensor(X, dtype=torch.float32)
    
    def _get_spatial_coordinates(self) -> Optional[torch.Tensor]:
        """Extract spatial coordinates as torch tensor."""
        if self.spatial_key in self.adata.obsm:
            coords = self.adata.obsm[self.spatial_key]
            # Normalize coordinates to [0, 1]
            coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)
            return torch.tensor(coords, dtype=torch.float32)
        return None
    
    def _get_cell_types(self) -> Optional[torch.Tensor]:
        """Extract cell type labels if available."""
        # Try common column names for cell types (in order of preference)
        cell_type_columns = [
            'annotation_JC',  # HeartCellAtlas annotation
            'cell_type', 
            'celltype', 
            'annotation',
            'cell_types',
            'cluster', 
            'leiden', 
            'louvain',
        ]
        
        for col in cell_type_columns:
            if col in self.adata.obs.columns:
                labels = pd.Categorical(self.adata.obs[col])
                self.cell_type_names = labels.categories.tolist()
                print(f"Using cell type annotation from '{col}': {len(self.cell_type_names)} types")
                return torch.tensor(labels.codes, dtype=torch.long)
        return None
    
    def _compute_differentiation_stage(self) -> Optional[torch.Tensor]:
        """
        Compute differentiation staging for inverse modelling.
        
        Uses a combination of:
        1. Pseudotime analysis (if computed)
        2. Cell type-based ordering (cardiac differentiation hierarchy)
        3. Marker gene-based scoring
        
        Returns:
            Differentiation scores [0, 1] where 0=progenitor, 1=mature
        """
        # Check if pseudotime already computed
        if 'dpt_pseudotime' in self.adata.obs.columns:
            print("Using pre-computed diffusion pseudotime for differentiation staging")
            pt = self.adata.obs['dpt_pseudotime'].values
            # Normalize to [0, 1]
            pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
            return torch.tensor(pt, dtype=torch.float32)

        # ── PRIMARY: CytoTRACE-style gene-count scoring ────────────────
        try:
            scores = compute_cytotrace_scores(self.adata)
            rng = scores.max() - scores.min()
            if rng > 0.05:          # sanity check: non-trivial spread
                print(f"CytoTRACE differentiation scores "
                      f"(range: {scores.min():.3f} – {scores.max():.3f})")
                return torch.tensor(scores, dtype=torch.float32)
            else:
                print("CytoTRACE scores have near-zero range, falling back")
        except Exception as e:
            print(f"CytoTRACE scoring failed ({e}), falling back")
        
        # Try to compute pseudotime
        try:
            import scanpy as sc
            adata_copy = self.adata.copy()
            
            # Need neighbors for diffusion map
            if 'neighbors' not in adata_copy.uns:
                sc.pp.neighbors(adata_copy, n_neighbors=15)
            
            # Compute diffusion map
            sc.tl.diffmap(adata_copy, n_comps=10)
            
            # Set root cell (cell with lowest first diffusion component)
            adata_copy.uns['iroot'] = int(np.argmin(adata_copy.obsm['X_diffmap'][:, 0]))
            
            # Compute diffusion pseudotime
            sc.tl.dpt(adata_copy)
            
            pt = adata_copy.obs['dpt_pseudotime'].values
            pt = np.nan_to_num(pt, nan=0.5)  # Handle NaN values
            pt = (pt - np.nanmin(pt)) / (np.nanmax(pt) - np.nanmin(pt) + 1e-8)
            
            print(f"Computed diffusion pseudotime for differentiation staging (range: {pt.min():.3f} - {pt.max():.3f})")
            return torch.tensor(pt, dtype=torch.float32)
            
        except Exception as e:
            print(f"Could not compute pseudotime: {e}")
        
        # Fallback: Cell type-based differentiation ordering
        if self.cell_types is not None and hasattr(self, 'cell_type_names'):
            print("Using cell type-based differentiation ordering")
            
            # Cardiac differentiation hierarchy (lower = less differentiated)
            cardiac_differentiation_order = {
                # Progenitor/stem cells
                'progenitor': 0.1,
                'stem': 0.1,
                'mesenchymal': 0.2,
                
                # Supporting cells
                'fibroblast': 0.3,
                'Fibroblast': 0.3,
                'FB': 0.3,
                'endothelial': 0.4,
                'Endothelial': 0.4,
                'EC': 0.4,
                'Endo': 0.4,
                'vascular': 0.4,
                'Vascular': 0.4,
                
                # Immune cells
                'immune': 0.4,
                'Immune': 0.4,
                'macrophage': 0.4,
                'Macrophage': 0.4,
                'Myeloid': 0.4,
                'lymphocyte': 0.4,
                
                # Smooth muscle
                'smooth_muscle': 0.5,
                'Smooth_muscle': 0.5,
                'SMC': 0.5,
                'vSMC': 0.5,
                'pericyte': 0.5,
                'Pericyte': 0.5,
                
                # Cardiac muscle - early
                'atrial': 0.7,
                'Atrial': 0.7,
                'aCM': 0.7,
                'Atrial_CM': 0.7,
                
                # Cardiac muscle - mature
                'ventricular': 0.9,
                'Ventricular': 0.9,
                'vCM': 0.9,
                'Ventricular_CM': 0.9,
                'cardiomyocyte': 0.85,
                'Cardiomyocyte': 0.85,
                'CM': 0.85,
                
                # Conduction system - most specialized
                'conduction': 1.0,
                'Conduction': 1.0,
                'purkinje': 1.0,
                'Purkinje': 1.0,
                'nodal': 1.0,
            }
            
            # Assign differentiation score based on cell type
            diff_scores = np.zeros(len(self.cell_types))
            cell_types_np = self.cell_types.numpy()
            
            for i, ct_name in enumerate(self.cell_type_names):
                mask = cell_types_np == i
                # Find matching differentiation score
                score = 0.5  # Default middle value
                for key, val in cardiac_differentiation_order.items():
                    if key.lower() in ct_name.lower():
                        score = val
                        break
                diff_scores[mask] = score
            
            print(f"Assigned differentiation scores based on cell types (range: {diff_scores.min():.3f} - {diff_scores.max():.3f})")
            return torch.tensor(diff_scores, dtype=torch.float32)
        
        return None
    
    def __len__(self) -> int:
        return self.expression.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell's data."""
        item = {
            'expression': self.expression[idx],
            'idx': torch.tensor(idx, dtype=torch.long)
        }
        
        if self.spatial_coords is not None:
            item['spatial'] = self.spatial_coords[idx]
            
        if self.cell_types is not None:
            item['cell_type'] = self.cell_types[idx]
        
        if self.differentiation_stage is not None:
            item['differentiation_stage'] = self.differentiation_stage[idx]
            
        return item
    
    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]
    
    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]
    
    @property
    def n_cell_types(self) -> Optional[int]:
        if self.cell_types is not None:
            return len(torch.unique(self.cell_types))
        return None
    
    @property
    def has_spatial(self) -> bool:
        return self.spatial_coords is not None


class CardiacDataModule:
    """
    Data module for managing cardiac spatial transcriptomics datasets.
    
    Handles loading multiple datasets, creating train/val/test splits,
    and providing DataLoaders for training.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 0,
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42,
        **dataset_kwargs
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        
        self.datasets = {}
        self.combined_dataset = None
        
    def load_dataset(self, name: str, path: Union[str, Path]) -> SpatialTranscriptomicsDataset:
        """Load a single dataset."""
        dataset = SpatialTranscriptomicsDataset(path, **self.dataset_kwargs)
        self.datasets[name] = dataset
        return dataset
    
    def load_visium_datasets(self, pattern: str = "visium*.h5ad") -> Dict[str, SpatialTranscriptomicsDataset]:
        """Load all Visium datasets matching a pattern."""
        import glob
        
        files = list(self.data_dir.glob(pattern))
        for f in files:
            name = f.stem
            self.load_dataset(name, f)
            
        print(f"Loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def get_combined_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine all loaded datasets into single tensors."""
        if not self.datasets:
            raise ValueError("No datasets loaded")
            
        expressions = []
        coords = []
        cell_types = []
        
        for name, ds in self.datasets.items():
            expressions.append(ds.expression)
            if ds.spatial_coords is not None:
                coords.append(ds.spatial_coords)
            if ds.cell_types is not None:
                cell_types.append(ds.cell_types)
                
        expression = torch.cat(expressions, dim=0)
        spatial = torch.cat(coords, dim=0) if coords else None
        types = torch.cat(cell_types, dim=0) if cell_types else None
        
        return expression, spatial, types
    
    def create_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test index splits."""
        n_samples = sum(ds.n_cells for ds in self.datasets.values())
        
        torch.manual_seed(self.seed)
        indices = torch.randperm(n_samples)
        
        n_train = int(n_samples * self.train_split)
        n_val = int(n_samples * self.val_split)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        return train_idx, val_idx, test_idx


def load_heart_cell_atlas(data_dir: Path) -> Dict[str, SpatialTranscriptomicsDataset]:
    """
    Load Heart Cell Atlas v2 Visium datasets.
    
    Args:
        data_dir: Path to HeartCellAtlasv2 directory
        
    Returns:
        Dictionary of datasets by region name
    """
    datasets = {}
    regions = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']
    
    for region in regions:
        path = data_dir / f'visium-OCT_{region}_raw.h5ad'
        if path.exists():
            print(f"Loading {region}...")
            datasets[region] = SpatialTranscriptomicsDataset(
                path,
                n_top_genes=2000,
                normalize=True,
                log_transform=True
            )
            
    return datasets


def load_nature_genetics_visium(data_dir: Path) -> Dict[str, SpatialTranscriptomicsDataset]:
    """
    Load Nature Genetics developmental heart Visium datasets.
    
    Args:
        data_dir: Path to the Visium spaceranger data directory
        
    Returns:
        Dictionary of datasets by section name
    """
    datasets = {}
    
    # Find all section directories
    section_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('V')]
    
    for section_dir in section_dirs[:5]:  # Load first 5 for now
        name = section_dir.name
        print(f"Loading {name}...")
        try:
            datasets[name] = SpatialTranscriptomicsDataset(
                section_dir,
                n_top_genes=2000,
                normalize=True,
                log_transform=True
            )
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            
    return datasets
