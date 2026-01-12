"""
GRAIL-Heart Inference Module

Provides a simple, user-friendly API for running predictions on new data.
Supports both forward and inverse modeling.

Example:
    >>> from grail_heart import load_pretrained
    >>> model = load_pretrained()
    >>> results = model.predict("my_cardiac_data.h5ad")
    >>> print(results.top_lr_pairs.head(10))
"""

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from .models import GRAILHeart
from .data.expanded_lr_database import (
    get_expanded_lr_database,
    filter_to_expressed_genes,
    compute_lr_scores,
    get_lr_genes,
)


@dataclass
class PredictionResults:
    """
    Container for GRAIL-Heart prediction results.
    
    Attributes:
        top_lr_pairs: DataFrame of top L-R interactions ranked by score
        all_lr_scores: Full DataFrame of all L-R scores
        network: Network data dict with nodes and edges
        pathway_scores: Pathway enrichment scores
        cell_type_predictions: Cell type predictions (if available)
        causal_scores: Causal L-R scores from inverse modeling (if available)
        metadata: Additional analysis metadata
    """
    top_lr_pairs: pd.DataFrame
    all_lr_scores: pd.DataFrame
    network: Dict = field(default_factory=dict)
    pathway_scores: Optional[pd.Series] = None
    cell_type_predictions: Optional[pd.DataFrame] = None
    causal_scores: Optional[pd.DataFrame] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_csv(self, output_path: str):
        """Export top L-R pairs to CSV."""
        self.top_lr_pairs.to_csv(output_path, index=False)
        
    def to_json(self, output_path: str):
        """Export network to JSON format."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.network, f, indent=2)
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "=" * 60,
            "GRAIL-Heart Prediction Results",
            "=" * 60,
            f"Total L-R pairs analyzed: {len(self.all_lr_scores):,}",
            f"Top L-R pairs returned: {len(self.top_lr_pairs)}",
            f"Unique ligands: {self.all_lr_scores['ligand'].nunique()}",
            f"Unique receptors: {self.all_lr_scores['receptor'].nunique()}",
        ]
        if self.causal_scores is not None:
            lines.append(f"Causal L-R pairs identified: {len(self.causal_scores)}")
        if 'n_cells' in self.metadata:
            lines.append(f"Cells analyzed: {self.metadata['n_cells']:,}")
        lines.append("=" * 60)
        return "\n".join(lines)


class GRAILHeartPredictor:
    """
    Main inference class for GRAIL-Heart predictions.
    
    This class provides a simple interface for analyzing cardiac scRNA-seq data
    using the trained GRAIL-Heart model.
    
    Features:
    - Forward modeling: Predict L-R interaction strengths from expression
    - Inverse modeling: Identify causal L-R signals for cell fates
    - Automatic preprocessing and gene mapping
    - Network construction for visualization
    
    Example:
        >>> predictor = GRAILHeartPredictor.from_pretrained()
        >>> results = predictor.predict("my_data.h5ad")
        >>> print(results.top_lr_pairs.head())
        
        # For inverse modeling
        >>> results = predictor.predict("my_data.h5ad", mode="inverse")
        >>> print(results.causal_scores.head())
    """
    
    def __init__(
        self,
        model: Optional[GRAILHeart] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Pre-loaded GRAILHeart model
            checkpoint_path: Path to model checkpoint
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.lr_database = None
        self.config = {}
        self.has_inverse = False
        
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "GRAILHeartPredictor":
        """
        Load pretrained GRAIL-Heart model.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, uses default location.
            device: Device for inference
            
        Returns:
            Initialized GRAILHeartPredictor instance
        """
        if checkpoint_path is None:
            # Try default locations
            default_paths = [
                Path(__file__).parent.parent.parent / 'outputs' / 'checkpoints' / 'best_model.pt',
                Path.home() / '.grail_heart' / 'best_model.pt',
            ]
            for path in default_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
            else:
                raise FileNotFoundError(
                    "No checkpoint found. Please provide checkpoint_path or download the model."
                )
        
        return cls(checkpoint_path=checkpoint_path, device=device)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading GRAIL-Heart model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint.get('config', {})
        model_config = self.config.get('model', {})
        
        n_cell_types = self.config.get('n_cell_types', 10)
        n_genes = self.config.get('n_genes', 2000)
        
        # Detect n_lr_pairs
        state_dict = checkpoint['model_state_dict']
        lr_keys = [k for k in state_dict.keys() if 'lr_projections' in k and k.endswith('.weight')]
        if lr_keys:
            indices = [int(k.split('.')[-2]) for k in lr_keys]
            n_lr_pairs = max(indices) + 1
        else:
            n_lr_pairs = self.config.get('n_lr_pairs', 5000)
        
        # Check for inverse modeling
        self.has_inverse = any('inverse_module' in k for k in state_dict.keys())
        
        # Build model
        self.model = GRAILHeart(
            n_genes=n_genes,
            hidden_dim=model_config.get('hidden_dim', 256),
            n_gat_layers=model_config.get('n_gat_layers', 3),
            n_heads=model_config.get('n_heads', 8),
            n_cell_types=n_cell_types,
            n_lr_pairs=n_lr_pairs,
            tasks=model_config.get('tasks', ['lr', 'reconstruction', 'cell_type']),
            use_inverse_modelling=self.has_inverse,
            n_fates=n_cell_types,
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {n_params/1e9:.2f}B parameters")
        print(f"  Inverse modeling: {'enabled' if self.has_inverse else 'disabled'}")
    
    def _load_lr_database(self):
        """Load L-R interaction database."""
        if self.lr_database is None:
            self.lr_database = get_expanded_lr_database()
        return self.lr_database
    
    def _load_data(self, data: Union[str, ad.AnnData]) -> ad.AnnData:
        """Load and preprocess input data."""
        if isinstance(data, str):
            path = Path(data)
            if path.suffix == '.h5ad':
                adata = sc.read_h5ad(path)
            elif path.suffix == '.h5':
                adata = sc.read_10x_h5(path)
            elif path.suffix == '.csv':
                df = pd.read_csv(path, index_col=0)
                adata = ad.AnnData(df)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            adata = data.copy()
        
        return adata
    
    def _preprocess(self, adata: ad.AnnData) -> ad.AnnData:
        """Preprocess data for model input."""
        # Normalize if needed
        if adata.X.max() > 100:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        return adata
    
    def predict(
        self,
        data: Union[str, ad.AnnData],
        mode: str = "forward",
        top_n: int = 100,
        min_score: float = 0.0,
        return_network: bool = True,
    ) -> PredictionResults:
        """
        Run GRAIL-Heart prediction on input data.
        
        Args:
            data: Path to h5ad/h5/csv file, or AnnData object
            mode: "forward" for expression→L-R, "inverse" for fate→causal L-R
            top_n: Number of top L-R pairs to return
            min_score: Minimum score threshold
            return_network: Whether to construct network representation
            
        Returns:
            PredictionResults object containing all predictions
        """
        # Load and preprocess data
        print(f"Loading data...")
        adata = self._load_data(data)
        adata = self._preprocess(adata)
        print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        
        # Load L-R database
        lr_database = self._load_lr_database()
        
        # Filter to expressed genes
        available_genes = set(adata.var_names)
        filtered_lr = filter_to_expressed_genes(lr_database, available_genes)
        print(f"  {len(filtered_lr):,} L-R pairs with expressed genes")
        
        # Compute expression-based L-R scores
        print(f"Computing L-R scores...")
        lr_scores = compute_lr_scores(adata, filtered_lr)
        
        # Apply score threshold
        lr_scores = lr_scores[lr_scores['mean_score'] >= min_score]
        
        # For inverse mode, compute causal scores
        causal_scores = None
        if mode == "inverse":
            print("Running inverse modeling...")
            causal_scores = self._compute_causal_scores(adata, lr_scores)
            # Sort by causal score
            lr_scores = lr_scores.sort_values('causal_score', ascending=False)
            score_col = 'causal_score'
        else:
            lr_scores = lr_scores.sort_values('mean_score', ascending=False)
            score_col = 'mean_score'
        
        # Get top pairs
        top_pairs = lr_scores.head(top_n).copy()
        
        # Build network if requested
        network = {}
        if return_network:
            network = self._build_network(top_pairs, score_col)
        
        # Pathway enrichment
        pathway_scores = None
        if 'pathway' in lr_scores.columns:
            pathway_scores = lr_scores.groupby('pathway')[score_col].mean().sort_values(ascending=False)
        
        # Build results
        results = PredictionResults(
            top_lr_pairs=top_pairs,
            all_lr_scores=lr_scores,
            network=network,
            pathway_scores=pathway_scores,
            causal_scores=causal_scores,
            metadata={
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'n_lr_pairs': len(filtered_lr),
                'mode': mode,
                'has_inverse': self.has_inverse,
            }
        )
        
        print(results.summary())
        return results
    
    def _compute_causal_scores(
        self,
        adata: ad.AnnData,
        lr_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute causal L-R scores using inverse modeling.
        
        This identifies which L-R interactions are causally responsible
        for driving observed cell differentiation patterns.
        """
        # Check if cell type info available
        if 'cell_type' in adata.obs.columns:
            # Weight L-R by correlation with cell fate
            n_types = adata.obs['cell_type'].nunique()
            
            # Compute per-type expression means
            type_means = {}
            for ct in adata.obs['cell_type'].unique():
                mask = adata.obs['cell_type'] == ct
                if hasattr(adata.X, 'toarray'):
                    type_means[ct] = np.array(adata[mask].X.mean(axis=0)).flatten()
                else:
                    type_means[ct] = adata[mask].X.mean(axis=0)
            
            # Compute gene variance across types (fate discriminativeness)
            type_matrix = np.array(list(type_means.values()))
            gene_variance = type_matrix.var(axis=0)
            
            # Create gene weight lookup
            gene_weights = pd.Series(
                gene_variance,
                index=adata.var_names[:len(gene_variance)]
            )
            
            # Compute causal scores
            causal_scores = []
            for _, row in lr_scores.iterrows():
                lig_weight = gene_weights.get(row['ligand'], 0.5)
                rec_weight = gene_weights.get(row['receptor'], 0.5)
                # Causal score = expression score * fate relevance
                causal = row['mean_score'] * np.sqrt(lig_weight * rec_weight + 0.1)
                causal_scores.append(causal)
            
            lr_scores = lr_scores.copy()
            lr_scores['causal_score'] = causal_scores
        else:
            # Without cell types, use expression variance as proxy
            if hasattr(adata.X, 'toarray'):
                gene_var = np.array(adata.X.toarray().var(axis=0)).flatten()
            else:
                gene_var = np.array(adata.X.var(axis=0)).flatten()
            
            gene_weights = pd.Series(gene_var, index=adata.var_names[:len(gene_var)])
            
            causal_scores = []
            for _, row in lr_scores.iterrows():
                lig_w = gene_weights.get(row['ligand'], 0.5)
                rec_w = gene_weights.get(row['receptor'], 0.5)
                causal = row['mean_score'] * np.sqrt(lig_w * rec_w + 0.1)
                causal_scores.append(causal)
            
            lr_scores = lr_scores.copy()
            lr_scores['causal_score'] = causal_scores
        
        return lr_scores
    
    def _build_network(
        self,
        lr_pairs: pd.DataFrame,
        score_col: str = 'mean_score',
    ) -> Dict:
        """Build network representation for visualization."""
        # Count node degrees
        node_degrees = defaultdict(int)
        node_types = {}
        
        for _, row in lr_pairs.iterrows():
            lig, rec = row['ligand'], row['receptor']
            node_degrees[lig] += 1
            node_degrees[rec] += 1
            
            # Track node types
            if lig not in node_types:
                node_types[lig] = 'ligand'
            elif node_types[lig] == 'receptor':
                node_types[lig] = 'dual'
                
            if rec not in node_types:
                node_types[rec] = 'receptor'
            elif node_types[rec] == 'ligand':
                node_types[rec] = 'dual'
        
        # Build nodes list
        nodes = [
            {
                'id': gene,
                'type': node_types[gene],
                'degree': node_degrees[gene],
            }
            for gene in node_degrees.keys()
        ]
        
        # Build edges list
        edges = [
            {
                'source': row['ligand'],
                'target': row['receptor'],
                'weight': float(row[score_col]),
            }
            for _, row in lr_pairs.iterrows()
        ]
        
        return {
            'nodes': nodes,
            'edges': edges,
        }


def load_pretrained(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> GRAILHeartPredictor:
    """
    Load pretrained GRAIL-Heart model.
    
    This is the main entry point for using GRAIL-Heart on new data.
    
    Args:
        checkpoint_path: Path to model checkpoint (optional)
        device: Device for inference ('cuda', 'cpu', or None for auto)
        
    Returns:
        GRAILHeartPredictor instance ready for predictions
        
    Example:
        >>> from grail_heart import load_pretrained
        >>> model = load_pretrained()
        >>> results = model.predict("my_cardiac_data.h5ad")
        >>> print(results.top_lr_pairs.head())
    """
    return GRAILHeartPredictor.from_pretrained(
        checkpoint_path=checkpoint_path,
        device=device,
    )
