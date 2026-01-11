"""
GRAIL-Heart Benchmark Comparison Script

Compare GRAIL-Heart against baseline methods for spatial transcriptomics analysis:
1. CellPhoneDB-style L-R scoring (correlation-based)
2. CellChat-style L-R inference
3. SpaGCN (spatial GCN baseline)
4. GraphSAINT (graph sampling baseline)
5. Simple MLP (no graph structure)
6. Standard GCN (no attention)

Usage:
    python benchmark_comparison.py --config configs/default.yaml
    python benchmark_comparison.py --config configs/default.yaml --methods all
    python benchmark_comparison.py --config configs/default.yaml --methods cellphonedb,cellchat
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from grail_heart.data import (
    SpatialTranscriptomicsDataset,
    SpatialGraphBuilder,
)
from grail_heart.data.cellchat_database import get_omnipath_lr_database
from grail_heart.models import create_grail_heart
from grail_heart.models.grail_heart import GRAILHeart
from grail_heart.training import GRAILHeartLoss
from grail_heart.training.metrics import (
    compute_lr_metrics,
    compute_reconstruction_metrics,
)

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import random


CARDIAC_REGIONS = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# BASELINE METHODS
# ============================================================================

class CellPhoneDBBaseline:
    """
    CellPhoneDB-style correlation-based L-R scoring.
    
    Computes L-R interaction scores based on expression correlation
    between ligand and receptor genes in neighboring cells.
    """
    
    def __init__(self, lr_pairs_df: pd.DataFrame):
        self.lr_pairs_df = lr_pairs_df
        self.name = "CellPhoneDB"
        
    def score_interactions(
        self,
        expression: np.ndarray,
        gene_names: List[str],
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Score L-R interactions using mean expression products.
        
        Args:
            expression: Gene expression matrix [N, G]
            gene_names: List of gene names
            edge_index: Graph edges [2, E]
            
        Returns:
            Interaction scores [E]
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n_edges = edge_index.shape[1]
        scores = np.zeros(n_edges)
        
        for _, row in self.lr_pairs_df.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            if ligand not in gene_to_idx or receptor not in gene_to_idx:
                continue
                
            lig_idx = gene_to_idx[ligand]
            rec_idx = gene_to_idx[receptor]
            
            # Score = mean(ligand_expr[source]) * mean(receptor_expr[target])
            source_cells = edge_index[0]
            target_cells = edge_index[1]
            
            lig_expr = expression[source_cells, lig_idx]
            rec_expr = expression[target_cells, rec_idx]
            
            # Product-based scoring (CellPhoneDB style)
            edge_scores = lig_expr * rec_expr
            scores = np.maximum(scores, edge_scores)
            
        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
            
        return scores
    
    def fit(self, *args, **kwargs):
        """No training required for CellPhoneDB."""
        pass
    
    def predict(
        self,
        data,
        gene_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """Generate predictions."""
        expression = data.x.numpy()
        edge_index = data.edge_index.numpy()
        
        lr_scores = self.score_interactions(expression, gene_names, edge_index)
        
        return {
            'lr_scores': lr_scores,
        }


class CellChatBaseline:
    """
    CellChat-style L-R inference with tripartite averaging.
    
    Considers ligand-receptor-cofactor relationships and
    uses geometric mean of expression levels.
    """
    
    def __init__(self, lr_pairs_df: pd.DataFrame):
        self.lr_pairs_df = lr_pairs_df
        self.name = "CellChat"
        
    def score_interactions(
        self,
        expression: np.ndarray,
        gene_names: List[str],
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Score L-R interactions using geometric mean.
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n_edges = edge_index.shape[1]
        scores = np.zeros(n_edges)
        
        for _, row in self.lr_pairs_df.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            if ligand not in gene_to_idx or receptor not in gene_to_idx:
                continue
                
            lig_idx = gene_to_idx[ligand]
            rec_idx = gene_to_idx[receptor]
            
            source_cells = edge_index[0]
            target_cells = edge_index[1]
            
            lig_expr = expression[source_cells, lig_idx] + 1e-6
            rec_expr = expression[target_cells, rec_idx] + 1e-6
            
            # Geometric mean (CellChat style)
            edge_scores = np.sqrt(lig_expr * rec_expr)
            
            # Apply Hill function for saturation
            K = np.median(edge_scores[edge_scores > 0]) if (edge_scores > 0).any() else 1.0
            edge_scores = edge_scores / (K + edge_scores)
            
            scores = np.maximum(scores, edge_scores)
            
        return scores
    
    def fit(self, *args, **kwargs):
        pass
    
    def predict(
        self,
        data,
        gene_names: List[str],
    ) -> Dict[str, np.ndarray]:
        expression = data.x.numpy()
        edge_index = data.edge_index.numpy()
        
        lr_scores = self.score_interactions(expression, gene_names, edge_index)
        
        return {'lr_scores': lr_scores}


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline (no graph structure).
    
    Uses only gene expression features, ignoring spatial relationships.
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        n_cell_types: int = 10,
    ):
        super().__init__()
        self.name = "MLP"
        
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # L-R predictor (edge-level)
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, n_genes),
        )
        
        # Cell type classifier
        self.classifier = nn.Linear(hidden_dim, n_cell_types)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        # Encode
        h = self.encoder(x)
        
        # L-R prediction (concatenate source and target)
        source = h[edge_index[0]]
        target = h[edge_index[1]]
        edge_features = torch.cat([source, target], dim=-1)
        lr_scores = self.lr_predictor(edge_features).squeeze(-1)
        
        # Reconstruction
        reconstruction = self.decoder(h)
        
        # Classification
        cell_type = self.classifier(h)
        
        return {
            'lr_scores': lr_scores,
            'reconstruction': reconstruction,
            'cell_type': cell_type,
            'node_embeddings': h,
        }


class GCNBaseline(nn.Module):
    """
    Standard GCN baseline (no attention mechanism).
    
    Uses vanilla Graph Convolutional Networks without edge types or attention.
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_cell_types: int = 10,
    ):
        super().__init__()
        self.name = "GCN"
        
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
        )
        
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.decoder = nn.Linear(hidden_dim, n_genes)
        self.classifier = nn.Linear(hidden_dim, n_cell_types)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        h = self.encoder(x)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.gelu(h)
            h = F.dropout(h, p=0.1, training=self.training)
            
        # L-R prediction
        source = h[edge_index[0]]
        target = h[edge_index[1]]
        edge_features = torch.cat([source, target], dim=-1)
        lr_scores = self.lr_predictor(edge_features).squeeze(-1)
        
        # Reconstruction
        reconstruction = self.decoder(h)
        
        # Classification
        cell_type = self.classifier(h)
        
        return {
            'lr_scores': lr_scores,
            'reconstruction': reconstruction,
            'cell_type': cell_type,
            'node_embeddings': h,
        }


class GraphSAGEBaseline(nn.Module):
    """
    GraphSAGE baseline for comparison.
    
    Uses sampling-based aggregation without attention.
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_cell_types: int = 10,
    ):
        super().__init__()
        self.name = "GraphSAGE"
        
        self.encoder = nn.Linear(n_genes, hidden_dim)
        
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.decoder = nn.Linear(hidden_dim, n_genes)
        self.classifier = nn.Linear(hidden_dim, n_cell_types)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        h = self.encoder(x)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.gelu(h)
            h = F.dropout(h, p=0.1, training=self.training)
            
        source = h[edge_index[0]]
        target = h[edge_index[1]]
        edge_features = torch.cat([source, target], dim=-1)
        lr_scores = self.lr_predictor(edge_features).squeeze(-1)
        
        reconstruction = self.decoder(h)
        cell_type = self.classifier(h)
        
        return {
            'lr_scores': lr_scores,
            'reconstruction': reconstruction,
            'cell_type': cell_type,
            'node_embeddings': h,
        }


class SingleTaskGAT(nn.Module):
    """
    Single-task GAT for L-R prediction only.
    
    Baseline to show multi-task learning benefit.
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        n_cell_types: int = 10,
    ):
        super().__init__()
        self.name = "SingleTaskGAT"
        
        self.encoder = nn.Linear(n_genes, hidden_dim)
        
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, concat=True)
            for _ in range(n_layers)
        ])
        
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Placeholder decoders (not trained in single-task mode)
        self.decoder = nn.Linear(hidden_dim, n_genes)
        self.classifier = nn.Linear(hidden_dim, n_cell_types)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        h = self.encoder(x)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.gelu(h)
            h = F.dropout(h, p=0.1, training=self.training)
            
        source = h[edge_index[0]]
        target = h[edge_index[1]]
        edge_features = torch.cat([source, target], dim=-1)
        lr_scores = self.lr_predictor(edge_features).squeeze(-1)
        
        reconstruction = self.decoder(h)
        cell_type = self.classifier(h)
        
        return {
            'lr_scores': lr_scores,
            'reconstruction': reconstruction,
            'cell_type': cell_type,
            'node_embeddings': h,
        }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """
    Runs benchmark comparisons between GRAIL-Heart and baselines.
    """
    
    METHODS = {
        'cellphonedb': CellPhoneDBBaseline,
        'cellchat': CellChatBaseline,
        'mlp': MLPBaseline,
        'gcn': GCNBaseline,
        'graphsage': GraphSAGEBaseline,
        'single_task_gat': SingleTaskGAT,
        'grail_heart': GRAILHeart,
    }
    
    def __init__(
        self,
        config: dict,
        output_dir: Path,
        device: str = 'cuda',
    ):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_neural_method(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        n_epochs: int = 30,
        lr: float = 1e-4,
        single_task: bool = False,
    ) -> nn.Module:
        """Train a neural network baseline."""
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(batch)
                
                # L-R loss
                # L-R loss - use lr_labels/lr_target if available, otherwise edge_type
                if hasattr(batch, 'lr_labels') and batch.lr_labels is not None:
                    lr_loss = F.binary_cross_entropy_with_logits(
                        outputs['lr_scores'],
                        batch.lr_labels.float(),
                    )
                elif hasattr(batch, 'lr_target') and batch.lr_target is not None:
                    lr_loss = F.binary_cross_entropy_with_logits(
                        outputs['lr_scores'],
                        batch.lr_target.float(),
                    )
                elif hasattr(batch, 'edge_type'):
                    lr_loss = F.binary_cross_entropy_with_logits(
                        outputs['lr_scores'],
                        batch.edge_type.float(),
                    )
                else:
                    lr_loss = torch.tensor(0.0, device=self.device)
                
                if single_task:
                    loss = lr_loss
                else:
                    # Reconstruction loss
                    recon_loss = F.mse_loss(outputs['reconstruction'], batch.x)
                    
                    # Classification loss - handle both 'cell_type' and 'cell_type_logits' keys
                    cell_type_output = outputs.get('cell_type', outputs.get('cell_type_logits'))
                    if cell_type_output is not None and hasattr(batch, 'y') and batch.y is not None:
                        class_loss = F.cross_entropy(cell_type_output, batch.y)
                    else:
                        class_loss = torch.tensor(0.0, device=self.device)
                    
                    loss = lr_loss + recon_loss + class_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")
                
        return model
    
    def evaluate_method(
        self,
        method_name: str,
        model: Any,
        test_loader: DataLoader,
        gene_names: List[str],
    ) -> Dict[str, float]:
        """Evaluate a method on test data."""
        all_lr_scores = []
        all_lr_targets = []
        all_recon_pred = []
        all_recon_target = []
        all_cell_preds = []
        all_cell_targets = []
        
        inference_times = []
        
        for batch in test_loader:
            start_time = time.time()
            
            if method_name in ['cellphonedb', 'cellchat']:
                # Non-neural methods
                outputs = model.predict(batch, gene_names)
                lr_scores = outputs['lr_scores']
                all_lr_scores.append(lr_scores)
                
            else:
                # Neural methods
                model.eval()
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    if method_name == 'grail_heart':
                        outputs = model(batch, run_inverse=False)
                    else:
                        outputs = model(batch)
                    
                    if 'lr_scores' in outputs:
                        all_lr_scores.append(torch.sigmoid(outputs['lr_scores']).cpu().numpy())
                    if 'reconstruction' in outputs:
                        all_recon_pred.append(outputs['reconstruction'].cpu().numpy())
                        all_recon_target.append(batch.x.cpu().numpy())
                    # Handle both 'cell_type' and 'cell_type_logits' keys
                    cell_type_output = outputs.get('cell_type', outputs.get('cell_type_logits'))
                    if cell_type_output is not None and hasattr(batch, 'y'):
                        preds = cell_type_output.argmax(dim=1)
                        all_cell_preds.append(preds.cpu().numpy())
                        all_cell_targets.append(batch.y.cpu().numpy())
            
            # Use lr_labels/lr_target if available for evaluation, otherwise fall back to edge_type
            if hasattr(batch, 'lr_labels') and batch.lr_labels is not None:
                all_lr_targets.append(batch.lr_labels.cpu().numpy() if torch.is_tensor(batch.lr_labels) else batch.lr_labels)
            elif hasattr(batch, 'lr_target') and batch.lr_target is not None:
                all_lr_targets.append(batch.lr_target.cpu().numpy() if torch.is_tensor(batch.lr_target) else batch.lr_target)
            elif hasattr(batch, 'edge_type'):
                all_lr_targets.append(batch.edge_type.cpu().numpy() if torch.is_tensor(batch.edge_type) else batch.edge_type)
                
            inference_times.append(time.time() - start_time)
        
        # Compute metrics
        metrics = {
            'method': method_name,
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
        }
        
        # L-R metrics
        if all_lr_scores and all_lr_targets:
            lr_scores = np.concatenate(all_lr_scores)
            lr_targets = np.concatenate(all_lr_targets)
            
            if len(np.unique(lr_targets)) > 1:
                metrics['lr_auroc'] = roc_auc_score(lr_targets, lr_scores)
                metrics['lr_auprc'] = average_precision_score(lr_targets, lr_scores)
            else:
                metrics['lr_auroc'] = 0.0
                metrics['lr_auprc'] = 0.0
                
            preds = (lr_scores > 0.5).astype(int)
            metrics['lr_accuracy'] = accuracy_score(lr_targets, preds)
            metrics['lr_f1'] = f1_score(lr_targets, preds, zero_division=0)
        
        # Reconstruction metrics
        if all_recon_pred:
            recon_pred = np.concatenate(all_recon_pred)
            recon_target = np.concatenate(all_recon_target)
            
            metrics['recon_mse'] = np.mean((recon_pred - recon_target) ** 2)
            metrics['recon_mae'] = np.mean(np.abs(recon_pred - recon_target))
            
            # R² score
            ss_res = np.sum((recon_target - recon_pred) ** 2)
            ss_tot = np.sum((recon_target - np.mean(recon_target)) ** 2)
            metrics['recon_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Pearson correlation
            correlations = []
            for i in range(min(100, recon_pred.shape[0])):
                if np.std(recon_target[i]) > 1e-8:
                    r, _ = pearsonr(recon_pred[i], recon_target[i])
                    correlations.append(r)
            metrics['recon_pearson'] = np.mean(correlations) if correlations else 0.0
        
        # Classification metrics
        if all_cell_preds:
            cell_preds = np.concatenate(all_cell_preds)
            cell_targets = np.concatenate(all_cell_targets)
            metrics['accuracy'] = accuracy_score(cell_targets, cell_preds)
            metrics['f1'] = f1_score(cell_targets, cell_preds, average='weighted', zero_division=0)
            
        # Parameter count
        if hasattr(model, 'parameters'):
            metrics['n_params'] = sum(p.numel() for p in model.parameters())
        else:
            metrics['n_params'] = 0
            
        return metrics
    
    def run_benchmarks(
        self,
        methods: List[str],
        train_graphs: List,
        test_graphs: List,
        lr_pairs_df: pd.DataFrame,
        gene_names: List[str],
        n_cell_types: int,
    ) -> pd.DataFrame:
        """Run all benchmark comparisons."""
        results = []
        
        train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
        
        n_genes = train_graphs[0].x.shape[1]
        
        for method_name in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method_name.upper()}")
            print(f"{'='*60}")
            
            try:
                if method_name == 'cellphonedb':
                    model = CellPhoneDBBaseline(lr_pairs_df)
                    
                elif method_name == 'cellchat':
                    model = CellChatBaseline(lr_pairs_df)
                    
                elif method_name == 'mlp':
                    model = MLPBaseline(n_genes, hidden_dim=256, n_cell_types=n_cell_types)
                    print("  Training MLP baseline...")
                    model = self.train_neural_method(model, train_loader)
                    
                elif method_name == 'gcn':
                    model = GCNBaseline(n_genes, hidden_dim=256, n_cell_types=n_cell_types)
                    print("  Training GCN baseline...")
                    model = self.train_neural_method(model, train_loader)
                    
                elif method_name == 'graphsage':
                    model = GraphSAGEBaseline(n_genes, hidden_dim=256, n_cell_types=n_cell_types)
                    print("  Training GraphSAGE baseline...")
                    model = self.train_neural_method(model, train_loader)
                    
                elif method_name == 'single_task_gat':
                    model = SingleTaskGAT(n_genes, hidden_dim=256, n_cell_types=n_cell_types)
                    print("  Training Single-Task GAT baseline...")
                    model = self.train_neural_method(model, train_loader, single_task=True)
                    
                elif method_name == 'grail_heart':
                    model_config = self.config['model']
                    model = GRAILHeart(
                        n_genes=n_genes,
                        hidden_dim=model_config.get('hidden_dim', 256),
                        n_gat_layers=model_config.get('n_gat_layers', 3),
                        n_heads=model_config.get('n_heads', 8),
                        n_edge_types=model_config.get('n_edge_types', 2),
                        n_cell_types=n_cell_types,
                        n_lr_pairs=len(lr_pairs_df),
                        use_inverse_modelling=False,  # Fair comparison
                    )
                    print("  Training GRAIL-Heart...")
                    model = self.train_neural_method(model, train_loader)
                    
                else:
                    print(f"  Unknown method: {method_name}")
                    continue
                
                # Evaluate
                print("  Evaluating...")
                metrics = self.evaluate_method(method_name, model, test_loader, gene_names)
                results.append(metrics)
                
                print(f"  Results:")
                print(f"    L-R AUROC: {metrics.get('lr_auroc', 'N/A'):.4f}")
                print(f"    L-R AUPRC: {metrics.get('lr_auprc', 'N/A'):.4f}")
                if 'recon_r2' in metrics:
                    print(f"    Recon R²: {metrics.get('recon_r2', 'N/A'):.4f}")
                if 'accuracy' in metrics:
                    print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.output_dir / f'benchmark_results_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        return results_df


def get_region_from_filename(filename: str) -> Optional[str]:
    """Extract cardiac region from filename."""
    filename_upper = filename.upper()
    for region in CARDIAC_REGIONS:
        if f'_{region}_' in filename_upper or f'-{region}_' in filename_upper or filename_upper.startswith(f'VISIUM-OCT_{region}'):
            return region
    return None


def load_data(config: dict, data_dir: Path) -> Tuple[List, List, pd.DataFrame, List[str], int]:
    """Load and prepare data for benchmarks."""
    print("\n=== Loading Data ===")
    
    data_config = config['data']
    
    # Load L-R database
    cache_path = data_dir / 'lr_database_cache.csv'
    lr_pairs_df = get_omnipath_lr_database(cache_path=cache_path)
    print(f"Loaded {len(lr_pairs_df)} L-R pairs")
    
    # Find h5ad files
    atlas_dir = data_dir / 'HeartCellAtlasv2'
    h5ad_files = list(atlas_dir.glob('visium*.h5ad'))
    print(f"Found {len(h5ad_files)} h5ad files")
    
    # Build graphs
    graph_builder = SpatialGraphBuilder(
        method=data_config.get('graph_method', 'knn'),
        k=data_config.get('k_neighbors', 6),
    )
    
    all_graphs = []
    all_cell_types = set()
    gene_names = None
    
    for f in h5ad_files[:4]:  # Limit for benchmark speed
        region = get_region_from_filename(f.name)
        if region is None:
            continue
            
        try:
            ds = SpatialTranscriptomicsDataset(
                data_path=f,
                n_top_genes=data_config['n_top_genes'],
                normalize=data_config['normalize'],
                log_transform=data_config['log_transform'],
            )
            
            if not ds.has_spatial:
                continue
                
            if gene_names is None:
                gene_names = ds.gene_names
                
            graph = graph_builder.build_from_dataset(ds)
            graph.region = region
            
            # Label L-R edges for proper evaluation
            gene_to_idx = {g: idx for idx, g in enumerate(ds.gene_names)}
            n_edges = graph.edge_index.shape[1]
            lr_labels = torch.zeros(n_edges, dtype=torch.float)
            
            # Mark edges where L-R pairs are expressed
            for _, row in lr_pairs_df.iterrows():
                ligand = row['ligand']
                receptor = row['receptor']
                
                if ligand in gene_to_idx and receptor in gene_to_idx:
                    lig_idx = gene_to_idx[ligand]
                    rec_idx = gene_to_idx[receptor]
                    
                    source_cells = graph.edge_index[0]
                    target_cells = graph.edge_index[1]
                    
                    lig_expr = graph.x[source_cells, lig_idx] > 0.1
                    rec_expr = graph.x[target_cells, rec_idx] > 0.1
                    
                    lr_mask = lig_expr & rec_expr
                    lr_labels[lr_mask] = 1.0
            
            # edge_type for GAT indexing (long), lr_labels for BCE loss (float)
            graph.edge_type = (lr_labels > 0.5).long()
            graph.lr_labels = lr_labels
            
            all_graphs.append(graph)
            
            if hasattr(graph, 'y') and graph.y is not None:
                all_cell_types.update(graph.y.unique().tolist())
                
            print(f"  {region}: {graph.x.shape[0]} cells, {graph.edge_type.sum().item()} L-R edges")
            
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
            
    # Split into train/test
    train_graphs = all_graphs[:-1]
    test_graphs = all_graphs[-1:]
    
    n_cell_types = len(all_cell_types) if all_cell_types else 10
    
    return train_graphs, test_graphs, lr_pairs_df, gene_names, n_cell_types


def main():
    parser = argparse.ArgumentParser(description='GRAIL-Heart Benchmark Comparison')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--methods', type=str, default='all',
                       help='Methods to benchmark (comma-separated or "all")')
    parser.add_argument('--output_dir', type=str, default='outputs/benchmarks',
                       help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = yaml.safe_load(open(args.config, 'r'))
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    data_dir = Path(config['paths']['data_dir'])
    
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    set_seed(config['hardware']['seed'])
    
    # Load data
    train_graphs, test_graphs, lr_pairs_df, gene_names, n_cell_types = load_data(config, data_dir)
    
    if not train_graphs:
        print("No data loaded!")
        return
        
    # Determine methods
    if args.methods == 'all':
        methods = list(BenchmarkRunner.METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(',')]
    
    # Run benchmarks
    runner = BenchmarkRunner(config, output_dir, device)
    results_df = runner.run_benchmarks(
        methods=methods,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        lr_pairs_df=lr_pairs_df,
        gene_names=gene_names,
        n_cell_types=n_cell_types,
    )
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*80)
    
    # Format as table
    summary_cols = ['method', 'lr_auroc', 'lr_auprc', 'recon_r2', 'accuracy', 'n_params', 'inference_time_mean']
    available_cols = [c for c in summary_cols if c in results_df.columns]
    
    print(results_df[available_cols].to_string(index=False))
    
    # Identify best method
    if 'lr_auroc' in results_df.columns:
        best_lr = results_df.loc[results_df['lr_auroc'].idxmax()]
        print(f"\nBest L-R prediction: {best_lr['method']} (AUROC={best_lr['lr_auroc']:.4f})")
        
    if 'recon_r2' in results_df.columns:
        best_recon = results_df.loc[results_df['recon_r2'].idxmax()]
        print(f"Best reconstruction: {best_recon['method']} (R²={best_recon['recon_r2']:.4f})")


if __name__ == '__main__':
    main()
