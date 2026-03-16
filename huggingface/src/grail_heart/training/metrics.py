"""
Evaluation Metrics for GRAIL-Heart

Metrics for assessing L-R prediction, cell typing accuracy,
and signaling network quality.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from typing import Dict, Optional, Tuple, List
from scipy.stats import pearsonr, spearmanr


def compute_lr_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute L-R prediction metrics.
    
    Args:
        scores: Predicted interaction scores
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    scores_np = scores.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Probabilities
    probs = 1 / (1 + np.exp(-scores_np))  # sigmoid
    preds = (probs > threshold).astype(int)
    
    metrics = {}
    
    # AUC-ROC
    if len(np.unique(targets_np)) > 1:
        metrics['auroc'] = roc_auc_score(targets_np, probs)
        metrics['auprc'] = average_precision_score(targets_np, probs)
    else:
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0
        
    # Precision, Recall, F1
    metrics['accuracy'] = accuracy_score(targets_np, preds)
    metrics['f1'] = f1_score(targets_np, preds, zero_division=0)
    
    # Per-class metrics
    if targets_np.sum() > 0:
        metrics['precision'] = (preds * targets_np).sum() / max(preds.sum(), 1)
        metrics['recall'] = (preds * targets_np).sum() / targets_np.sum()
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        
    return metrics


def compute_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        pred: Predicted expression [N, G]
        target: Ground truth expression [N, G]
        
    Returns:
        Dictionary of metrics
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    metrics = {}
    
    # MSE and MAE
    metrics['mse'] = np.mean((pred_np - target_np) ** 2)
    metrics['mae'] = np.mean(np.abs(pred_np - target_np))
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Correlation per cell
    correlations = []
    for i in range(pred_np.shape[0]):
        if np.std(target_np[i]) > 1e-8:
            r, _ = pearsonr(pred_np[i], target_np[i])
            correlations.append(r)
            
    if correlations:
        metrics['pearson_mean'] = np.mean(correlations)
        metrics['pearson_median'] = np.median(correlations)
    else:
        metrics['pearson_mean'] = 0.0
        metrics['pearson_median'] = 0.0
        
    # R-squared
    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - target_np.mean()) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    
    return metrics


def compute_cell_type_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute cell type classification metrics.
    
    Args:
        logits: Predicted logits [N, C]
        targets: Ground truth labels [N]
        
    Returns:
        Dictionary of metrics
    """
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(targets_np, preds)
    
    # Macro F1
    metrics['f1_macro'] = f1_score(targets_np, preds, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(targets_np, preds, average='weighted', zero_division=0)
    
    # Clustering metrics
    metrics['ari'] = adjusted_rand_score(targets_np, preds)
    metrics['nmi'] = normalized_mutual_info_score(targets_np, preds)
    
    return metrics


def compute_signaling_metrics(
    pred_adj: torch.Tensor,
    ref_adj: Optional[torch.Tensor] = None,
    spatial_coords: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute signaling network quality metrics.
    
    Args:
        pred_adj: Predicted adjacency matrix [N, N]
        ref_adj: Optional reference adjacency [N, N]
        spatial_coords: Optional spatial coordinates [N, 2]
        
    Returns:
        Dictionary of metrics
    """
    adj_np = pred_adj.detach().cpu().numpy()
    metrics = {}
    
    # Network statistics
    metrics['density'] = (adj_np > 0.1).mean()
    metrics['mean_weight'] = adj_np.mean()
    metrics['max_weight'] = adj_np.max()
    
    # Degree distribution
    out_degree = (adj_np > 0.1).sum(axis=1)
    metrics['mean_out_degree'] = out_degree.mean()
    metrics['std_out_degree'] = out_degree.std()
    
    # Spatial locality
    if spatial_coords is not None:
        coords_np = spatial_coords.detach().cpu().numpy()
        dist_matrix = np.sqrt(((coords_np[:, None] - coords_np[None, :]) ** 2).sum(-1))
        
        # Correlation between adjacency and inverse distance
        adj_flat = adj_np.flatten()
        inv_dist_flat = 1 / (dist_matrix.flatten() + 1e-8)
        
        if np.std(adj_flat) > 1e-8:
            r, _ = spearmanr(adj_flat, inv_dist_flat)
            metrics['spatial_correlation'] = r
        else:
            metrics['spatial_correlation'] = 0.0
            
    # Comparison to reference
    if ref_adj is not None:
        ref_np = ref_adj.detach().cpu().numpy()
        
        # MSE
        metrics['ref_mse'] = np.mean((adj_np - ref_np) ** 2)
        
        # Edge overlap
        pred_edges = adj_np > 0.1
        ref_edges = ref_np > 0.1
        
        intersection = (pred_edges & ref_edges).sum()
        union = (pred_edges | ref_edges).sum()
        
        metrics['edge_jaccard'] = intersection / max(union, 1)
        metrics['edge_precision'] = intersection / max(pred_edges.sum(), 1)
        metrics['edge_recall'] = intersection / max(ref_edges.sum(), 1)
        
    return metrics


class MetricTracker:
    """
    Tracks and aggregates metrics over training.
    
    Args:
        metrics: List of metric names to track
        prefix: Prefix for metric names (e.g., 'train', 'val')
    """
    
    def __init__(
        self,
        metrics: List[str],
        prefix: str = '',
    ):
        self.metrics = metrics
        self.prefix = prefix
        self.reset()
        
    def reset(self):
        """Reset all tracked values."""
        self._values = {m: [] for m in self.metrics}
        self._counts = {m: 0 for m in self.metrics}
        
    def update(self, values: Dict[str, float], count: int = 1):
        """
        Update tracked metrics.
        
        Args:
            values: Dictionary of metric values
            count: Number of samples (for weighted averaging)
        """
        for name, value in values.items():
            if name in self._values:
                self._values[name].append(value * count)
                self._counts[name] += count
                
    def compute(self) -> Dict[str, float]:
        """
        Compute averaged metrics.
        
        Returns:
            Dictionary of averaged metrics
        """
        result = {}
        for name in self.metrics:
            if self._counts[name] > 0:
                avg = sum(self._values[name]) / self._counts[name]
                key = f"{self.prefix}_{name}" if self.prefix else name
                result[key] = avg
                
        return result
    
    def get_best(self, metric: str, mode: str = 'max') -> float:
        """Get best value for a metric."""
        if metric not in self._values or not self._values[metric]:
            return float('-inf') if mode == 'max' else float('inf')
            
        values = [v / max(c, 1) for v, c in zip(self._values[metric], [self._counts[metric]] * len(self._values[metric]))]
        
        if mode == 'max':
            return max(values)
        else:
            return min(values)


def print_metrics(metrics: Dict[str, float], epoch: int = None):
    """Pretty print metrics."""
    if epoch is not None:
        print(f"\n=== Epoch {epoch} ===")
        
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")
