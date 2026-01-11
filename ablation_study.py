"""
GRAIL-Heart Ablation Study Script

Systematic ablation experiments to quantify the contribution of each model component:
1. Edge-type awareness (spatial vs L-R edges)
2. Number of GAT layers (including 1-layer with special handling)
3. Attention heads
4. Multi-task learning contributions (including single-task experiments)
5. Graph construction methods
6. Encoder architectures
7. Hidden dimensions (including 512 with memory optimization)

Usage:
    python ablation_study.py --config configs/ablation.yaml
    python ablation_study.py --config configs/ablation.yaml --ablation edge_types
    python ablation_study.py --config configs/ablation.yaml --ablation all
    python ablation_study.py --config configs/ablation.yaml --ablation tasks  # For multi-task ablation
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
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy
import warnings

# Suppress overflow warnings - we handle them with safe_sigmoid
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from grail_heart.data import (
    SpatialTranscriptomicsDataset,
    SpatialGraphBuilder,
)
from grail_heart.data.cellchat_database import get_omnipath_lr_database
from grail_heart.models import create_grail_heart
from grail_heart.models.grail_heart import GRAILHeart
from grail_heart.training import (
    GRAILHeartLoss,
    GRAILHeartTrainer,
    create_optimizer,
    create_scheduler,
)

from torch_geometric.loader import DataLoader
import random

# Cardiac regions
CARDIAC_REGIONS = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_cuda_memory(force=False):
    """Clear CUDA memory to prevent OOM errors between experiments."""
    # First, run garbage collection (this should always work)
    try:
        gc.collect()
    except Exception:
        pass
    
    if not torch.cuda.is_available():
        return
    
    try:
        # Try standard cleanup
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    
    # If force mode, try harder with device reset
    if force:
        try:
            # Reset all CUDA memory statistics
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass
        
        try:
            # Try to reset the CUDA device (nuclear option)
            torch.cuda.ipc_collect()
        except Exception:
            pass


def reset_cuda_state():
    """
    Attempt to fully reset CUDA state after critical OOM.
    This is the nuclear option when standard cleanup fails.
    """
    try:
        gc.collect()
    except Exception:
        pass
    
    if torch.cuda.is_available():
        try:
            # Clear all cached memory
            torch.cuda.empty_cache()
        except Exception:
            pass
        
        try:
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        
        try:
            # Synchronize all streams
            torch.cuda.synchronize()
        except Exception:
            pass
        
        try:
            # IPC collect for multi-process scenarios
            torch.cuda.ipc_collect()
        except Exception:
            pass


def move_model_to_cpu(model):
    """Safely move model to CPU to free GPU memory in critical OOM situations."""
    try:
        model.cpu()
        clear_cuda_memory(force=False)
    except Exception as e:
        print(f"Warning: Could not move model to CPU: {e}")


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    # Use stable computation
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    result = np.zeros_like(x)
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)
    return result


def compute_lr_metrics_safe(scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute L-R metrics with overflow protection."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
    
    scores_np = scores.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Convert targets to binary (0/1)
    targets_binary = (targets_np > 0.5).astype(int)
    
    # Safe sigmoid
    probs = safe_sigmoid(scores_np)
    preds = (probs > 0.5).astype(int)
    
    metrics = {}
    
    # Check if we have both classes
    unique_targets = np.unique(targets_binary)
    if len(unique_targets) > 1:
        try:
            metrics['auroc'] = roc_auc_score(targets_binary, probs)
            metrics['auprc'] = average_precision_score(targets_binary, probs)
        except Exception:
            metrics['auroc'] = 0.5
            metrics['auprc'] = 0.5
    else:
        metrics['auroc'] = 0.5
        metrics['auprc'] = 0.5
        
    metrics['accuracy'] = accuracy_score(targets_binary, preds)
    metrics['f1'] = f1_score(targets_binary, preds, zero_division=0)
    
    return metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class AblationExperiment:
    """
    Manages ablation experiments for GRAIL-Heart.
    
    Experiments:
    1. edge_types: Test single vs dual edge types
    2. gat_layers: Test 1, 2, 3, 4 GAT layers (1-layer uses special jk=None)
    3. attention_heads: Test 1, 2, 4, 8 heads
    4. tasks: Test all multi-task combinations including single-task
    5. graph_method: Test knn with different k values
    6. encoder: Test MLP vs Variational encoder
    7. hidden_dim: Test 64, 128, 256, 512 dimensions (512 uses memory optimization)
    """
    
    ABLATION_CONFIGS = {
        'edge_types': [
            {'name': 'spatial_only', 'n_edge_types': 1},
            {'name': 'dual_edges', 'n_edge_types': 2},  # baseline
        ],
        'gat_layers': [
            # 1 layer requires jk=None to avoid projection dimension mismatch
            {'name': '1_layer', 'n_gat_layers': 1, 'jk_mode': None},
            {'name': '2_layers', 'n_gat_layers': 2, 'jk_mode': 'cat'},
            {'name': '3_layers', 'n_gat_layers': 3, 'jk_mode': 'cat'},  # baseline
            {'name': '4_layers', 'n_gat_layers': 4, 'jk_mode': 'cat'},
        ],
        'attention_heads': [
            {'name': '1_head', 'n_heads': 1},
            {'name': '4_heads', 'n_heads': 4},
            {'name': '8_heads', 'n_heads': 8},  # baseline
        ],
        'tasks': [
            # Single-task experiments - need special loss handling
            {'name': 'lr_only', 'tasks': ['lr'], 'single_task': True},
            {'name': 'recon_only', 'tasks': ['reconstruction'], 'single_task': True},
            {'name': 'classify_only', 'tasks': ['cell_type'], 'single_task': True},
            # Multi-task combinations
            {'name': 'lr_recon', 'tasks': ['lr', 'reconstruction'], 'single_task': False},
            {'name': 'lr_classify', 'tasks': ['lr', 'cell_type'], 'single_task': False},
            {'name': 'recon_classify', 'tasks': ['reconstruction', 'cell_type'], 'single_task': False},
            {'name': 'all_tasks', 'tasks': ['lr', 'reconstruction', 'cell_type'], 'single_task': False},  # baseline
        ],
        'graph_method': [
            {'name': 'knn_4', 'k_neighbors': 4},
            {'name': 'knn_6', 'k_neighbors': 6},  # baseline
            {'name': 'knn_10', 'k_neighbors': 10},
        ],
        'encoder': [
            {'name': 'mlp_encoder', 'use_variational': False},  # baseline
            {'name': 'variational_encoder', 'use_variational': True},
        ],
        'hidden_dim': [
            {'name': 'dim_64', 'hidden_dim': 64, 'encoder_dims': [256, 64]},
            {'name': 'dim_128', 'hidden_dim': 128, 'encoder_dims': [384, 128]},
            {'name': 'dim_256', 'hidden_dim': 256, 'encoder_dims': [512, 256]},  # baseline
            # 512-dim uses gradient accumulation to avoid OOM
            {'name': 'dim_512', 'hidden_dim': 512, 'encoder_dims': [1024, 512], 'memory_optimized': True},
        ],
        'decoder_type': [
            {'name': 'basic_decoder', 'decoder_type': 'basic'},
            {'name': 'residual_decoder', 'decoder_type': 'residual'},  # baseline
        ],
        'dropout': [
            {'name': 'dropout_0', 'dropout': 0.0},
            {'name': 'dropout_0.1', 'dropout': 0.1},  # baseline
            {'name': 'dropout_0.2', 'dropout': 0.2},
        ],
    }
    
    def __init__(
        self,
        base_config: dict,
        output_dir: Path,
        device: str = 'cuda',
    ):
        self.base_config = base_config
        self.output_dir = output_dir
        self.device = device
        self.results = defaultdict(list)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
    def create_ablation_config(
        self,
        ablation_type: str,
        ablation_settings: dict,
    ) -> dict:
        """Create a modified config for ablation experiment."""
        config = copy.deepcopy(self.base_config)
        
        # Apply ablation-specific settings
        for key, value in ablation_settings.items():
            if key in ['name', 'single_task', 'memory_optimized', 'jk_mode']:
                continue  # These are metadata, not config params
            elif key in ['n_gat_layers', 'n_heads', 'hidden_dim', 'encoder_dims',
                        'use_variational', 'tasks', 'n_edge_types', 'decoder_type',
                        'use_inverse_modelling', 'dropout']:
                config['model'][key] = value
            elif key in ['graph_method', 'k_neighbors', 'radius']:
                config['data'][key] = value
        
        # Store special flags in config for run_single_experiment to use
        config['_ablation_meta'] = {
            'jk_mode': ablation_settings.get('jk_mode', 'cat'),
            'single_task': ablation_settings.get('single_task', False),
            'memory_optimized': ablation_settings.get('memory_optimized', False),
        }
                
        return config
    
    def run_single_experiment(
        self,
        config: dict,
        experiment_name: str,
        train_graphs: List,
        val_graphs: List,
        lr_pairs_df: pd.DataFrame,
        n_cell_types: int,
    ) -> Dict[str, float]:
        """Run a single ablation experiment."""
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"{'='*60}")
        
        # Clear memory before each experiment
        clear_cuda_memory()
        
        set_seed(config['hardware']['seed'])
        
        # Get dimensions from first graph
        sample_graph = train_graphs[0]
        n_genes = sample_graph.x.shape[1]
        
        # Get ablation metadata
        ablation_meta = config.get('_ablation_meta', {})
        jk_mode = ablation_meta.get('jk_mode', 'cat')
        is_single_task = ablation_meta.get('single_task', False)
        memory_optimized = ablation_meta.get('memory_optimized', False)
        
        # Create model with ablation config
        model_config = config['model']
        active_tasks = model_config.get('tasks', ['lr', 'reconstruction', 'cell_type'])
        
        try:
            # Build model with potentially modified architecture
            model = self._create_model_with_jk(
                n_genes=n_genes,
                model_config=model_config,
                n_cell_types=n_cell_types,
                n_lr_pairs=len(lr_pairs_df),
                jk_mode=jk_mode,
            )
            model = model.to(self.device)
        except Exception as e:
            print(f"Error creating model: {e}")
            import traceback
            traceback.print_exc()
            return {'experiment': experiment_name, 'error': str(e)}
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        print(f"Active tasks: {active_tasks}")
        print(f"JK mode: {jk_mode}, Single task: {is_single_task}, Memory optimized: {memory_optimized}")
        
        # Create optimizer
        train_config = config['training']
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
        )
        
        # Training loop (reduced epochs for ablation)
        n_epochs = config.get('ablation', {}).get('n_epochs', 30)
        best_val_loss = float('inf')
        best_metrics = {}
        patience = 10
        patience_counter = 0
        
        # Use gradient accumulation for memory-optimized experiments
        accumulation_steps = 4 if memory_optimized else 1
        
        # Get n_edge_types for clamping edge_type indices
        n_edge_types = model_config.get('n_edge_types', 2)
        
        train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                
                # Clamp edge_type to valid range for this model's n_edge_types
                # This is critical for spatial_only (n_edge_types=1) ablation
                if hasattr(batch, 'edge_type') and batch.edge_type is not None:
                    batch.edge_type = batch.edge_type.clamp(0, n_edge_types - 1)
                
                try:
                    outputs = model(batch, run_inverse=False)
                    
                    # Compute losses based on active tasks with gradient-safe handling
                    loss = self._compute_task_losses(
                        outputs, batch, active_tasks, is_single_task
                    )
                    
                    if loss is not None:
                        try:
                            loss_value = loss.item()
                        except RuntimeError as item_e:
                            if 'out of memory' in str(item_e).lower():
                                print(f"OOM at batch {batch_idx} when extracting loss value, skipping...")
                                raise RuntimeError('out of memory')
                            raise
                        
                        if loss_value > 0:
                            # Scale loss for gradient accumulation
                            if accumulation_steps > 1:
                                loss = loss / accumulation_steps
                            
                            loss.backward()
                            
                            # Update weights after accumulation
                            if (batch_idx + 1) % accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                                optimizer.zero_grad()
                            
                            train_loss += loss_value * (1.0 if accumulation_steps == 1 else accumulation_steps / accumulation_steps)
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"OOM at batch {batch_idx}, skipping...")
                        try:
                            optimizer.zero_grad()
                        except:
                            pass
                        clear_cuda_memory(force=False)
                        continue
                    raise
                
            # Handle remaining gradients
            if len(train_loader) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss /= max(len(train_loader), 1)
            
            # Validation
            model.eval()
            val_metrics = self.evaluate(model, val_loader, model_config, active_tasks)
            
            if val_metrics.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        
        best_metrics['n_params'] = n_params
        best_metrics['experiment'] = experiment_name
        
        # Clean up model
        del model
        clear_cuda_memory()
        
        return best_metrics
    
    def _create_model_with_jk(
        self,
        n_genes: int,
        model_config: dict,
        n_cell_types: int,
        n_lr_pairs: int,
        jk_mode: Optional[str],
    ) -> GRAILHeart:
        """Create model with specified JK mode - handles 1-layer case."""
        from grail_heart.models.gat_layers import GATStack
        
        # Standard model creation
        model = GRAILHeart(
            n_genes=n_genes,
            hidden_dim=model_config.get('hidden_dim', 256),
            n_gat_layers=model_config.get('n_gat_layers', 3),
            n_heads=model_config.get('n_heads', 8),
            n_edge_types=model_config.get('n_edge_types', 2),
            n_cell_types=n_cell_types,
            n_lr_pairs=n_lr_pairs,
            encoder_dims=model_config.get('encoder_dims', [512, 256]),
            dropout=model_config.get('dropout', 0.1),
            use_spatial=model_config.get('use_spatial', True),
            use_variational=model_config.get('use_variational', False),
            tasks=model_config.get('tasks', ['lr', 'reconstruction', 'cell_type']),
            decoder_type=model_config.get('decoder_type', 'residual'),
            use_inverse_modelling=False,  # Disable for ablation to save memory
        )
        
        # Replace GAT with correct JK mode if different from default
        n_gat_layers = model_config.get('n_gat_layers', 3)
        hidden_dim = model_config.get('hidden_dim', 256)
        n_heads = model_config.get('n_heads', 8)
        n_edge_types = model_config.get('n_edge_types', 2)
        dropout = model_config.get('dropout', 0.1)
        
        # For 1-layer, we need jk=None to avoid dimension issues
        if jk_mode is None or n_gat_layers == 1:
            model.gat = GATStack(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=n_gat_layers,
                heads=n_heads,
                n_edge_types=n_edge_types,
                dropout=dropout,
                jk=None,  # No jumping knowledge for 1-layer
            )
        
        return model
    
    def _compute_task_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,
        active_tasks: List[str],
        is_single_task: bool,
    ) -> Optional[torch.Tensor]:
        """
        Compute task losses with proper gradient handling for single-task experiments.
        
        The key issue with single-task is that if we initialize loss = 0 (a Python float),
        and only one task computes a loss, the gradient won't flow. We need to ensure
        at least one task produces a valid loss tensor.
        """
        losses = []
        
        # L-R prediction loss
        if 'lr' in active_tasks and 'lr_scores' in outputs:
            # Use lr_labels (float) for BCE loss, fall back to edge_type.float()
            if hasattr(batch, 'lr_labels') and batch.lr_labels is not None:
                lr_targets = batch.lr_labels
            elif hasattr(batch, 'edge_type') and batch.edge_type is not None:
                lr_targets = batch.edge_type.float()
            else:
                lr_targets = None
            
            if lr_targets is not None and lr_targets.shape[0] == outputs['lr_scores'].shape[0]:
                lr_loss = F.binary_cross_entropy_with_logits(
                    outputs['lr_scores'], lr_targets
                )
                losses.append(lr_loss)
        
        # Reconstruction loss
        if 'reconstruction' in active_tasks and 'reconstruction' in outputs:
            recon_loss = F.mse_loss(outputs['reconstruction'], batch.x)
            losses.append(recon_loss)
        
        # Classification loss
        if 'cell_type' in active_tasks and 'cell_type' in outputs:
            if hasattr(batch, 'y') and batch.y is not None:
                # Ensure y is long for cross_entropy
                cell_targets = batch.y.long()
                if cell_targets.max() < outputs['cell_type'].shape[1]:
                    cell_loss = F.cross_entropy(outputs['cell_type'], cell_targets)
                    losses.append(cell_loss)
        
        # Combine losses
        if not losses:
            return None
        
        total_loss = sum(losses)
        return total_loss
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        model_config: dict,
        active_tasks: List[str],
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        model.eval()
        
        # Get n_edge_types for clamping
        n_edge_types = model_config.get('n_edge_types', 2)
        
        all_lr_scores = []
        all_lr_targets = []
        all_recon_pred = []
        all_recon_target = []
        all_cell_preds = []
        all_cell_targets = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # Clamp edge_type to valid range for this model
                if hasattr(batch, 'edge_type') and batch.edge_type is not None:
                    batch.edge_type = batch.edge_type.clamp(0, n_edge_types - 1)
                
                outputs = model(batch, run_inverse=False)
                
                # L-R metrics - use lr_labels (float) if available
                if 'lr' in active_tasks and 'lr_scores' in outputs:
                    if hasattr(batch, 'lr_labels') and batch.lr_labels is not None:
                        all_lr_scores.append(outputs['lr_scores'].cpu())
                        all_lr_targets.append(batch.lr_labels.cpu())
                    elif hasattr(batch, 'edge_type') and batch.edge_type is not None:
                        all_lr_scores.append(outputs['lr_scores'].cpu())
                        all_lr_targets.append(batch.edge_type.float().cpu())
                
                # Reconstruction metrics
                if 'reconstruction' in active_tasks and 'reconstruction' in outputs:
                    all_recon_pred.append(outputs['reconstruction'].cpu())
                    all_recon_target.append(batch.x.cpu())
                    recon_loss = F.mse_loss(outputs['reconstruction'], batch.x)
                    total_loss += recon_loss.item()
                    n_batches += 1
                
                # Classification metrics
                if 'cell_type' in active_tasks and 'cell_type' in outputs:
                    if hasattr(batch, 'y') and batch.y is not None:
                        preds = outputs['cell_type'].argmax(dim=1)
                        all_cell_preds.append(preds.cpu())
                        all_cell_targets.append(batch.y.cpu())
        
        metrics = {'val_loss': total_loss / max(n_batches, 1)}
        
        # Compute L-R metrics
        if all_lr_scores and all_lr_targets:
            lr_scores = torch.cat(all_lr_scores)
            lr_targets = torch.cat(all_lr_targets)
            lr_metrics = compute_lr_metrics_safe(lr_scores, lr_targets)
            metrics.update({f'lr_{k}': v for k, v in lr_metrics.items()})
        
        # Compute reconstruction metrics
        if all_recon_pred and all_recon_target:
            recon_pred = torch.cat(all_recon_pred).numpy()
            recon_target = torch.cat(all_recon_target).numpy()
            
            # MSE
            metrics['recon_mse'] = np.mean((recon_pred - recon_target) ** 2)
            
            # R² score
            ss_res = np.sum((recon_target - recon_pred) ** 2)
            ss_tot = np.sum((recon_target - np.mean(recon_target)) ** 2)
            metrics['recon_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Pearson correlation (sample of cells)
            from scipy.stats import pearsonr
            correlations = []
            n_samples = min(100, recon_pred.shape[0])
            indices = np.random.choice(recon_pred.shape[0], n_samples, replace=False)
            for i in indices:
                if np.std(recon_target[i]) > 1e-8:
                    r, _ = pearsonr(recon_pred[i], recon_target[i])
                    if not np.isnan(r):
                        correlations.append(r)
            metrics['recon_pearson'] = np.mean(correlations) if correlations else 0.0
        
        # Compute classification metrics
        if all_cell_preds and all_cell_targets:
            cell_preds = torch.cat(all_cell_preds).numpy()
            cell_targets = torch.cat(all_cell_targets).numpy()
            metrics['accuracy'] = (cell_preds == cell_targets).mean()
            
        return metrics
    
    def run_ablation_suite(
        self,
        ablation_types: List[str],
        train_graphs: List,
        val_graphs: List,
        lr_pairs_df: pd.DataFrame,
        n_cell_types: int,
    ) -> pd.DataFrame:
        """Run full suite of ablation experiments."""
        results = []
        
        for ablation_type in ablation_types:
            if ablation_type not in self.ABLATION_CONFIGS:
                print(f"Unknown ablation type: {ablation_type}")
                continue
                
            print(f"\n{'#'*60}")
            print(f"ABLATION STUDY: {ablation_type}")
            print(f"{'#'*60}")
            
            for settings in self.ABLATION_CONFIGS[ablation_type]:
                experiment_name = f"{ablation_type}/{settings['name']}"
                config = self.create_ablation_config(ablation_type, settings)
                
                try:
                    # Clear memory before each experiment
                    clear_cuda_memory()
                    
                    # Check if CUDA is still healthy before starting experiment
                    if torch.cuda.is_available():
                        try:
                            # Simple test to verify CUDA is working
                            test_tensor = torch.zeros(1, device='cuda')
                            del test_tensor
                            torch.cuda.empty_cache()
                        except Exception as cuda_check_error:
                            print(f"\n{'!'*60}")
                            print(f"CUDA is in an unhealthy state. Cannot continue experiments.")
                            print(f"Error: {cuda_check_error}")
                            print(f"Saving partial results and exiting...")
                            print(f"{'!'*60}\n")
                            # Save partial results and exit the loop
                            break
                    
                    metrics = self.run_single_experiment(
                        config=config,
                        experiment_name=experiment_name,
                        train_graphs=train_graphs,
                        val_graphs=val_graphs,
                        lr_pairs_df=lr_pairs_df,
                        n_cell_types=n_cell_types,
                    )
                    
                    metrics['ablation_type'] = ablation_type
                    metrics['ablation_setting'] = settings['name']
                    results.append(metrics)
                    
                except Exception as e:
                    error_str = str(e).lower()
                    print(f"Error in {experiment_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check if this is an unrecoverable CUDA error
                    if 'cuda' in error_str and 'out of memory' in error_str:
                        print(f"\n{'!'*60}")
                        print("CRITICAL: CUDA OOM error detected.")
                        print("The GPU is in an unrecoverable state within this process.")
                        print("Saving partial results and terminating ablation study.")
                        print("To continue, please restart the script.")
                        print(f"{'!'*60}\n")
                        
                        # Add error entry for this experiment
                        results.append({
                            'experiment': experiment_name,
                            'ablation_type': ablation_type,
                            'ablation_setting': settings['name'],
                            'error': 'CUDA OOM - unrecoverable',
                        })
                        
                        # Break out of both loops by raising a custom exception
                        raise RuntimeError("CUDA_UNRECOVERABLE")
                    
                    # Try to recover from other errors
                    try:
                        reset_cuda_state()
                    except:
                        pass
            else:
                # This else belongs to the for loop - only executes if loop wasn't broken
                continue
            # If we broke out of the inner loop, break outer loop too
            break
                    
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.output_dir / f'ablation_results_{timestamp}.csv'
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


def load_data(config: dict, data_dir: Path, lr_pairs_df: pd.DataFrame) -> Tuple[List, List, int]:
    """Load and prepare data for ablation studies."""
    print("\n=== Loading Data ===")
    
    data_config = config['data']
    
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
    
    # Use only 3 files for faster ablation
    for f in h5ad_files[:3]:
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
                
            graph = graph_builder.build_from_dataset(ds)
            graph.region = region
            
            # Label L-R edges for proper evaluation
            # edge_type must be LONG for GAT indexing (self.att_src[edge_type])
            # We store lr_labels separately as float for BCE loss
            gene_to_idx = {g: idx for idx, g in enumerate(ds.gene_names)}
            n_edges = graph.edge_index.shape[1]
            lr_labels = torch.zeros(n_edges, dtype=torch.float)  # For BCE loss
            
            # Mark edges where L-R pairs are expressed
            for _, row in lr_pairs_df.iterrows():
                ligand = row['ligand']
                receptor = row['receptor']
                
                if ligand in gene_to_idx and receptor in gene_to_idx:
                    lig_idx = gene_to_idx[ligand]
                    rec_idx = gene_to_idx[receptor]
                    
                    # Check expression
                    source_cells = graph.edge_index[0]
                    target_cells = graph.edge_index[1]
                    
                    lig_expr = graph.x[source_cells, lig_idx] > 0.1
                    rec_expr = graph.x[target_cells, rec_idx] > 0.1
                    
                    lr_mask = lig_expr & rec_expr
                    lr_labels[lr_mask] = 1.0
            
            # edge_type for GAT: 0=spatial, 1=L-R (must be long for indexing)
            graph.edge_type = (lr_labels > 0.5).long()
            # Store float labels for BCE loss
            graph.lr_labels = lr_labels
            all_graphs.append(graph)
            
            if hasattr(graph, 'y') and graph.y is not None:
                all_cell_types.update(graph.y.unique().tolist())
                
            print(f"  {region}: {graph.x.shape[0]} cells, {graph.edge_type.sum().item()} L-R edges")
            
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
            
    # Split into train/val
    if len(all_graphs) >= 2:
        train_graphs = all_graphs[:-1]
        val_graphs = all_graphs[-1:]
    else:
        train_graphs = all_graphs
        val_graphs = all_graphs
    
    n_cell_types = len(all_cell_types) if all_cell_types else 10
    
    return train_graphs, val_graphs, n_cell_types


def main():
    parser = argparse.ArgumentParser(description='GRAIL-Heart Ablation Study')
    parser.add_argument('--config', type=str, default='configs/ablation.yaml',
                       help='Path to config file')
    parser.add_argument('--ablation', type=str, default='all',
                       help='Ablation type or "all"')
    parser.add_argument('--output_dir', type=str, default='outputs/ablation',
                       help='Output directory')
    args = parser.parse_args()
    
    # Load config (fall back to default if ablation config doesn't exist)
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path('configs/default.yaml')
    config = load_config(str(config_path))
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    data_dir = Path(config['paths']['data_dir'])
    
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Clear any existing CUDA memory
    clear_cuda_memory()
    
    # Load L-R database first
    cache_path = data_dir / 'lr_database_cache.csv'
    lr_pairs_df = get_omnipath_lr_database(cache_path=cache_path)
    print(f"Loaded {len(lr_pairs_df)} L-R pairs")
    
    # Load data with L-R labeling
    train_graphs, val_graphs, n_cell_types = load_data(config, data_dir, lr_pairs_df)
    
    if not train_graphs:
        print("No data loaded! Check data path.")
        return
        
    # Determine ablation types
    if args.ablation == 'all':
        ablation_types = list(AblationExperiment.ABLATION_CONFIGS.keys())
    else:
        ablation_types = [args.ablation]
    
    # Run ablation study
    experiment = AblationExperiment(
        base_config=config,
        output_dir=output_dir,
        device=device,
    )
    
    try:
        results_df = experiment.run_ablation_suite(
            ablation_types=ablation_types,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            lr_pairs_df=lr_pairs_df,
            n_cell_types=n_cell_types,
        )
    except RuntimeError as e:
        if "CUDA_UNRECOVERABLE" in str(e):
            print("\nAblation study terminated early due to CUDA OOM.")
            print("Partial results have been saved.")
            return
        raise
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    for ablation_type in ablation_types:
        subset = results_df[results_df['ablation_type'] == ablation_type]
        if len(subset) == 0:
            continue
            
        print(f"\n{ablation_type}:")
        print("-" * 40)
        
        for _, row in subset.iterrows():
            r2 = row.get('recon_r2', 0)
            auroc = row.get('lr_auroc', 0)
            acc = row.get('accuracy', 0)
            params = row.get('n_params', 0)
            
            r2_str = f"{r2:.3f}" if not pd.isna(r2) else "N/A"
            auroc_str = f"{auroc:.3f}" if not pd.isna(auroc) else "N/A"
            acc_str = f"{acc:.3f}" if not pd.isna(acc) else "N/A"
            
            print(f"  {row['ablation_setting']:20s} | "
                  f"R²={r2_str} | "
                  f"AUROC={auroc_str} | "
                  f"Acc={acc_str} | "
                  f"Params={params:,}")


if __name__ == '__main__':
    main()
