"""
Trainer for GRAIL-Heart

Handles model training, validation, checkpointing, and logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from datetime import datetime

from .losses import GRAILHeartLoss, create_negative_samples
from .metrics import (
    compute_lr_metrics,
    compute_reconstruction_metrics,
    compute_cell_type_metrics,
    compute_signaling_metrics,
    MetricTracker,
    print_metrics,
)


class GRAILHeartTrainer:
    """
    Trainer for GRAIL-Heart models.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation and evaluation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Logging to TensorBoard and/or WandB
    
    Args:
        model: GRAIL-Heart model instance
        optimizer: Optimizer instance
        loss_fn: Loss function instance
        device: Device to train on
        output_dir: Directory for checkpoints and logs
        scheduler: Optional learning rate scheduler
        grad_clip: Gradient clipping value
        grad_accum_steps: Gradient accumulation steps
        mixed_precision: Whether to use AMP
        log_interval: Batches between logging
        save_interval: Epochs between checkpoint saves
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        output_dir: Union[str, Path],
        scheduler: Optional[object] = None,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        mixed_precision: bool = True,
        log_interval: int = 10,
        save_interval: int = 5,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Check if model has inverse modelling enabled
        self.use_inverse_modelling = getattr(model, 'use_inverse_modelling', False)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        
        # Metric trackers - include inverse modelling metrics if enabled
        train_metric_names = ['total_loss', 'lr_loss', 'recon_loss', 'cell_type_loss']
        val_metric_names = ['total_loss', 'lr_loss', 'recon_loss', 'auroc', 'auprc', 'pearson_mean']
        
        if self.use_inverse_modelling:
            train_metric_names.extend(['fate_loss', 'causal_sparsity'])
            val_metric_names.extend(['fate_accuracy'])
            
        self.train_metrics = MetricTracker(
            train_metric_names,
            prefix='train'
        )
        self.val_metrics = MetricTracker(
            val_metric_names,
            prefix='val'
        )
        
        # Logging
        self.logger = None
        self._init_logging()
        
    def _init_logging(self):
        """Initialize logging backends."""
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            self.tb_writer = None
            
        # WandB (optional)
        self.wandb_run = None
        
    def init_wandb(self, project: str, config: Dict):
        """Initialize WandB logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(project=project, config=config)
        except ImportError:
            print("WandB not available")
            
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass - enable inverse modelling if available
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                outputs = self.model(data, run_inverse=self.use_inverse_modelling)
                
                # Prepare targets
                targets = self._prepare_targets(data, outputs)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(outputs, targets)
                loss = loss / self.grad_accum_steps
                
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    
                # Gradient clipping
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                self.global_step += 1
                
            # Update metrics
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            metrics_update = {
                'total_loss': loss_dict['total_loss'].item(),
                'lr_loss': loss_dict.get('lr_loss', torch.tensor(0)).item(),
                'recon_loss': loss_dict.get('recon_loss', torch.tensor(0)).item(),
                'cell_type_loss': loss_dict.get('cell_type_loss', torch.tensor(0)).item(),
                'contrastive_loss': loss_dict.get('contrastive_loss', torch.tensor(0)).item(),
            }
            
            # Add inverse modelling metrics if available
            if self.use_inverse_modelling:
                metrics_update['fate_loss'] = loss_dict.get('fate_loss', torch.tensor(0)).item()
                metrics_update['causal_sparsity'] = loss_dict.get('causal_sparsity', torch.tensor(0)).item()
                
            self.train_metrics.update(metrics_update, count=batch_size)
            
            # Logging
            if batch_idx % self.log_interval == 0:
                self._log_step(loss_dict, batch_idx, len(train_loader))
                
        # Epoch metrics
        epoch_metrics = self.train_metrics.compute()
        epoch_time = time.time() - epoch_start
        epoch_metrics['epoch_time'] = epoch_time
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        all_scores = []
        all_targets = []
        all_recon_pred = []
        all_recon_target = []
        
        for data in val_loader:
            data = data.to(self.device)
            
            outputs = self.model(data, run_inverse=self.use_inverse_modelling)
            targets = self._prepare_targets(data, outputs)
            
            loss, loss_dict = self.loss_fn(outputs, targets)
            
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            self.val_metrics.update({
                'total_loss': loss_dict['total_loss'].item(),
                'lr_loss': loss_dict.get('lr_loss', torch.tensor(0)).item(),
                'recon_loss': loss_dict.get('recon_loss', torch.tensor(0)).item(),
                'contrastive_loss': loss_dict.get('contrastive_loss', torch.tensor(0)).item(),
            }, count=batch_size)
            
            # Collect predictions for metric computation
            if 'lr_scores' in outputs:
                all_scores.append(outputs['lr_scores'].cpu())
                all_targets.append(targets['lr_targets'].cpu())
                
            if 'reconstruction' in outputs:
                all_recon_pred.append(outputs['reconstruction'].cpu())
                all_recon_target.append(targets['expression'].cpu())
                
        # Compute detailed metrics
        val_metrics = self.val_metrics.compute()
        
        if all_scores:
            scores = torch.cat(all_scores)
            targets = torch.cat(all_targets)
            lr_metrics = compute_lr_metrics(scores, targets)
            val_metrics.update({f'val_{k}': v for k, v in lr_metrics.items()})
            
        if all_recon_pred:
            pred = torch.cat(all_recon_pred)
            target = torch.cat(all_recon_target)
            recon_metrics = compute_reconstruction_metrics(pred, target)
            val_metrics.update({f'val_{k}': v for k, v in recon_metrics.items()})
            
        return val_metrics
    
    def _prepare_targets(
        self,
        data: Data,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Prepare target tensors for loss computation."""
        targets = {}
        
        # Expression reconstruction target
        targets['expression'] = data.x
        
        # Edge index for contrastive learning
        if hasattr(data, 'edge_index'):
            targets['edge_index'] = data.edge_index
        
        # L-R interaction targets
        if hasattr(data, 'edge_type'):
            # L-R edges have type 1
            targets['lr_targets'] = (data.edge_type == 1).float()
        else:
            # Create synthetic targets (all edges positive, sample negatives)
            edge_index, labels = create_negative_samples(
                data.edge_index, data.num_nodes, num_neg_per_pos=1
            )
            targets['lr_targets'] = labels
            
        # Cell type targets (also used as cell fate target for inverse modelling)
        if hasattr(data, 'y') and data.y is not None:
            targets['cell_types'] = data.y
            # Prefer soft neighbourhood composition as fate target (WP3)
            if hasattr(data, 'neighborhood_composition') and data.neighborhood_composition is not None:
                targets['cell_fate'] = data.neighborhood_composition  # [N, C] soft
            else:
                targets['cell_fate'] = data.y  # fallback: hard labels
            
        # Differentiation stage targets (for inverse modelling)
        if hasattr(data, 'differentiation_stage') and data.differentiation_stage is not None:
            targets['differentiation_stage'] = data.differentiation_stage
            
        # Spatial distances
        if hasattr(data, 'pos') and data.pos is not None:
            targets['spatial_dist'] = torch.cdist(data.pos, data.pos)
            
        return targets
    
    def _log_step(self, loss_dict: Dict, batch_idx: int, total_batches: int):
        """Log training step."""
        msg = f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}]"
        msg += f" Loss: {loss_dict['total_loss'].item():.4f}"
        
        if 'lr_loss' in loss_dict:
            msg += f" LR: {loss_dict['lr_loss'].item():.4f}"
        if 'recon_loss' in loss_dict:
            msg += f" Recon: {loss_dict['recon_loss'].item():.4f}"
            
        print(msg)
        
        # TensorBoard
        if self.tb_writer:
            for name, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.tb_writer.add_scalar(f'train/{name}', value, self.global_step)
                
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        print(f"\n=== Epoch {self.current_epoch} Summary ===")
        print("Training:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("Validation:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        # TensorBoard
        if self.tb_writer:
            for name, value in train_metrics.items():
                self.tb_writer.add_scalar(f'epoch/{name}', value, self.current_epoch)
            for name, value in val_metrics.items():
                self.tb_writer.add_scalar(f'epoch/{name}', value, self.current_epoch)
                
        # WandB
        if self.wandb_run:
            import wandb
            wandb.log({**train_metrics, **val_metrics, 'epoch': self.current_epoch})
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        patience: int = 10,
        monitor: str = 'val_total_loss',
        mode: str = 'min',
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            patience: Early stopping patience
            monitor: Metric to monitor for early stopping
            mode: 'min' or 'max' for monitoring
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
        
        best_value = float('inf') if mode == 'min' else float('-inf')
        patience_counter = 0
        
        print(f"\nStarting training for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(monitor, val_metrics['val_total_loss']))
                else:
                    self.scheduler.step()
                    
            # Log
            self._log_epoch(train_metrics, val_metrics)
            
            # History
            history['train_loss'].append(train_metrics['train_total_loss'])
            history['val_loss'].append(val_metrics['val_total_loss'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check improvement
            current_value = val_metrics.get(monitor, val_metrics['val_total_loss'])
            
            improved = (mode == 'min' and current_value < best_value) or \
                      (mode == 'max' and current_value > best_value)
                      
            if improved:
                best_value = current_value
                patience_counter = 0
                self.save_checkpoint('best.pt', val_metrics)
                print(f"  New best {monitor}: {best_value:.4f}")
            else:
                patience_counter += 1
                
            # Save periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)
                
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
                
        # Final save
        self.save_checkpoint('final.pt', val_metrics)
        
        # Close logging
        if self.tb_writer:
            self.tb_writer.close()
            
        return history
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        path = self.output_dir / 'checkpoints' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: Union[str, Path], map_location: Optional[str] = None):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            map_location: Device to load checkpoint to (e.g., 'cpu'). 
                         If None, uses self.device.
        """
        load_device = map_location if map_location else self.device
        # weights_only=False is safe here since we generated these checkpoints ourselves
        checkpoint = torch.load(path, map_location=load_device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> optim.Optimizer:
    """Create optimizer."""
    
    # Separate parameters with/without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    if optimizer_name == 'adamw':
        return optim.AdamW(param_groups, lr=lr, **kwargs)
    elif optimizer_name == 'adam':
        return optim.Adam(param_groups, lr=lr, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(param_groups, lr=lr, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    n_epochs: int = 100,
    warmup_epochs: int = 5,
    **kwargs,
):
    """Create learning rate scheduler."""
    
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=n_epochs, **kwargs)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, **kwargs)
    elif scheduler_name == 'onecycle':
        # Requires steps_per_epoch in kwargs
        return OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], 
                         epochs=n_epochs, **kwargs)
    else:
        return None
