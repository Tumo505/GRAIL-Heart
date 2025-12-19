"""
Loss Functions for GRAIL-Heart Training

Multi-task losses for L-R prediction, expression reconstruction,
cell type classification, and signaling network inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LRInteractionLoss(nn.Module):
    """
    Loss for ligand-receptor interaction prediction.
    
    Uses binary cross-entropy with optional class weighting
    to handle imbalanced positive/negative edges.
    
    Args:
        pos_weight: Weight for positive class (interactions)
        focal_gamma: Focal loss gamma (0 = standard BCE)
    """
    
    def __init__(
        self,
        pos_weight: float = 2.0,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        
    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L-R prediction loss.
        
        Args:
            scores: Predicted interaction scores [E]
            targets: Ground truth labels [E]
            
        Returns:
            Loss value
        """
        probs = torch.sigmoid(scores)
        
        if self.focal_gamma > 0:
            # Focal loss
            ce_loss = F.binary_cross_entropy_with_logits(
                scores, targets, reduction='none'
            )
            pt = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = focal_weight * ce_loss
            
            # Apply pos_weight
            weight = torch.where(targets == 1, self.pos_weight, 1.0)
            loss = loss * weight
            
            return loss.mean()
        else:
            # Standard weighted BCE
            pos_weight = torch.tensor([self.pos_weight], device=scores.device)
            return F.binary_cross_entropy_with_logits(
                scores, targets, pos_weight=pos_weight
            )


class ReconstructionLoss(nn.Module):
    """
    Loss for gene expression reconstruction.
    
    Supports multiple distribution assumptions for count data:
    - MSE: Mean squared error
    - Poisson: Poisson negative log-likelihood
    - NB: Negative binomial
    - ZINB: Zero-inflated negative binomial
    
    Args:
        loss_type: Type of reconstruction loss
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean',
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred: Predicted expression [N, G]
            target: Ground truth expression [N, G]
            mask: Optional mask for gene subset [N, G] or [G]
            
        Returns:
            Loss value
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'mae':
            loss = F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'poisson':
            # Poisson NLL: pred - target * log(pred + eps)
            eps = 1e-8
            loss = pred - target * torch.log(pred + eps)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        if mask is not None:
            loss = loss * mask
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CellTypeLoss(nn.Module):
    """
    Loss for cell type classification.
    
    Args:
        n_classes: Number of cell type classes
        label_smoothing: Label smoothing factor
        class_weights: Optional class weights for imbalanced data
    """
    
    def __init__(
        self,
        n_classes: int,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.register_buffer('class_weights', class_weights)
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: Predicted logits [N, n_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Loss value
        """
        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class SignalingNetworkLoss(nn.Module):
    """
    Loss for signaling network inference.
    
    Encourages sparse, biologically plausible networks
    with spatial locality constraints.
    
    Args:
        sparsity_weight: Weight for sparsity regularization
        spatial_decay_weight: Weight for spatial decay constraint
    """
    
    def __init__(
        self,
        sparsity_weight: float = 0.01,
        spatial_decay_weight: float = 0.1,
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.spatial_decay_weight = spatial_decay_weight
        
    def forward(
        self,
        adjacency: torch.Tensor,
        spatial_dist: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute signaling network losses.
        
        Args:
            adjacency: Predicted adjacency [N, N]
            spatial_dist: Spatial distance matrix [N, N]
            reference: Optional reference network [N, N]
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Sparsity loss (L1 on adjacency)
        losses['sparsity'] = self.sparsity_weight * adjacency.abs().mean()
        
        # Spatial decay constraint
        if spatial_dist is not None:
            # Penalize strong connections between distant cells
            decay_target = torch.exp(-spatial_dist)
            spatial_loss = F.mse_loss(adjacency, adjacency * decay_target)
            losses['spatial_decay'] = self.spatial_decay_weight * spatial_loss
            
        # Supervised loss if reference available
        if reference is not None:
            losses['reference'] = F.mse_loss(adjacency, reference)
            
        return losses


class GRAILHeartLoss(nn.Module):
    """
    Combined multi-task loss for GRAIL-Heart.
    
    Weighted combination of:
    - L-R interaction prediction loss
    - Expression reconstruction loss
    - Cell type classification loss
    - Signaling network regularization
    - KL divergence (for variational model)
    
    Args:
        lr_weight: Weight for L-R loss
        recon_weight: Weight for reconstruction loss
        cell_type_weight: Weight for cell type loss
        signaling_weight: Weight for signaling loss
        kl_weight: Weight for KL divergence
    """
    
    def __init__(
        self,
        lr_weight: float = 1.0,
        recon_weight: float = 0.5,
        cell_type_weight: float = 1.0,
        signaling_weight: float = 0.1,
        kl_weight: float = 0.001,
        n_cell_types: Optional[int] = None,
    ):
        super().__init__()
        
        self.lr_weight = lr_weight
        self.recon_weight = recon_weight
        self.cell_type_weight = cell_type_weight
        self.signaling_weight = signaling_weight
        self.kl_weight = kl_weight
        
        self.lr_loss = LRInteractionLoss(pos_weight=2.0)
        self.recon_loss = ReconstructionLoss(loss_type='mse')
        self.signaling_loss = SignalingNetworkLoss()
        
        if n_cell_types is not None:
            self.cell_type_loss = CellTypeLoss(n_cell_types)
        else:
            self.cell_type_loss = None
            
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth dictionary
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L-R interaction loss
        if 'lr_scores' in outputs and 'lr_targets' in targets:
            lr_loss = self.lr_loss(outputs['lr_scores'], targets['lr_targets'])
            loss_dict['lr_loss'] = lr_loss
            total_loss = total_loss + self.lr_weight * lr_loss
            
        # Reconstruction loss
        if 'reconstruction' in outputs and 'expression' in targets:
            recon_loss = self.recon_loss(outputs['reconstruction'], targets['expression'])
            loss_dict['recon_loss'] = recon_loss
            total_loss = total_loss + self.recon_weight * recon_loss
            
        # Cell type classification loss
        if 'cell_type_logits' in outputs and 'cell_types' in targets:
            if self.cell_type_loss is not None:
                ct_loss = self.cell_type_loss(outputs['cell_type_logits'], targets['cell_types'])
                loss_dict['cell_type_loss'] = ct_loss
                total_loss = total_loss + self.cell_type_weight * ct_loss
                
        # Signaling network loss
        if 'signaling' in outputs:
            spatial_dist = targets.get('spatial_dist', None)
            sig_losses = self.signaling_loss(
                outputs['signaling']['adjacency'],
                spatial_dist=spatial_dist,
            )
            for name, loss in sig_losses.items():
                loss_dict[f'signaling_{name}'] = loss
                total_loss = total_loss + self.signaling_weight * loss
                
        # KL divergence (variational)
        if 'kl_loss' in outputs:
            loss_dict['kl_loss'] = outputs['kl_loss']
            total_loss = total_loss + self.kl_weight * outputs['kl_loss']
            
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


def create_lr_targets(
    edge_index: torch.Tensor,
    lr_edge_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Create L-R interaction targets from edge mask.
    
    Args:
        edge_index: Graph edges [2, E]
        lr_edge_mask: Boolean mask for L-R edges [E]
        
    Returns:
        Target labels [E]
    """
    return lr_edge_mask.float()


def create_negative_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_per_pos: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create negative edge samples for contrastive learning.
    
    Args:
        edge_index: Positive edges [2, E]
        num_nodes: Total number of nodes
        num_neg_per_pos: Number of negative samples per positive
        
    Returns:
        Combined edge_index with negatives
        Labels (1 for positive, 0 for negative)
    """
    num_pos = edge_index.size(1)
    num_neg = num_pos * num_neg_per_pos
    
    # Sample random negative edges
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
    neg_edges = torch.stack([neg_src, neg_dst], dim=0)
    
    # Combine
    combined_edges = torch.cat([edge_index, neg_edges], dim=1)
    labels = torch.cat([
        torch.ones(num_pos, device=edge_index.device),
        torch.zeros(num_neg, device=edge_index.device),
    ])
    
    return combined_edges, labels
