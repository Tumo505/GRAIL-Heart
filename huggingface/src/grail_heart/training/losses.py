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
    - combined: MSE + cosine + correlation
    
    Args:
        loss_type: Type of reconstruction loss
        reduction: Reduction method ('mean', 'sum', 'none')
        cosine_weight: Weight for cosine loss (combined mode)
        correlation_weight: Weight for correlation loss (combined mode)
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean',
        cosine_weight: float = 0.3,
        correlation_weight: float = 0.3,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.cosine_weight = cosine_weight
        self.correlation_weight = correlation_weight
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        pi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred: Predicted expression [N, G]
            target: Ground truth expression [N, G]
            mask: Optional mask for gene subset [N, G] or [G]
            theta: Dispersion parameter for NB/ZINB [N, G] or [G]
            pi: Dropout probability for ZINB [N, G]
            
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
        elif self.loss_type == 'zinb':
            # Zero-inflated negative binomial
            if theta is None or pi is None:
                raise ValueError("ZINB loss requires theta and pi parameters")
            loss = self._zinb_loss(pred, theta, pi, target)
        elif self.loss_type == 'combined':
            # MSE + Cosine + Correlation
            loss = self._combined_loss(pred, target)
            return loss  # Already reduced
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
    
    def _zinb_loss(
        self,
        mean: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ZINB negative log-likelihood."""
        eps = 1e-8
        theta = theta + eps
        mean = mean + eps
        
        # NB log prob
        nb_case = (
            torch.lgamma(target + theta)
            - torch.lgamma(theta)
            - torch.lgamma(target + 1)
            + theta * torch.log(theta / (theta + mean))
            + target * torch.log(mean / (theta + mean))
        )
        
        # Zero case: P(x=0) = π + (1-π) * NB(0)
        zero_nb = theta * torch.log(theta / (theta + mean))
        zero_case = torch.log(pi + (1 - pi) * torch.exp(zero_nb) + eps)
        
        # Non-zero case: P(x>0) = (1-π) * NB(x)
        non_zero_case = torch.log(1 - pi + eps) + nb_case
        
        # Select based on target
        nll = -torch.where(target < 0.5, zero_case, non_zero_case)
        
        return nll
    
    def _combined_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Combined MSE + cosine + correlation loss."""
        # MSE component
        mse = F.mse_loss(pred, target)
        
        # Cosine similarity
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cosine = (1 - (pred_norm * target_norm).sum(dim=-1)).mean()
        
        # Per-cell correlation
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        pred_std = pred_centered.std(dim=-1, keepdim=True) + 1e-8
        target_std = target_centered.std(dim=-1, keepdim=True) + 1e-8
        correlation = ((pred_centered / pred_std) * (target_centered / target_std)).mean(dim=-1)
        corr_loss = (1 - correlation).mean()
        
        return mse + self.cosine_weight * cosine + self.correlation_weight * corr_loss
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
    - Contrastive loss (for better embeddings)
    - Inverse modelling losses (NEW): fate prediction, causal consistency
    
    Args:
        lr_weight: Weight for L-R loss
        recon_weight: Weight for reconstruction loss
        cell_type_weight: Weight for cell type loss
        signaling_weight: Weight for signaling loss
        kl_weight: Weight for KL divergence
        contrastive_weight: Weight for contrastive loss
        recon_loss_type: Type of reconstruction loss ('mse', 'combined', 'zinb')
        use_inverse_losses: Whether to use inverse modelling losses (NEW)
        fate_weight: Weight for fate prediction loss (NEW)
        causal_weight: Weight for causal consistency loss (NEW)
    """
    
    def __init__(
        self,
        lr_weight: float = 1.0,
        recon_weight: float = 0.5,
        cell_type_weight: float = 1.0,
        signaling_weight: float = 0.1,
        kl_weight: float = 0.001,
        contrastive_weight: float = 0.5,
        n_cell_types: Optional[int] = None,
        use_contrastive: bool = True,
        recon_loss_type: str = 'combined',
        # NEW: Inverse modelling loss parameters
        use_inverse_losses: bool = True,
        fate_weight: float = 0.5,
        causal_weight: float = 0.3,
        differentiation_weight: float = 0.2,
        gene_target_weight: float = 0.3,
        cycle_weight: float = 0.3,
        pathway_grounding_weight: float = 0.1,
    ):
        super().__init__()
        
        self.lr_weight = lr_weight
        self.recon_weight = recon_weight
        self.cell_type_weight = cell_type_weight
        self.signaling_weight = signaling_weight
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        self.use_contrastive = use_contrastive
        self.recon_loss_type = recon_loss_type
        
        # NEW: Inverse modelling loss weights
        self.use_inverse_losses = use_inverse_losses
        self.fate_weight = fate_weight
        self.causal_weight = causal_weight
        self.differentiation_weight = differentiation_weight
        self.gene_target_weight = gene_target_weight

        self.cycle_weight = cycle_weight
        self.pathway_grounding_weight = pathway_grounding_weight
        
        self.lr_loss = LRInteractionLoss(pos_weight=2.0)
        self.recon_loss = ReconstructionLoss(
            loss_type=recon_loss_type,
            cosine_weight=0.3,
            correlation_weight=0.3,
        )
        self.signaling_loss = SignalingNetworkLoss()
        
        if n_cell_types is not None:
            self.cell_type_loss = CellTypeLoss(n_cell_types)
        else:
            self.cell_type_loss = None
        
        # Contrastive loss
        if use_contrastive:
            from .contrastive import ContrastiveLoss
            self.contrastive_loss = ContrastiveLoss(
                temperature=0.07,
                use_spatial=True,
                spatial_weight=0.5,
                supervised_weight=1.0,
            )
        else:
            self.contrastive_loss = None
            
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss (forward + inverse modelling).
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth dictionary
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # ===== FORWARD MODELLING LOSSES =====
        
        # L-R interaction loss
        if 'lr_scores' in outputs and 'lr_targets' in targets:
            lr_loss = self.lr_loss(outputs['lr_scores'], targets['lr_targets'])
            loss_dict['lr_loss'] = lr_loss
            total_loss = total_loss + self.lr_weight * lr_loss
            
        # Reconstruction loss
        if 'reconstruction' in outputs and 'expression' in targets:
            # Handle ZINB decoder outputs (dict with mean, theta, dropout)
            recon_output = outputs['reconstruction']
            if isinstance(recon_output, dict) and 'mean' in recon_output:
                # ZINB decoder output
                recon_loss = self.recon_loss(
                    recon_output,
                    targets['expression'],
                    theta=recon_output.get('theta'),
                    pi=recon_output.get('dropout'),
                )
            else:
                # Standard decoder output
                recon_loss = self.recon_loss(recon_output, targets['expression'])
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
        
        # Contrastive loss
        if self.use_contrastive and self.contrastive_loss is not None:
            embeddings = outputs.get('node_embeddings', outputs.get('cell_embeddings', None))
            if embeddings is not None:
                cell_types = targets.get('cell_types', None)
                edge_index = targets.get('edge_index', None)
                
                contra_loss, contra_dict = self.contrastive_loss(
                    embeddings=embeddings,
                    labels=cell_types,
                    edge_index=edge_index,
                )
                loss_dict['contrastive_loss'] = contra_loss
                total_loss = total_loss + self.contrastive_weight * contra_loss


        if self.use_inverse_losses:
            # Cell fate prediction loss
            #    If target is soft (neighbourhood composition [N, C]) → KL divergence
            #    If target is hard (class indices [N]) → cross-entropy (fallback)
            if 'fate_logits' in outputs and 'cell_fate' in targets:
                fate_target = targets['cell_fate']
                fate_logits = outputs['fate_logits']
                if fate_target.dim() == 2:
                    # Align dimensions: different graphs may have different
                    # numbers of cell types in their neighbourhood composition
                    n_pred = fate_logits.size(1)
                    n_tgt = fate_target.size(1)
                    if n_pred > n_tgt:
                        fate_target = F.pad(fate_target, (0, n_pred - n_tgt))
                    elif n_tgt > n_pred:
                        fate_logits = F.pad(fate_logits, (0, n_tgt - n_pred),
                                            value=-1e9)
                    # Soft labels → KL(target || softmax(logits))
                    log_probs = F.log_softmax(fate_logits, dim=-1)
                    # Clamp target for numerical safety
                    fate_target_safe = fate_target.clamp(min=1e-8)
                    # Re-normalise after padding so it sums to 1
                    fate_target_safe = fate_target_safe / fate_target_safe.sum(
                        dim=-1, keepdim=True)
                    fate_loss = F.kl_div(log_probs, fate_target_safe, reduction='batchmean',
                                         log_target=False)
                else:
                    # Hard labels → CE (backward compat)
                    fate_loss = F.cross_entropy(fate_logits, fate_target)
                loss_dict['fate_loss'] = fate_loss
                total_loss = total_loss + self.fate_weight * fate_loss

            # Differentiation score — pairwise ranking loss
            #    Instead of MSE to a pseudo-uniform target, enforce *ordinal
            #    consistency*: if cell i is more differentiated than cell j,
            #    the model's predicted score should reflect that.
            if 'differentiation_score' in outputs and 'differentiation_stage' in targets:
                pred_diff = outputs['differentiation_score'].squeeze()
                true_diff = targets['differentiation_stage']
                # Margin-based ranking loss on random pairs
                n_pairs = min(512, pred_diff.size(0) // 2)
                if n_pairs > 0:
                    idx = torch.randperm(pred_diff.size(0), device=pred_diff.device)[:2 * n_pairs]
                    i_idx, j_idx = idx[:n_pairs], idx[n_pairs:]
                    diff_ij = (true_diff[i_idx] - true_diff[j_idx]).sign()  # +1, 0, -1
                    pred_ij = pred_diff[i_idx] - pred_diff[j_idx]
                    # Margin ranking: want pred_ij * diff_ij > margin
                    rank_loss = F.margin_ranking_loss(
                        pred_diff[i_idx], pred_diff[j_idx],
                        diff_ij, margin=0.1,
                    )
                else:
                    rank_loss = F.mse_loss(pred_diff, true_diff)
                loss_dict['differentiation_loss'] = rank_loss
                total_loss = total_loss + self.differentiation_weight * rank_loss

            # Causal sparsity regularization
            if 'causal_lr_scores' in outputs:
                causal_sparsity = outputs['causal_lr_scores'].mean()
                loss_dict['causal_sparsity'] = causal_sparsity
                total_loss = total_loss + self.causal_weight * causal_sparsity

            # Target gene prediction loss
            if 'predicted_expression_from_lr' in outputs and 'expression' in targets:
                src, dst = targets.get('edge_index', (None, None))
                if dst is not None:
                    target_expression = targets['expression'][dst]
                    gene_target_loss = F.mse_loss(
                        outputs['predicted_expression_from_lr'],
                        target_expression,
                    )
                    loss_dict['gene_target_loss'] = gene_target_loss
                    total_loss = total_loss + self.gene_target_weight * gene_target_loss

            # Cycle-consistency loss
            #    Reconstructed LR ≈ original sigmoid(lr_scores)
            if 'cycle_reconstructed_lr' in outputs and 'lr_scores' in outputs:
                lr_target = torch.sigmoid(outputs['lr_scores']).detach()
                cycle_recon = outputs['cycle_reconstructed_lr']
                # Align lengths (cycle may only cover a subset of edges)
                min_len = min(cycle_recon.size(0), lr_target.size(0))
                cycle_loss = F.mse_loss(cycle_recon[:min_len], lr_target[:min_len])
                loss_dict['cycle_loss'] = cycle_loss
                total_loss = total_loss + self.cycle_weight * cycle_loss

            # 6. Pathway grounding loss
            #    From LRToTargetGeneDecoder.pathway_grounding_loss()
            if 'pathway_grounding_loss' in outputs:
                pg_loss = outputs['pathway_grounding_loss']
                loss_dict['pathway_grounding_loss'] = pg_loss
                total_loss = total_loss + self.pathway_grounding_weight * pg_loss
            
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
