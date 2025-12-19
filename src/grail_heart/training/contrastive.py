"""
Contrastive Learning Module for GRAIL-Heart

Implements:
- InfoNCE loss for learning discriminative embeddings
- Supervised contrastive loss using cell type labels
- Spatial contrastive loss for spatially coherent embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Combined contrastive loss for spatial transcriptomics.
    
    Combines:
    1. Supervised contrastive loss (using cell type labels)
    2. Spatial contrastive loss (nearby cells should be similar)
    3. InfoNCE loss for self-supervised learning
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        use_spatial: bool = True,
        spatial_weight: float = 0.5,
        supervised_weight: float = 1.0,
    ):
        """
        Args:
            temperature: Temperature for softmax scaling
            base_temperature: Base temperature for loss normalization
            use_spatial: Whether to use spatial contrastive loss
            spatial_weight: Weight for spatial contrastive loss
            supervised_weight: Weight for supervised contrastive loss
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.use_spatial = use_spatial
        self.spatial_weight = spatial_weight
        self.supervised_weight = supervised_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Node embeddings [N, D]
            labels: Cell type labels [N] (optional)
            edge_index: Spatial graph edges [2, E] (optional)
            mask: Mask for valid nodes [N] (optional)
            
        Returns:
            Total loss and dictionary of individual losses
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # Supervised contrastive loss
        if labels is not None and self.supervised_weight > 0:
            sup_loss = self._supervised_contrastive_loss(embeddings, labels, mask)
            losses['supervised_contrastive'] = sup_loss.item()
            total_loss = total_loss + self.supervised_weight * sup_loss
        
        # Spatial contrastive loss
        if edge_index is not None and self.use_spatial and self.spatial_weight > 0:
            spatial_loss = self._spatial_contrastive_loss(embeddings, edge_index)
            losses['spatial_contrastive'] = spatial_loss.item()
            total_loss = total_loss + self.spatial_weight * spatial_loss
        
        losses['total_contrastive'] = total_loss.item()
        
        return total_loss, losses
    
    def _supervised_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (SupCon).
        
        Pulls together embeddings of the same cell type,
        pushes apart embeddings of different cell types.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # For large batches, subsample to avoid memory issues
        max_samples = 2048
        if batch_size > max_samples:
            idx = torch.randperm(batch_size)[:max_samples]
            embeddings = embeddings[idx]
            labels = labels[idx]
            batch_size = max_samples
        
        # Create label mask: same label = 1, different = 0
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(label_mask) - torch.eye(batch_size, device=device)
        label_mask = label_mask * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Compute log-sum-exp of all negatives
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero
        pos_count = label_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        
        mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / pos_count
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss
    
    def _spatial_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Spatial contrastive loss.
        
        Encourages spatially adjacent cells to have similar embeddings.
        """
        device = embeddings.device
        
        # Get source and target embeddings for edges
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        
        # Subsample edges if too many
        max_edges = 50000
        if src_idx.shape[0] > max_edges:
            idx = torch.randperm(src_idx.shape[0])[:max_edges]
            src_idx = src_idx[idx]
            tgt_idx = tgt_idx[idx]
        
        src_emb = embeddings[src_idx]
        tgt_emb = embeddings[tgt_idx]
        
        # Positive pairs: spatially adjacent cells
        pos_sim = (src_emb * tgt_emb).sum(dim=1) / self.temperature
        
        # Negative pairs: random cells (not spatially adjacent)
        n_edges = src_idx.shape[0]
        neg_idx = torch.randint(0, embeddings.shape[0], (n_edges,), device=device)
        neg_emb = embeddings[neg_idx]
        neg_sim = (src_emb * neg_emb).sum(dim=1) / self.temperature
        
        # InfoNCE-style loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(n_edges, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps embeddings to a lower-dimensional space for contrastive loss.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class AugmentationModule(nn.Module):
    """
    Data augmentation for contrastive learning.
    
    Creates augmented views of gene expression data.
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        noise_std: float = 0.1,
        mask_rate: float = 0.15,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.mask_rate = mask_rate
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply random augmentations to gene expression.
        
        Args:
            x: Gene expression [N, G]
            training: Whether in training mode
            
        Returns:
            Augmented expression
        """
        if not training:
            return x
        
        # Random gene dropout
        if self.dropout_rate > 0:
            mask = torch.bernoulli(
                torch.ones_like(x) * (1 - self.dropout_rate)
            )
            x = x * mask
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Random gene masking (set to zero)
        if self.mask_rate > 0:
            mask = torch.bernoulli(
                torch.ones_like(x) * (1 - self.mask_rate)
            )
            x = x * mask
        
        return x


class SimCLRLoss(nn.Module):
    """
    SimCLR-style contrastive loss for self-supervised learning.
    
    Creates two augmented views and maximizes agreement between them.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.
        
        Args:
            z1: Embeddings of first view [N, D]
            z2: Embeddings of second view [N, D]
            
        Returns:
            NT-Xent loss
        """
        device = z1.device
        batch_size = z1.shape[0]
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        
        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]
        
        # Create labels: positive pairs are (i, i+N) and (i+N, i)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2N]
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss
