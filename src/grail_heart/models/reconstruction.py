"""
Advanced Gene Expression Reconstruction Module for GRAIL-Heart

Implements improved decoders for gene expression reconstruction:
- Variational decoder with reparameterization
- Negative Binomial and ZINB distributions for count data
- Gene-specific dispersion parameters
- Skip connections and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.block(x))


class GeneAttention(nn.Module):
    """
    Gene-specific attention for expression reconstruction.
    
    Learns to weight different parts of the embedding for each gene.
    """
    
    def __init__(self, hidden_dim: int, n_genes: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.gene_queries = nn.Parameter(torch.randn(n_genes, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_queries)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Cell embeddings [N, D]
        Returns:
            Gene expression [N, G]
        """
        batch_size = z.size(0)
        n_genes = self.gene_queries.size(0)
        
        # Project
        keys = self.key_proj(z)  # [N, D]
        values = self.value_proj(z)  # [N, D]
        
        # Attention scores: [N, G]
        scores = torch.matmul(self.gene_queries, keys.T).T / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        # Weighted sum
        out = attn * values.unsqueeze(1).expand(-1, n_genes, -1).sum(dim=-1)
        
        return out


class ResidualGeneDecoder(nn.Module):
    """
    Gene expression decoder with skip connections from input.
    
    Key insight: Instead of pure autoencoding (compress then expand),
    this decoder learns a RESIDUAL correction to the input expression.
    This is much easier to learn and generalizes better.
    
    output = input + learned_correction(embedding)
    
    Args:
        hidden_dim: Input embedding dimension  
        n_genes: Number of genes
        decoder_dims: Hidden dimensions
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        decoder_dims: List[int] = [512, 512],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        
        # Embedding to correction
        layers = []
        in_dim = hidden_dim
        for dim in decoder_dims:
            layers.extend([
                nn.Linear(in_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = dim
            
        # Output correction (can be positive or negative)
        layers.append(nn.Linear(in_dim, n_genes))
        self.correction_net = nn.Sequential(*layers)
        
        # Learnable gate: how much to use original vs corrected
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, n_genes),
            nn.Sigmoid(),
        )
        
        # Gene-specific scale for the correction
        self.correction_scale = nn.Parameter(torch.ones(n_genes) * 0.1)
        
    def forward(
        self, 
        z: torch.Tensor, 
        x_original: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode with residual connection.
        
        Args:
            z: Cell embeddings [N, D]
            x_original: Original input expression [N, n_genes]
            
        Returns:
            Reconstructed expression [N, n_genes]
        """
        # Learn correction from embedding
        correction = self.correction_net(z) * self.correction_scale
        
        if x_original is not None:
            # Gated residual: gate controls how much original to keep
            gate = self.gate(z)
            out = gate * x_original + (1 - gate) * correction
        else:
            # Fallback to pure correction if no input provided
            out = correction
            
        return out


class ImprovedGeneDecoder(nn.Module):
    """
    Improved gene expression decoder with residual connections
    and gene-specific parameters.
    
    Args:
        hidden_dim: Input embedding dimension
        n_genes: Number of output genes
        decoder_dims: Hidden dimensions for decoder
        n_residual_blocks: Number of residual blocks
        dropout: Dropout rate
        use_gene_bias: Whether to learn gene-specific biases
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        decoder_dims: List[int] = [512, 1024, 512],
        n_residual_blocks: int = 2,
        dropout: float = 0.1,
        use_gene_bias: bool = True,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, decoder_dims[0]),
            nn.LayerNorm(decoder_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(decoder_dims[0], dropout)
            for _ in range(n_residual_blocks)
        ])
        
        # Decoder layers
        layers = []
        in_dim = decoder_dims[0]
        for dim in decoder_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(in_dim, n_genes)
        
        # Gene-specific bias (captures baseline expression)
        if use_gene_bias:
            self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        else:
            self.register_parameter('gene_bias', None)
            
        # Gene-specific scale
        self.gene_scale = nn.Parameter(torch.ones(n_genes))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to gene expression.
        
        Args:
            z: Cell embeddings [N, D]
            
        Returns:
            Reconstructed expression [N, n_genes]
        """
        # Project
        h = self.input_proj(z)
        
        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)
        
        # Decode
        h = self.decoder(h)
        
        # Output
        out = self.output_proj(h)
        
        # Apply gene-specific parameters
        out = out * self.gene_scale
        if self.gene_bias is not None:
            out = out + self.gene_bias
            
        return out


class ZINBDecoder(nn.Module):
    """
    Zero-Inflated Negative Binomial Decoder.
    
    Models gene expression as ZINB distribution, which is more
    appropriate for scRNA-seq count data with:
    - Overdispersion (NB component)
    - Excess zeros (ZI component)
    
    Outputs:
        mean: Expected expression (μ)
        dispersion: Gene-specific dispersion (θ)
        dropout: Dropout probability (π)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        decoder_dims: List[int] = [512, 512],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        
        # Shared decoder
        layers = []
        in_dim = hidden_dim
        for dim in decoder_dims:
            layers.extend([
                nn.Linear(in_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = dim
            
        self.decoder = nn.Sequential(*layers)
        
        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(in_dim, n_genes),
            nn.Softplus(),
        )
        
        self.dropout_head = nn.Sequential(
            nn.Linear(in_dim, n_genes),
            nn.Sigmoid(),
        )
        
        # Gene-specific dispersion (inverse of NB's r parameter)
        # Learned per gene, not per cell
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        
    def forward(
        self, 
        z: torch.Tensor,
        return_distribution: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode to ZINB parameters.
        
        Args:
            z: Cell embeddings [N, D]
            return_distribution: Whether to return full distribution params
            
        Returns:
            Dictionary with 'mean' and optionally 'theta', 'dropout'
        """
        h = self.decoder(z)
        
        mean = self.mean_head(h)
        pi = self.dropout_head(h)  # Dropout probability
        theta = torch.exp(self.log_theta).unsqueeze(0).expand(z.size(0), -1)
        
        if return_distribution:
            return {
                'mean': mean,
                'theta': theta,
                'dropout': pi,
            }
        else:
            # Return expected value accounting for dropout
            return {'mean': mean * (1 - pi)}
    
    def sample(
        self, 
        z: torch.Tensor, 
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the ZINB distribution."""
        params = self.forward(z, return_distribution=True)
        
        mean = params['mean']
        theta = params['theta']
        pi = params['dropout']
        
        # Sample from NB
        # NB parameterization: mean=μ, var=μ + μ²/θ
        rate = theta / (theta + mean)
        nb_samples = torch.distributions.NegativeBinomial(
            total_count=theta, 
            probs=rate
        ).sample((n_samples,))
        
        # Apply dropout mask
        dropout_mask = torch.bernoulli(pi.unsqueeze(0).expand(n_samples, -1, -1))
        samples = nb_samples * (1 - dropout_mask)
        
        return samples.squeeze(0) if n_samples == 1 else samples


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial loss function.
    
    Computes negative log-likelihood of observed counts
    under ZINB distribution.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        mean: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
        target: torch.Tensor,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute ZINB NLL.
        
        Args:
            mean: Predicted mean [N, G]
            theta: Dispersion [N, G] or [G]
            pi: Dropout probability [N, G]
            target: Observed counts [N, G]
            scale_factor: Library size scaling
        """
        # Scale mean by library size
        mean = mean * scale_factor
        
        # Negative Binomial component
        # log P(x | μ, θ) = log Γ(x+θ) - log Γ(θ) - log Γ(x+1) 
        #                   + θ log(θ/(θ+μ)) + x log(μ/(θ+μ))
        
        theta = theta + self.eps
        mean = mean + self.eps
        
        # Log prob of NB
        nb_case = (
            torch.lgamma(target + theta) 
            - torch.lgamma(theta) 
            - torch.lgamma(target + 1)
            + theta * torch.log(theta / (theta + mean))
            + target * torch.log(mean / (theta + mean))
        )
        
        # Zero-inflation
        # P(x=0) = π + (1-π) * NB(0|μ,θ)
        # P(x>0) = (1-π) * NB(x|μ,θ)
        
        zero_nb = theta * torch.log(theta / (theta + mean))
        
        zero_case = torch.log(pi + (1 - pi) * torch.exp(zero_nb) + self.eps)
        non_zero_case = torch.log(1 - pi + self.eps) + nb_case
        
        # Select based on whether count is zero
        nll = -torch.where(target < 0.5, zero_case, non_zero_case)
        
        return nll.mean()


class NegativeBinomialLoss(nn.Module):
    """
    Negative Binomial loss (without zero-inflation).
    
    More appropriate for normalized, non-sparse data.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        mean: torch.Tensor,
        theta: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NB NLL.
        
        Args:
            mean: Predicted mean [N, G]
            theta: Dispersion [N, G] or [G]
            target: Observed values [N, G]
        """
        theta = theta + self.eps
        mean = mean + self.eps
        
        # NB log likelihood
        nll = (
            torch.lgamma(theta)
            + torch.lgamma(target + 1)
            - torch.lgamma(target + theta)
            - theta * torch.log(theta / (theta + mean))
            - target * torch.log(mean / (theta + mean))
        )
        
        return nll.mean()


class CosineReconstructionLoss(nn.Module):
    """
    Cosine similarity loss for reconstruction.
    
    Useful when absolute values matter less than relative patterns.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 1 - cosine similarity."""
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        
        return (1 - cosine_sim).mean()


class CombinedReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss with multiple components.
    
    Combines:
    - MSE for overall magnitude
    - Cosine for pattern similarity
    - Gene-weighted loss for important genes
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.5,
        correlation_weight: float = 0.5,
        gene_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.correlation_weight = correlation_weight
        
        if gene_weights is not None:
            self.register_buffer('gene_weights', gene_weights)
        else:
            self.gene_weights = None
            
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined reconstruction loss.
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # MSE loss
        mse = F.mse_loss(pred, target, reduction='none')
        if self.gene_weights is not None:
            mse = mse * self.gene_weights.unsqueeze(0)
        mse = mse.mean()
        loss_dict['mse'] = mse
        total_loss = total_loss + self.mse_weight * mse
        
        # Cosine loss
        if self.cosine_weight > 0:
            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            cosine = (1 - (pred_norm * target_norm).sum(dim=-1)).mean()
            loss_dict['cosine'] = cosine
            total_loss = total_loss + self.cosine_weight * cosine
        
        # Per-cell correlation loss
        if self.correlation_weight > 0:
            pred_centered = pred - pred.mean(dim=-1, keepdim=True)
            target_centered = target - target.mean(dim=-1, keepdim=True)
            
            pred_std = pred_centered.std(dim=-1, keepdim=True) + 1e-8
            target_std = target_centered.std(dim=-1, keepdim=True) + 1e-8
            
            correlation = ((pred_centered / pred_std) * (target_centered / target_std)).mean(dim=-1)
            corr_loss = (1 - correlation).mean()
            loss_dict['correlation'] = corr_loss
            total_loss = total_loss + self.correlation_weight * corr_loss
            
        return total_loss, loss_dict


def create_reconstruction_decoder(
    hidden_dim: int,
    n_genes: int,
    decoder_type: str = 'improved',
    **kwargs
) -> nn.Module:
    """
    Factory function to create reconstruction decoder.
    
    Args:
        hidden_dim: Input embedding dimension
        n_genes: Number of genes
        decoder_type: 'basic', 'improved', 'zinb'
        **kwargs: Additional arguments for specific decoder
    """
    if decoder_type == 'basic':
        from .predictors import GeneExpressionDecoder
        return GeneExpressionDecoder(hidden_dim, n_genes, **kwargs)
    elif decoder_type == 'improved':
        return ImprovedGeneDecoder(hidden_dim, n_genes, **kwargs)
    elif decoder_type == 'zinb':
        return ZINBDecoder(hidden_dim, n_genes, **kwargs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
