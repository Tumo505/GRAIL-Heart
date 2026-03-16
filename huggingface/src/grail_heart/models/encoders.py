"""
Gene Expression Encoders for GRAIL-Heart

Neural network modules for encoding gene expression profiles
into latent representations suitable for graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class GeneEncoder(nn.Module):
    """
    Encodes gene expression profiles into latent representations.
    
    Uses a multi-layer MLP with batch normalization and dropout
    to transform raw gene expression into fixed-size embeddings.
    
    Args:
        n_genes: Number of input genes
        hidden_dims: List of hidden layer dimensions
        latent_dim: Output embedding dimension
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'silu')
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 128,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        
        # Select activation function
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'elu': nn.ELU,
        }
        act_fn = activations.get(activation, nn.GELU)
        
        # Build encoder layers
        layers = []
        in_dim = n_genes
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        # Final projection to latent space
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode gene expression profiles.
        
        Args:
            x: Gene expression [batch_size, n_genes]
            
        Returns:
            Latent embeddings [batch_size, latent_dim]
        """
        return self.encoder(x)


class VariationalGeneEncoder(nn.Module):
    """
    Variational encoder for gene expression.
    
    Outputs mean and log variance for variational inference,
    enabling uncertainty quantification in embeddings.
    
    Args:
        n_genes: Number of input genes
        hidden_dims: List of hidden layer dimensions
        latent_dim: Output embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        
        # Shared encoder
        layers = []
        in_dim = n_genes
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        self.shared_encoder = nn.Sequential(*layers)
        
        # Mean and log variance projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with variational inference.
        
        Args:
            x: Gene expression [batch_size, n_genes]
            
        Returns:
            z: Sampled latent [batch_size, latent_dim]
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]
        """
        h = self.shared_encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling (use mean)."""
        h = self.shared_encoder(x)
        return self.fc_mu(h)


class SpatialPositionEncoder(nn.Module):
    """
    Encodes spatial coordinates into positional embeddings.
    
    Uses sinusoidal positional encoding similar to transformers,
    with learnable projection for spatial-aware representations.
    
    Args:
        coord_dim: Input coordinate dimensions (2 for 2D, 3 for 3D)
        embed_dim: Output embedding dimension
        max_freq: Maximum frequency for sinusoidal encoding
        n_freq_bands: Number of frequency bands
    """
    
    def __init__(
        self,
        coord_dim: int = 2,
        embed_dim: int = 64,
        max_freq: float = 10.0,
        n_freq_bands: int = 16,
    ):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.n_freq_bands = n_freq_bands
        
        # Frequency bands (log-spaced)
        freqs = torch.exp(
            torch.linspace(0, np.log(max_freq), n_freq_bands)
        )
        self.register_buffer('freqs', freqs)
        
        # Input dimension: coord_dim * n_freq_bands * 2 (sin + cos)
        encoding_dim = coord_dim * n_freq_bands * 2
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(encoding_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial coordinates.
        
        Args:
            coords: Spatial coordinates [batch_size, coord_dim]
            
        Returns:
            Positional embeddings [batch_size, embed_dim]
        """
        # Compute sinusoidal encoding
        # coords: [B, D], freqs: [F]
        # coords.unsqueeze(-1): [B, D, 1]
        # freqs: [F] -> [1, 1, F]
        
        scaled = coords.unsqueeze(-1) * self.freqs.view(1, 1, -1) * 2 * np.pi
        # scaled: [B, D, F]
        
        # Compute sin and cos
        encodings = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        # encodings: [B, D, 2F]
        
        # Flatten
        encodings = encodings.view(coords.shape[0], -1)
        # encodings: [B, D * 2F]
        
        return self.projection(encodings)


class CellTypeEncoder(nn.Module):
    """
    Encodes cell type labels into embeddings.
    
    Uses learnable embeddings for categorical cell types,
    with optional attention over multiple possible types.
    
    Args:
        n_cell_types: Number of cell type categories
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        n_cell_types: int,
        embed_dim: int = 64,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(n_cell_types, embed_dim)
        
    def forward(self, cell_types: torch.Tensor) -> torch.Tensor:
        """
        Encode cell types.
        
        Args:
            cell_types: Cell type indices [batch_size]
            
        Returns:
            Cell type embeddings [batch_size, embed_dim]
        """
        return self.embedding(cell_types)


class MultiModalEncoder(nn.Module):
    """
    Combined encoder for multiple input modalities.
    
    Fuses gene expression, spatial position, and cell type
    information into a unified cell representation.
    
    Args:
        n_genes: Number of input genes
        gene_hidden_dims: Hidden dims for gene encoder
        gene_latent_dim: Gene embedding dimension
        spatial_dim: Spatial coordinate dimension
        spatial_embed_dim: Spatial embedding dimension
        n_cell_types: Number of cell types (None if not used)
        cell_type_embed_dim: Cell type embedding dimension
        output_dim: Final output dimension
        dropout: Dropout rate
        fusion: Fusion method ('concat', 'add', 'gated')
    """
    
    def __init__(
        self,
        n_genes: int,
        gene_hidden_dims: List[int] = [512, 256],
        gene_latent_dim: int = 128,
        spatial_dim: int = 2,
        spatial_embed_dim: int = 64,
        n_cell_types: Optional[int] = None,
        cell_type_embed_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.1,
        fusion: str = 'concat',
    ):
        super().__init__()
        
        self.fusion = fusion
        
        # Gene encoder
        self.gene_encoder = GeneEncoder(
            n_genes=n_genes,
            hidden_dims=gene_hidden_dims,
            latent_dim=gene_latent_dim,
            dropout=dropout,
        )
        
        # Spatial encoder
        self.spatial_encoder = SpatialPositionEncoder(
            coord_dim=spatial_dim,
            embed_dim=spatial_embed_dim,
        )
        
        # Cell type encoder (optional)
        self.cell_type_encoder = None
        if n_cell_types is not None:
            self.cell_type_encoder = CellTypeEncoder(
                n_cell_types=n_cell_types,
                embed_dim=cell_type_embed_dim,
            )
            
        # Compute fusion dimension
        if fusion == 'concat':
            fusion_dim = gene_latent_dim + spatial_embed_dim
            if n_cell_types is not None:
                fusion_dim += cell_type_embed_dim
        else:
            # For add or gated, all embeddings must be same size
            fusion_dim = output_dim
            
        # Output projection
        if fusion == 'concat':
            self.output_proj = nn.Sequential(
                nn.Linear(fusion_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
            )
        elif fusion == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Identity()
            
    def forward(
        self,
        expression: torch.Tensor,
        spatial: Optional[torch.Tensor] = None,
        cell_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode multi-modal cell data.
        
        Args:
            expression: Gene expression [batch_size, n_genes]
            spatial: Spatial coordinates [batch_size, spatial_dim]
            cell_type: Cell type indices [batch_size]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Encode gene expression
        gene_embed = self.gene_encoder(expression)
        
        embeddings = [gene_embed]
        
        # Encode spatial position
        if spatial is not None:
            spatial_embed = self.spatial_encoder(spatial)
            embeddings.append(spatial_embed)
            
        # Encode cell type
        if cell_type is not None and self.cell_type_encoder is not None:
            cell_embed = self.cell_type_encoder(cell_type)
            embeddings.append(cell_embed)
            
        # Fuse embeddings
        if self.fusion == 'concat':
            fused = torch.cat(embeddings, dim=-1)
        elif self.fusion == 'add':
            fused = sum(embeddings)
        elif self.fusion == 'gated':
            # Gated fusion with gene as primary
            other = sum(embeddings[1:]) if len(embeddings) > 1 else torch.zeros_like(gene_embed)
            gate_input = torch.cat([gene_embed, other], dim=-1)
            gate = self.gate(gate_input)
            fused = gate * gene_embed + (1 - gate) * other
        else:
            fused = torch.cat(embeddings, dim=-1)
            
        return self.output_proj(fused)


# Need numpy for SpatialPositionEncoder
import numpy as np
