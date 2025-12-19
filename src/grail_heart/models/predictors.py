"""
Prediction Heads for GRAIL-Heart

Output modules for various prediction tasks including
ligand-receptor interaction scoring, cell type classification,
and signaling network inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class LRInteractionPredictor(nn.Module):
    """
    Predicts ligand-receptor interaction scores between cell pairs.
    
    Uses bilinear attention to score interactions based on
    node embeddings and L-R pair information.
    
    Args:
        hidden_dim: Input embedding dimension
        n_lr_pairs: Number of L-R pairs (for pair-specific scoring)
        use_bilinear: Whether to use bilinear scoring
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_lr_pairs: Optional[int] = None,
        use_bilinear: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_bilinear = use_bilinear
        
        if use_bilinear:
            # Bilinear scoring
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        else:
            # MLP scoring
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            
        # Optional L-R pair specific projections
        if n_lr_pairs is not None:
            self.lr_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(n_lr_pairs)
            ])
        else:
            self.lr_projections = None
            
    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        lr_pair_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict interaction scores.
        
        Args:
            z_src: Source (ligand-expressing) cell embeddings [E, D]
            z_dst: Target (receptor-expressing) cell embeddings [E, D]
            lr_pair_idx: Optional L-R pair indices [E]
            
        Returns:
            Interaction scores [E, 1]
        """
        # Apply L-R specific projection if available
        if self.lr_projections is not None and lr_pair_idx is not None:
            z_src_proj = torch.zeros_like(z_src)
            for i, proj in enumerate(self.lr_projections):
                mask = lr_pair_idx == i
                if mask.any():
                    z_src_proj[mask] = proj(z_src[mask])
            z_src = z_src_proj
            
        if self.use_bilinear:
            scores = self.bilinear(z_src, z_dst)
        else:
            combined = torch.cat([z_src, z_dst], dim=-1)
            scores = self.mlp(combined)
            
        return scores
    
    def predict_all_pairs(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scores for all edges in a graph.
        
        Args:
            z: Node embeddings [N, D]
            edge_index: Edge indices [2, E]
            
        Returns:
            Scores for all edges [E]
        """
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        
        scores = self.forward(z_src, z_dst)
        return scores.squeeze(-1)


class SignalingNetworkPredictor(nn.Module):
    """
    Predicts cell-cell signaling network from embeddings.
    
    Outputs a weighted adjacency matrix representing
    signaling strength between all cell pairs.
    
    Args:
        hidden_dim: Input embedding dimension
        n_heads: Number of attention heads
        temperature: Softmax temperature for attention
        sparse_k: Number of top connections to keep per cell (sparsification)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        temperature: float = 1.0,
        sparse_k: Optional[int] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.sparse_k = sparse_k
        
        # Multi-head attention for signaling
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        
        # Signaling type classifier
        self.signaling_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # autocrine, paracrine, endocrine
        )
        
    def forward(
        self,
        z: torch.Tensor,
        spatial_dist: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict signaling network.
        
        Args:
            z: Cell embeddings [N, D]
            spatial_dist: Optional spatial distance matrix [N, N]
            
        Returns:
            Dictionary with:
                - adjacency: Signaling adjacency matrix [N, N]
                - signaling_type: Type probabilities [N, N, 3]
        """
        N = z.size(0)
        
        # Compute attention-based adjacency
        Q = self.query(z)  # [N, D]
        K = self.key(z)    # [N, D]
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn = attn / self.temperature
        
        # Incorporate spatial distance if provided
        if spatial_dist is not None:
            # Decay with distance
            spatial_weight = torch.exp(-spatial_dist)
            attn = attn * spatial_weight
            
        # Softmax to get adjacency weights
        adjacency = F.softmax(attn, dim=-1)
        
        # Sparsify if requested
        if self.sparse_k is not None:
            topk_vals, topk_idx = adjacency.topk(self.sparse_k, dim=-1)
            sparse_adj = torch.zeros_like(adjacency)
            sparse_adj.scatter_(-1, topk_idx, topk_vals)
            adjacency = sparse_adj
            
        # Classify signaling type for each pair
        z_expanded_i = z.unsqueeze(1).expand(N, N, -1)
        z_expanded_j = z.unsqueeze(0).expand(N, N, -1)
        pair_features = torch.cat([z_expanded_i, z_expanded_j], dim=-1)
        signaling_type = self.signaling_classifier(pair_features)
        signaling_type = F.softmax(signaling_type, dim=-1)
        
        return {
            'adjacency': adjacency,
            'signaling_type': signaling_type,
        }


class CellTypePredictor(nn.Module):
    """
    Predicts cell types from embeddings.
    
    Args:
        hidden_dim: Input embedding dimension
        n_cell_types: Number of cell type classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_cell_types: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cell_types),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict cell type logits.
        
        Args:
            z: Cell embeddings [N, D]
            
        Returns:
            Cell type logits [N, n_cell_types]
        """
        return self.classifier(z)


class GeneExpressionDecoder(nn.Module):
    """
    Decodes embeddings back to gene expression space.
    
    Used for reconstruction loss and expression imputation.
    
    Args:
        hidden_dim: Input embedding dimension
        n_genes: Number of output genes
        decoder_dims: Hidden dimensions for decoder
        dropout: Dropout rate
        output_activation: Output activation ('none', 'softplus', 'sigmoid')
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        decoder_dims: List[int] = [256, 512],
        dropout: float = 0.1,
        output_activation: str = 'softplus',
    ):
        super().__init__()
        
        layers = []
        in_dim = hidden_dim
        
        for dim in decoder_dims:
            layers.extend([
                nn.Linear(in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = dim
            
        layers.append(nn.Linear(in_dim, n_genes))
        
        self.decoder = nn.Sequential(*layers)
        
        if output_activation == 'softplus':
            self.output_act = nn.Softplus()
        elif output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = nn.Identity()
            
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to gene expression.
        
        Args:
            z: Cell embeddings [N, D]
            
        Returns:
            Reconstructed expression [N, n_genes]
        """
        return self.output_act(self.decoder(z))


class MultiTaskHead(nn.Module):
    """
    Combined prediction head for multiple tasks.
    
    Supports:
    - L-R interaction prediction
    - Cell type classification
    - Expression reconstruction
    - Signaling network inference
    
    Args:
        hidden_dim: Input embedding dimension
        n_genes: Number of genes
        n_cell_types: Number of cell types
        n_lr_pairs: Number of L-R pairs
        tasks: List of tasks to include
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        n_cell_types: Optional[int] = None,
        n_lr_pairs: Optional[int] = None,
        tasks: List[str] = ['lr', 'reconstruction'],
        decoder_type: str = 'improved',  # 'basic', 'improved', 'zinb'
    ):
        super().__init__()
        
        self.tasks = tasks
        self.decoder_type = decoder_type
        
        if 'lr' in tasks:
            self.lr_predictor = LRInteractionPredictor(
                hidden_dim=hidden_dim,
                n_lr_pairs=n_lr_pairs,
            )
            
        if 'cell_type' in tasks and n_cell_types is not None:
            self.cell_type_predictor = CellTypePredictor(
                hidden_dim=hidden_dim,
                n_cell_types=n_cell_types,
            )
            
        if 'reconstruction' in tasks:
            if decoder_type == 'residual':
                from .reconstruction import ResidualGeneDecoder
                self.decoder = ResidualGeneDecoder(
                    hidden_dim=hidden_dim,
                    n_genes=n_genes,
                    decoder_dims=[512, 512],
                )
            elif decoder_type == 'improved':
                from .reconstruction import ImprovedGeneDecoder
                self.decoder = ImprovedGeneDecoder(
                    hidden_dim=hidden_dim,
                    n_genes=n_genes,
                    decoder_dims=[512, 1024, 512],
                    n_residual_blocks=2,
                )
            elif decoder_type == 'zinb':
                from .reconstruction import ZINBDecoder
                self.decoder = ZINBDecoder(
                    hidden_dim=hidden_dim,
                    n_genes=n_genes,
                )
            else:
                self.decoder = GeneExpressionDecoder(
                    hidden_dim=hidden_dim,
                    n_genes=n_genes,
                )
            
        if 'signaling' in tasks:
            self.signaling_predictor = SignalingNetworkPredictor(
                hidden_dim=hidden_dim,
            )
            
    def forward(
        self,
        z: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        spatial_dist: Optional[torch.Tensor] = None,
        x_original: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all predictions.
        
        Args:
            z: Cell embeddings [N, D]
            edge_index: Edge indices for L-R prediction [2, E]
            spatial_dist: Spatial distance matrix [N, N]
            x_original: Original gene expression for residual decoder [N, n_genes]
            
        Returns:
            Dictionary of predictions
        """
        outputs = {}
        
        if 'lr' in self.tasks and edge_index is not None:
            outputs['lr_scores'] = self.lr_predictor.predict_all_pairs(z, edge_index)
            
        if 'cell_type' in self.tasks:
            outputs['cell_type_logits'] = self.cell_type_predictor(z)
            
        if 'reconstruction' in self.tasks:
            if self.decoder_type == 'zinb':
                zinb_out = self.decoder(z, return_distribution=True)
                outputs['reconstruction'] = zinb_out['mean']
                outputs['zinb_theta'] = zinb_out['theta']
                outputs['zinb_dropout'] = zinb_out['dropout']
            elif self.decoder_type == 'residual':
                # Residual decoder needs original input
                outputs['reconstruction'] = self.decoder(z, x_original)
            else:
                outputs['reconstruction'] = self.decoder(z)
            
        if 'signaling' in self.tasks:
            outputs['signaling'] = self.signaling_predictor(z, spatial_dist)
            
        return outputs
