"""
GRAIL-Heart: Graph-based Reconstruction of Artificial Intercellular Links

Main model architecture that integrates gene encoding, graph attention,
and multi-task prediction for cardiac spatial transcriptomics analysis.

This model implements BOTH forward and inverse modelling:
- FORWARD: Expression → L-R predictions (original GRAIL-Heart)
- INVERSE: Observed fate/phenotype → Inferred causal L-R signals

Reference (Abstract):
    "This inverse modelling framework will also elucidate mechanosensitive pathways
    that modulate early-stage cardiac tissue patterning, linking molecular signalling
    to the formation of soft contractile structures."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union

from .encoders import GeneEncoder, MultiModalEncoder, VariationalGeneEncoder
from .gat_layers import GATStack, MultiHeadGATLayer, EdgeTypeGATConv
from .predictors import (
    LRInteractionPredictor,
    SignalingNetworkPredictor,
    CellTypePredictor,
    GeneExpressionDecoder,
    MultiTaskHead,
)
from .inverse_modelling import (
    InverseSignalInference,
    CellFatePredictionHead,
    CounterfactualReasoner,
    LRToTargetGeneDecoder,
    MechanosensitivePathwayModule,
)


class GRAILHeart(nn.Module):
    """
    GRAIL-Heart: Graph Neural Network for Cardiac Spatial Transcriptomics.
    
    Architecture:
    1. Gene Expression Encoder: MLP to encode gene expression profiles
    2. Spatial Position Encoder: Sinusoidal encoding for spatial coordinates
    3. Graph Attention Network: Multi-head attention over cell graph
    4. Multi-Task Predictor: L-R scoring, cell typing, expression reconstruction
    5. Inverse Modelling: Infer causal L-R signals from observed fates (NEW)
    
    Forward vs Inverse Modelling:
    - FORWARD: Expression + Spatial → L-R predictions, cell fates
    - INVERSE: Observed fate/phenotype → Inferred causal L-R signals
    
    Args:
        n_genes: Number of input genes
        hidden_dim: Hidden dimension throughout the model
        n_gat_layers: Number of GAT layers
        n_heads: Number of attention heads
        n_edge_types: Number of edge types (spatial, L-R, etc.)
        n_cell_types: Number of cell types for classification
        n_lr_pairs: Number of L-R pairs
        encoder_dims: Hidden dimensions for gene encoder
        dropout: Dropout rate
        use_spatial: Whether to use spatial coordinates
        use_variational: Whether to use variational encoder
        tasks: List of prediction tasks
        decoder_type: Type of gene expression decoder ('basic', 'improved', 'zinb')
        use_inverse_modelling: Whether to enable inverse modelling (NEW)
        n_fates: Number of cell fate categories for inverse modelling
        n_pathways: Number of signaling pathways for inverse modelling
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        n_gat_layers: int = 3,
        n_heads: int = 8,
        n_edge_types: int = 2,
        n_cell_types: Optional[int] = None,
        n_lr_pairs: Optional[int] = None,
        encoder_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        use_spatial: bool = True,
        use_variational: bool = False,
        tasks: List[str] = ['lr', 'reconstruction'],
        decoder_type: str = 'residual',  # 'basic', 'improved', 'residual', or 'zinb'
        # NEW: Inverse modelling parameters
        use_inverse_modelling: bool = True,
        n_fates: Optional[int] = None,
        n_pathways: int = 20,
        n_mechano_pathways: int = 8,
        # Biologically grounded gene-set masks (WP1)
        pathway_gene_mask: Optional[torch.Tensor] = None,
        mechano_gene_mask: Optional[torch.Tensor] = None,
        mechano_pathway_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.use_spatial = use_spatial
        self.use_variational = use_variational
        self.tasks = tasks
        self.decoder_type = decoder_type
        self.use_inverse_modelling = use_inverse_modelling
        
        # Gene Expression Encoder
        if use_variational:
            self.gene_encoder = VariationalGeneEncoder(
                n_genes=n_genes,
                hidden_dims=encoder_dims,
                latent_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.gene_encoder = GeneEncoder(
                n_genes=n_genes,
                hidden_dims=encoder_dims,
                latent_dim=hidden_dim,
                dropout=dropout,
            )
            
        # Multi-modal encoder if using spatial coordinates
        if use_spatial:
            self.multimodal_encoder = MultiModalEncoder(
                n_genes=n_genes,
                gene_hidden_dims=encoder_dims,
                gene_latent_dim=hidden_dim,
                spatial_dim=2,
                spatial_embed_dim=64,
                n_cell_types=n_cell_types,
                cell_type_embed_dim=64,
                output_dim=hidden_dim,
                dropout=dropout,
                fusion='concat',
            )
            
        # Graph Attention Network
        self.gat = GATStack(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=n_gat_layers,
            heads=n_heads,
            n_edge_types=n_edge_types,
            dropout=dropout,
            jk='cat',  # Jumping knowledge
        )
        
        # Multi-task prediction head (FORWARD modelling)
        self.predictor = MultiTaskHead(
            hidden_dim=hidden_dim,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            n_lr_pairs=n_lr_pairs,
            tasks=tasks,
            decoder_type=decoder_type,
        )
        
        # INVERSE MODELLING MODULE (NEW)
        # Implements the abstract's claim: "inverse modelling framework to elucidate
        # mechanosensitive pathways that modulate early-stage cardiac tissue patterning"
        self.inverse_module = None
        if use_inverse_modelling:
            # Determine number of fates from cell types or default
            actual_n_fates = n_fates if n_fates is not None else (n_cell_types if n_cell_types is not None else 10)
            
            self.inverse_module = InverseSignalInference(
                hidden_dim=hidden_dim,
                n_genes=n_genes,
                n_fates=actual_n_fates,
                n_pathways=n_pathways,
                n_mechano_pathways=n_mechano_pathways,
                pathway_gene_mask=pathway_gene_mask,
                mechano_gene_mask=mechano_gene_mask,
                mechano_pathway_names=mechano_pathway_names,
            )
        
        # Variational KL loss weight
        self.kl_weight = 0.001
        
    def encode(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        cell_type: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encode cell features to latent space.
        
        Args:
            x: Gene expression [N, n_genes]
            pos: Spatial coordinates [N, 2]
            cell_type: Cell type indices [N]
            
        Returns:
            Cell embeddings [N, hidden_dim]
            If variational: (z, mu, logvar)
        """
        if self.use_spatial and pos is not None:
            z = self.multimodal_encoder(x, pos, cell_type)
        else:
            if self.use_variational:
                z, mu, logvar = self.gene_encoder(x)
                return z, mu, logvar
            else:
                z = self.gene_encoder(x)
                
        return z
    
    def forward(
        self,
        data: Union[Data, Batch],
        return_embeddings: bool = False,
        run_inverse: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GRAIL-Heart (both forward and inverse modelling).
        
        Args:
            data: PyG Data object with:
                - x: Gene expression [N, n_genes]
                - edge_index: Graph edges [2, E]
                - pos: Spatial coordinates [N, 2] (optional)
                - y: Cell type labels [N] (optional)
                - edge_type: Edge types [E] (optional)
                - edge_weight: Edge weights [E] (optional)
            return_embeddings: Whether to return intermediate embeddings
            run_inverse: Whether to run inverse modelling (default: True)
            
        Returns:
            Dictionary with predictions and optional embeddings including:
            
            FORWARD MODELLING OUTPUTS:
                - lr_scores: L-R interaction predictions [E]
                - reconstruction: Reconstructed expression [N, n_genes]
                - node_embeddings: Cell embeddings [N, hidden_dim]
                
            INVERSE MODELLING OUTPUTS (if enabled):
                - fate_logits: Cell fate predictions [N, n_fates]
                - fate_trajectory: Continuous fate embedding [N, fate_dim]
                - differentiation_score: Differentiation progress [N, 1]
                - causal_lr_scores: Causal importance of each L-R [E]
                - pathway_activation: Signaling pathway activations [E, n_pathways]
                - target_gene_effects: Effect of L-R on genes [E, n_genes]
                - mechano_pathway_activation: Mechanosensitive pathway activity [E, n_mechano]
        """
        x = data.x
        edge_index = data.edge_index
        pos = data.pos if hasattr(data, 'pos') else None
        cell_type = data.y if hasattr(data, 'y') else None
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        # Encode cell features
        encode_result = self.encode(x, pos, cell_type)
        
        if self.use_variational and isinstance(encode_result, tuple):
            z, mu, logvar = encode_result
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z = encode_result
            kl_loss = None
            
        # Graph attention processing
        z_gat = self.gat(z, edge_index, edge_type, edge_weight)
        
        # Compute spatial distances if available
        spatial_dist = None
        if pos is not None:
            spatial_dist = torch.cdist(pos, pos)
            
        # ===== FORWARD MODELLING =====
        # Multi-task predictions (pass original expression for residual decoder)
        outputs = self.predictor(z_gat, edge_index, spatial_dist, x_original=x)
        
        # Always include node embeddings for contrastive learning
        outputs['node_embeddings'] = z_gat
        
        # Add KL loss for variational
        if kl_loss is not None:
            outputs['kl_loss'] = kl_loss * self.kl_weight
            
        # ===== INVERSE MODELLING =====
        # "This inverse modelling framework will also elucidate mechanosensitive 
        # pathways that modulate early-stage cardiac tissue patterning"
        if self.inverse_module is not None and run_inverse and 'lr_scores' in outputs:
            try:
                inverse_outputs = self.inverse_module(
                    z=z_gat,
                    edge_index=edge_index,
                    lr_scores=torch.sigmoid(outputs['lr_scores']),
                    expression=x,
                    spatial_coords=pos,
                )
                
                # Add inverse modelling outputs with 'inverse_' prefix for clarity
                outputs['fate_logits'] = inverse_outputs['fate_logits']
                outputs['fate_trajectory'] = inverse_outputs['fate_trajectory']
                outputs['differentiation_score'] = inverse_outputs['differentiation_score']
                outputs['causal_lr_scores'] = inverse_outputs['causal_lr_scores']
                outputs['pathway_activation'] = inverse_outputs['pathway_activation']
                outputs['target_gene_effects'] = inverse_outputs['target_gene_effects']
                outputs['mechano_pathway_activation'] = inverse_outputs['mechano_pathway_activation']
                outputs['mechano_differentiation_effect'] = inverse_outputs['mechano_differentiation_effect']
                outputs['fate_representation'] = inverse_outputs['fate_representation']
                
                if 'predicted_expression_from_lr' in inverse_outputs:
                    outputs['predicted_expression_from_lr'] = inverse_outputs['predicted_expression_from_lr']
                    outputs['delta_expression'] = inverse_outputs['delta_expression']

                # WP1: Pathway grounding regularisation
                if 'pathway_grounding_loss' in inverse_outputs:
                    outputs['pathway_grounding_loss'] = inverse_outputs['pathway_grounding_loss']

                # WP5: Cycle-consistency LR reconstruction
                if 'cycle_reconstructed_lr' in inverse_outputs:
                    outputs['cycle_reconstructed_lr'] = inverse_outputs['cycle_reconstructed_lr']
            except Exception as e:
                # Log but don't fail if inverse modelling has issues
                import warnings
                warnings.warn(f"Inverse modelling failed: {e}")
            
        # Optionally return additional embeddings
        if return_embeddings:
            outputs['z_initial'] = z
            outputs['z_gat'] = z_gat
            
        return outputs
    
    def infer_causal_signals(
        self,
        data: Data,
        observed_fate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        INVERSE INFERENCE: Given observed data, infer causal L-R signals.
        
        This is the core inverse modelling function that answers:
        "What L-R signals caused this differentiation pattern?"
        
        Args:
            data: PyG Data object with expression and spatial info
            observed_fate: Optional observed fate labels [N, n_fates]
                          If not provided, uses predicted fates
                          
        Returns:
            Dictionary with:
                - inferred_lr_importance: Importance of each L-R edge [E]
                - causal_pathways: Which pathways are causally important
                - mechano_importance: Mechanosensitive pathway importance
        """
        if self.inverse_module is None:
            raise ValueError("Inverse modelling not enabled. Set use_inverse_modelling=True")
            
        # Run forward pass first to get embeddings and L-R scores
        outputs = self.forward(data, run_inverse=True)
        
        # If no observed fate provided, use the predicted fate
        if observed_fate is None:
            observed_fate = F.softmax(outputs['fate_logits'], dim=-1)
            
        # Run inverse inference
        inverse_results = self.inverse_module.infer_causal_signals(
            observed_fate=observed_fate,
            observed_expression=data.x,
            z=outputs['node_embeddings'],
            edge_index=data.edge_index,
        )
        
        # Combine with forward outputs for complete analysis
        inverse_results['forward_lr_scores'] = outputs.get('lr_scores', None)
        inverse_results['causal_lr_scores'] = outputs.get('causal_lr_scores', None)
        inverse_results['pathway_activation'] = outputs.get('pathway_activation', None)
        inverse_results['mechano_pathway_activation'] = outputs.get('mechano_pathway_activation', None)
        
        return inverse_results
    
    def predict_interactions(
        self,
        data: Data,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict L-R interactions above threshold.
        
        Args:
            data: PyG Data object
            threshold: Score threshold for interactions
            
        Returns:
            edge_index: Predicted interaction edges
            scores: Interaction scores
        """
        outputs = self.forward(data)
        
        if 'lr_scores' in outputs:
            scores = torch.sigmoid(outputs['lr_scores'])
            mask = scores > threshold
            
            pred_edges = data.edge_index[:, mask]
            pred_scores = scores[mask]
            
            return pred_edges, pred_scores
        else:
            raise ValueError("L-R prediction not enabled in model")
    
    def get_signaling_network(
        self,
        data: Data,
        k_neighbors: int = 10,
    ) -> torch.Tensor:
        """
        Get inferred signaling network.
        
        Args:
            data: PyG Data object
            k_neighbors: Number of top neighbors per cell
            
        Returns:
            Signaling adjacency matrix [N, N]
        """
        outputs = self.forward(data)
        
        if 'signaling' in outputs:
            adj = outputs['signaling']['adjacency']
            
            # Sparsify to top-k
            if k_neighbors < adj.size(1):
                topk_vals, topk_idx = adj.topk(k_neighbors, dim=-1)
                sparse_adj = torch.zeros_like(adj)
                sparse_adj.scatter_(-1, topk_idx, topk_vals)
                return sparse_adj
                
            return adj
        else:
            raise ValueError("Signaling network prediction not enabled")


class GRAILHeartLite(nn.Module):
    """
    Lightweight version of GRAIL-Heart for faster training.
    
    Simplified architecture with single-scale GAT and
    reduced parameter count for rapid prototyping.
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        
        self.gat = EdgeTypeGATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // n_heads,
            heads=n_heads,
            n_edge_types=2,
            dropout=dropout,
        )
        
        self.predictor = LRInteractionPredictor(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_genes)
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.encoder(data.x)
        
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None
        z = self.gat(x, data.edge_index, edge_type)
        
        lr_scores = self.predictor.predict_all_pairs(z, data.edge_index)
        reconstruction = F.softplus(self.decoder(z))
        
        return {
            'lr_scores': lr_scores,
            'reconstruction': reconstruction,
        }


def create_grail_heart(
    n_genes: int,
    config: Optional[Dict] = None,
) -> GRAILHeart:
    """
    Factory function to create GRAIL-Heart model.
    
    Args:
        n_genes: Number of input genes
        config: Optional configuration dictionary
        
    Returns:
        Configured GRAIL-Heart model with forward AND inverse modelling
    """
    default_config = {
        'hidden_dim': 256,
        'n_gat_layers': 3,
        'n_heads': 8,
        'n_edge_types': 2,
        'n_cell_types': None,
        'n_lr_pairs': None,
        'encoder_dims': [512, 256],
        'dropout': 0.1,
        'use_spatial': True,
        'use_variational': False,
        'tasks': ['lr', 'reconstruction'],
        'decoder_type': 'residual',
        # Inverse modelling parameters (NEW)
        'use_inverse_modelling': True,
        'n_fates': None,  # Will default to n_cell_types
        'n_pathways': 20,
        'n_mechano_pathways': 8,
        # Biologically-grounded pathway masks (WP1)
        'pathway_gene_mask': None,
        'mechano_gene_mask': None,
        'mechano_pathway_names': None,
    }
    
    if config is not None:
        default_config.update(config)
        
    return GRAILHeart(n_genes=n_genes, **default_config)
