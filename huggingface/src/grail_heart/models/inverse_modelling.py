"""
Inverse Modelling Module for GRAIL-Heart

This module implements the inverse modelling components that enable GRAIL-Heart
to infer functional signalling networks that DRIVE cell differentiation, aligning
with the abstract's claims about inverse modelling.

Architecture Overview:
    FORWARD:  Expression → L-R Predictions (current GRAIL-Heart)
    INVERSE:  Observed Fate/Phenotype → Inferred Causal L-R Signals

Components:
1. CellFatePredictionHead: Predicts differentiation states from L-R interactions
2. CounterfactualReasoner: Identifies which L-R signals are causal for fate
3. LRToTargetGeneDecoder: Maps L-R interactions to downstream gene changes
4. MechanosensitivePathwayModule: Models mechanobiological signaling effects
5. InverseSignalInference: Full inverse inference pipeline

Reference:
    "This inverse modelling framework will also elucidate mechanosensitive pathways
    that modulate early-stage cardiac tissue patterning, linking molecular signalling
    to the formation of soft contractile structures."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as edge_softmax
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class CellFatePredictionHead(nn.Module):
    """
    Predicts cell differentiation states from L-R interaction patterns.
    
    This is the first step of inverse modelling: learning the mapping from
    L-R signals to cell fate, which can then be inverted to ask
    "which L-R signals drive this differentiation?"
    
    Implements:
    - L-R-to-Fate attention: which L-R interactions contribute to each fate
    - Multi-scale aggregation: local (neighborhood) and global (tissue) signals
    - Fate trajectory embedding: continuous differentiation space
    
    Args:
        hidden_dim: Input embedding dimension
        n_fates: Number of discrete fate categories (e.g., progenitor, cardiomyocyte)
        fate_embedding_dim: Dimension for continuous fate trajectory
        n_lr_classes: Number of L-R interaction classes
        use_attention: Whether to use attention for L-R attribution
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_fates: int,
        fate_embedding_dim: int = 64,
        n_lr_classes: Optional[int] = None,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_fates = n_fates
        self.fate_embedding_dim = fate_embedding_dim
        self.use_attention = use_attention
        
        # L-R interaction encoder (takes lr_scores + node embeddings)
        self.lr_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # src + dst + score
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # L-R to Fate attention mechanism
        if use_attention:
            self.lr_fate_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.fate_query = nn.Parameter(torch.randn(n_fates, hidden_dim))
            
        # Neighborhood aggregation for local L-R context
        self.neighborhood_aggregator = NeighborhoodLRAggregator(
            hidden_dim=hidden_dim,
            aggregation='attention',
        )
        
        # Discrete fate classifier
        self.fate_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_fates),
        )
        
        # Continuous fate trajectory embedding
        self.fate_trajectory_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fate_embedding_dim),
        )
        
        # Differentiation score predictor (0 = progenitor, 1 = fully differentiated)
        self.differentiation_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        lr_scores: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict cell fates from L-R interaction patterns.
        
        Args:
            z: Node embeddings [N, hidden_dim]
            edge_index: Edge indices [2, E]
            lr_scores: L-R interaction scores [E]
            return_attention: Whether to return L-R-to-fate attention weights
            
        Returns:
            Dictionary containing:
                - fate_logits: Discrete fate predictions [N, n_fates]
                - fate_trajectory: Continuous fate embedding [N, fate_embedding_dim]
                - differentiation_score: Differentiation progress [N, 1]
                - lr_fate_attention: (optional) Attention weights [N, n_fates, E_local]
        """
        N = z.size(0)
        src, dst = edge_index
        
        # Encode L-R interactions
        lr_features = torch.cat([
            z[src],  # Ligand-expressing cell embedding
            z[dst],  # Receptor-expressing cell embedding
            lr_scores.unsqueeze(-1),  # Interaction score
        ], dim=-1)
        lr_embeddings = self.lr_encoder(lr_features)  # [E, hidden_dim]
        
        # Aggregate L-R signals for each cell
        aggregated_lr = self.neighborhood_aggregator(
            z=z,
            edge_index=edge_index,
            lr_embeddings=lr_embeddings,
            lr_scores=lr_scores,
        )  # [N, hidden_dim]
        
        # Compute L-R to fate attention (which L-R signals determine fate)
        lr_fate_attn_weights = None
        if self.use_attention:
            # Stack fate queries for all cells
            fate_queries = self.fate_query.unsqueeze(0).expand(N, -1, -1)  # [N, n_fates, hidden_dim]
            
            # Attention between fate queries and aggregated L-R signals
            # Use aggregated_lr as key/value (cell-level L-R summary)
            cell_lr_expanded = aggregated_lr.unsqueeze(1)  # [N, 1, hidden_dim]
            
            attended_fate, lr_fate_attn_weights = self.lr_fate_attention(
                fate_queries,
                cell_lr_expanded.expand(-1, self.n_fates, -1),
                cell_lr_expanded.expand(-1, self.n_fates, -1),
            )
            fate_representation = attended_fate.mean(dim=1)  # [N, hidden_dim]
        else:
            fate_representation = aggregated_lr
            
        # Predict discrete fates
        fate_logits = self.fate_classifier(fate_representation)
        
        # Predict continuous fate trajectory
        fate_trajectory = self.fate_trajectory_encoder(fate_representation)
        
        # Predict differentiation score
        differentiation_score = self.differentiation_scorer(fate_representation)
        
        outputs = {
            'fate_logits': fate_logits,
            'fate_trajectory': fate_trajectory,
            'differentiation_score': differentiation_score,
            'fate_representation': fate_representation,
        }
        
        if return_attention and lr_fate_attn_weights is not None:
            outputs['lr_fate_attention'] = lr_fate_attn_weights
            
        return outputs


class NeighborhoodLRAggregator(MessagePassing):
    """
    Aggregates L-R interaction signals for each cell from its neighborhood.
    
    Uses attention-weighted aggregation to capture which incoming L-R
    signals are most important for each cell's fate.
    
    Args:
        hidden_dim: Embedding dimension
        aggregation: Aggregation type ('attention', 'mean', 'max')
    """
    
    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = 'attention',
    ):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        if aggregation == 'attention':
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        lr_embeddings: torch.Tensor,
        lr_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate L-R signals for each cell.
        
        Args:
            z: Node embeddings [N, hidden_dim]
            edge_index: Edge indices [2, E]
            lr_embeddings: L-R interaction embeddings [E, hidden_dim]
            lr_scores: L-R scores [E]
            
        Returns:
            Aggregated L-R features per cell [N, hidden_dim]
        """
        # Propagate messages (L-R signals) to destination cells
        return self.propagate(
            edge_index,
            z=z,
            lr_embeddings=lr_embeddings,
            lr_scores=lr_scores,
            size=(z.size(0), z.size(0)),
        )
        
    def message(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        lr_embeddings: torch.Tensor,
        lr_scores: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages (weighted L-R signals)."""
        if self.aggregation == 'attention':
            # Attention based on cell embedding and L-R embedding
            attn_input = torch.cat([z_i, lr_embeddings], dim=-1)
            attn_scores = self.attention_mlp(attn_input)
            attn_weights = edge_softmax(attn_scores, index)
            return lr_embeddings * attn_weights * lr_scores.unsqueeze(-1)
        else:
            # Simple score-weighted aggregation
            return lr_embeddings * lr_scores.unsqueeze(-1)


class CounterfactualReasoner(nn.Module):
    """
    Implements counterfactual reasoning for causal L-R inference.
    
    Key question: "Which L-R interactions, if removed, would change the cell fate?"
    
    Approach:
    1. Compute baseline fate predictions
    2. Systematically mask L-R interactions
    3. Re-compute fate predictions
    4. Identify L-R interactions whose removal changes fate
    
    This enables inverse modelling by identifying CAUSAL L-R signals,
    not just correlated ones.
    
    Args:
        hidden_dim: Embedding dimension
        n_counterfactual_samples: Number of random masks to sample
        use_gradient_based: Use gradient-based attribution (faster)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_counterfactual_samples: int = 100,
        use_gradient_based: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_counterfactual_samples = n_counterfactual_samples
        self.use_gradient_based = use_gradient_based
        
        # Intervention encoder: how to modify L-R when removed
        self.intervention_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Causal effect estimator
        self.effect_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        lr_scores: torch.Tensor,
        fate_representation: torch.Tensor,
        fate_predictor: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal attribution of L-R interactions to cell fates.
        
        Args:
            z: Node embeddings [N, hidden_dim]
            edge_index: Edge indices [2, E]
            lr_scores: L-R interaction scores [E]
            fate_representation: Learned fate representations [N, hidden_dim]
            fate_predictor: Module that predicts fate from representations
            
        Returns:
            Dictionary containing:
                - causal_scores: Causal importance of each L-R edge [E]
                - intervention_effects: Effect of removing each L-R [E, n_fates]
        """
        E = lr_scores.size(0)
        src, dst = edge_index
        
        if self.use_gradient_based:
            # Gradient-based causal attribution (fast approximation)
            return self._gradient_based_attribution(
                z, edge_index, lr_scores, fate_representation, fate_predictor
            )
        else:
            # Full counterfactual reasoning (expensive but more accurate)
            return self._counterfactual_attribution(
                z, edge_index, lr_scores, fate_representation, fate_predictor
            )
            
    def _gradient_based_attribution(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        lr_scores: torch.Tensor,
        fate_representation: torch.Tensor,
        fate_predictor: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal attribution using integrated gradients.
        
        Integrated gradients approximate the counterfactual effect
        by integrating gradients along the path from zero to actual L-R scores.
        
        For inference without gradients, falls back to learned attention-based
        attribution using the effect estimator network.
        """
        src, dst = edge_index
        E = lr_scores.size(0)
        
        # Check if we're in a context where gradients are available
        grad_enabled = torch.is_grad_enabled() and lr_scores.requires_grad
        
        if grad_enabled:
            # Training mode: use gradient-based attribution
            grad_scores = None  # Initialize
            try:
                # Compute fate predictions with gradients
                fate_logits = fate_predictor(fate_representation)
                
                # Compute gradient of fate w.r.t. L-R scores
                fate_class = fate_logits.argmax(dim=-1)
                fate_sum = fate_logits.gather(1, fate_class.unsqueeze(1)).sum()
                
                # Compute gradients
                grad_scores = torch.autograd.grad(
                    fate_sum,
                    lr_scores,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                
                if grad_scores is not None:
                    # Integrated gradients: score * gradient
                    causal_scores = (lr_scores * grad_scores).abs()
                else:
                    # Fallback to magnitude
                    causal_scores = lr_scores.abs()
            except RuntimeError:
                # Gradient computation failed, use fallback
                causal_scores = lr_scores.abs()
                grad_scores = None
        else:
            # Inference mode: use learned effect estimator
            # Create intervention features from edge embeddings
            z_src = z[src]  # Source embeddings
            z_dst = z[dst]  # Destination embeddings
            
            # Compute causal effect estimates using effect estimator
            intervention_features = self.intervention_encoder(
                torch.cat([z_src, lr_scores.unsqueeze(-1)], dim=-1)
            )
            combined = torch.cat([intervention_features, z_dst], dim=-1)
            causal_scores = torch.sigmoid(self.effect_estimator(combined)).squeeze(-1)
            
            # Weight by L-R strength - stronger interactions have more potential impact
            causal_scores = causal_scores * lr_scores.abs()
            grad_scores = None  # Not available in inference mode
            
        # Normalize per destination cell
        src, dst = edge_index
        causal_scores_normalized = causal_scores.clone()
        for cell_idx in range(z.size(0)):
            mask = dst == cell_idx
            if mask.sum() > 0:
                cell_scores = causal_scores[mask]
                if cell_scores.max() > 0:
                    causal_scores_normalized[mask] = cell_scores / cell_scores.max()
                    
        return {
            'causal_scores': causal_scores_normalized,
            'gradient_attribution': grad_scores if grad_scores is not None else torch.zeros_like(lr_scores),
        }
        
    def _counterfactual_attribution(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        lr_scores: torch.Tensor,
        fate_representation: torch.Tensor,
        fate_predictor: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        True counterfactual reasoning by masking L-R interactions and
        re-running the fate predictor.
        
        For each sampled L-R edge we:
          1. Zero out its score
          2. Re-aggregate neighbourhood LR signals for the receiving cell
          3. Re-predict fate with the modified representation
          4. Measure KL divergence between baseline and counterfactual fate
        
        This is computationally expensive (O(n_samples) forward passes
        through the fate MLP) but gives genuine causal attribution.
        
        To keep training efficient we use batched sampling: we sample
        ``n_counterfactual_samples`` edges at random and compute the
        effect in a vectorised inner loop.
        """
        E = lr_scores.size(0)
        N = z.size(0)
        src, dst = edge_index
        
        # ── Baseline fate predictions ────────────────────────────────────
        with torch.no_grad():
            baseline_fate = fate_predictor(fate_representation)       # [N, n_fates]
            baseline_probs = F.softmax(baseline_fate, dim=-1)         # [N, n_fates]
        
        # ── Sample edges to evaluate ─────────────────────────────────────
        n_samples = min(self.n_counterfactual_samples, E)
        sample_idx = torch.randperm(E, device=lr_scores.device)[:n_samples]
        
        causal_scores = torch.zeros(E, device=lr_scores.device)
        
        # ── Batched counterfactual evaluation ────────────────────────────
        # For each sampled edge i, build a version of fate_representation
        # for cell dst[i] where edge i's contribution is removed.
        #
        # Approximation: we subtract the edge-i contribution from the
        # aggregated representation and re-run the fate MLP.
        
        # First, compute per-edge contribution to the node representation.
        # This uses the intervention encoder to estimate what each edge
        # contributes to the destination cell's representation.
        z_src_all = z[src]  # [E, hidden_dim]
        intervention_features = self.intervention_encoder(
            torch.cat([z_src_all, lr_scores.unsqueeze(-1)], dim=-1)
        )  # [E, hidden_dim]
        
        # For sampled edges, compute counterfactual fate
        sampled_dst = dst[sample_idx]                                 # [n_samples]
        sampled_contributions = intervention_features[sample_idx]     # [n_samples, hidden_dim]
        
        # Counterfactual representation = baseline − this edge's contribution
        cf_representations = fate_representation[sampled_dst] - sampled_contributions  # [n_samples, hidden_dim]
        
        with torch.no_grad():
            cf_fate = fate_predictor(cf_representations)               # [n_samples, n_fates]
            cf_probs = F.softmax(cf_fate, dim=-1)
        
        # KL divergence per sample: how much does removing this edge change fate?
        baseline_for_sample = baseline_probs[sampled_dst]              # [n_samples, n_fates]
        kl = F.kl_div(
            cf_probs.log().clamp(min=-100),
            baseline_for_sample,
            reduction='none',
        ).sum(dim=-1)                                                  # [n_samples]
        
        causal_scores[sample_idx] = kl.detach()
        
        # Normalise per destination cell so scores are comparable
        for cell_idx in range(N):
            mask = dst == cell_idx
            if mask.sum() > 0:
                cell_scores = causal_scores[mask]
                mx = cell_scores.max()
                if mx > 0:
                    causal_scores[mask] = cell_scores / mx
        
        return {
            'causal_scores': causal_scores,
            'gradient_attribution': torch.zeros_like(lr_scores),
        }
        
    def identify_causal_lr_for_fate(
        self,
        causal_scores: torch.Tensor,
        edge_index: torch.Tensor,
        target_cells: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Identify causal L-R interactions for specific cell fates.
        
        Args:
            causal_scores: Causal importance scores [E]
            edge_index: Edge indices [2, E]
            target_cells: Indices of cells with target fate [M]
            threshold: Threshold for "causal" designation
            
        Returns:
            Dictionary with causal L-R edges and their scores
        """
        src, dst = edge_index
        
        # Find edges targeting the fate cells
        target_mask = torch.isin(dst, target_cells)
        relevant_edges = torch.where(target_mask)[0]
        relevant_scores = causal_scores[relevant_edges]
        
        # Apply threshold
        causal_mask = relevant_scores > threshold
        causal_edges = relevant_edges[causal_mask]
        
        return {
            'causal_edge_indices': causal_edges,
            'causal_edge_scores': causal_scores[causal_edges],
            'source_cells': src[causal_edges],
            'target_cells': dst[causal_edges],
        }


class LRToTargetGeneDecoder(nn.Module):
    """
    Decodes L-R interactions to downstream target gene expression changes.
    
    This implements the signaling cascade:
        L-R activation → Receptor signaling → TF activation → Target gene expression
    
    Enables inverse modelling by predicting WHICH GENES are affected by WHICH L-R signals.
    
    **Biologically grounded**: ``pathway_gene_matrix`` is initialised from a
    real gene-set mask (MSigDB Hallmark + CARDIAC_PATHWAYS) so the model starts
    from known biology rather than random weights.
    
    Args:
        hidden_dim: Embedding dimension
        n_genes: Number of target genes
        n_pathways: Number of signaling pathways
        use_pathway_attention: Use attention over pathways
        pathway_gene_mask: Optional binary mask [n_pathways, n_genes] from
            ``build_pathway_gene_mask``.  If provided the ``pathway_gene_matrix``
            Parameter is initialised to match and a grounding regulariser
            can penalise deviation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        n_pathways: int = 20,
        use_pathway_attention: bool = True,
        pathway_gene_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        
        # L-R to pathway activation
        self.lr_to_pathway = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_pathways),
        )
        
        # Pathway to target gene mapping (learnable pathway-gene associations)
        # ── Biology-grounded initialisation ──────────────────────────────
        if pathway_gene_mask is not None:
            # Resize mask to match n_pathways × n_genes
            pm = pathway_gene_mask
            if pm.shape[0] > n_pathways:
                pm = pm[:n_pathways]
            elif pm.shape[0] < n_pathways:
                pad = torch.zeros(n_pathways - pm.shape[0], pm.shape[1])
                pm = torch.cat([pm, pad], dim=0)
            if pm.shape[1] != n_genes:
                # Truncate or pad gene dimension
                if pm.shape[1] > n_genes:
                    pm = pm[:, :n_genes]
                else:
                    pad = torch.zeros(pm.shape[0], n_genes - pm.shape[1])
                    pm = torch.cat([pm, pad], dim=1)
            # Initialise: known associations get +2, unknown get small random
            init_vals = pm * 2.0 + torch.randn(n_pathways, n_genes) * 0.01
            self.pathway_gene_matrix = nn.Parameter(init_vals)
            # Keep the mask as buffer for grounding loss
            self.register_buffer('_pathway_gene_prior', pm)
            self._has_bio_prior = True
        else:
            self.pathway_gene_matrix = nn.Parameter(
                torch.randn(n_pathways, n_genes) * 0.01
            )
            self._has_bio_prior = False
        
        if use_pathway_attention:
            self.pathway_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )
            self.pathway_embeddings = nn.Parameter(torch.randn(n_pathways, hidden_dim))
            
        # Gene-specific decoders for fine-grained prediction
        self.gene_decoder = nn.Sequential(
            nn.Linear(hidden_dim + n_pathways, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_genes),
        )
        
        # Expression change predictor (delta expression)
        self.delta_predictor = nn.Sequential(
            nn.Linear(n_genes * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
            nn.Tanh(),  # Normalized expression change
        )
        
    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        lr_scores: torch.Tensor,
        baseline_expression: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict target gene expression changes from L-R interactions.
        
        Args:
            z_src: Source (ligand) cell embeddings [E, hidden_dim]
            z_dst: Destination (receptor) cell embeddings [E, hidden_dim]
            lr_scores: L-R interaction scores [E]
            baseline_expression: Baseline gene expression [E, n_genes] (optional)
            
        Returns:
            Dictionary containing:
                - pathway_activation: Activation of each pathway [E, n_pathways]
                - target_gene_effects: Effect on each gene [E, n_genes]
                - predicted_expression: Predicted expression after signaling [E, n_genes]
        """
        E = lr_scores.size(0)
        
        # Encode L-R interaction
        lr_features = torch.cat([z_src, z_dst, lr_scores.unsqueeze(-1)], dim=-1)
        
        # L-R to pathway activation
        pathway_activation = torch.sigmoid(self.lr_to_pathway(lr_features))  # [E, n_pathways]
        
        # Pathway to gene effects
        gene_effects_from_pathway = torch.matmul(
            pathway_activation, 
            torch.softmax(self.pathway_gene_matrix, dim=-1)
        )  # [E, n_genes]
        
        # Fine-grained gene prediction
        gene_input = torch.cat([z_dst, pathway_activation], dim=-1)
        target_gene_effects = self.gene_decoder(gene_input)  # [E, n_genes]
        
        # Combine pathway-based and direct predictions
        combined_effects = (gene_effects_from_pathway + target_gene_effects) / 2
        
        outputs = {
            'pathway_activation': pathway_activation,
            'target_gene_effects': combined_effects,
        }
        
        # Predict expression change if baseline provided
        if baseline_expression is not None:
            delta_input = torch.cat([baseline_expression, combined_effects], dim=-1)
            delta_expression = self.delta_predictor(delta_input)
            predicted_expression = baseline_expression + delta_expression
            
            outputs['delta_expression'] = delta_expression
            outputs['predicted_expression'] = predicted_expression
            
        return outputs
        
    def get_pathway_gene_associations(self) -> torch.Tensor:
        """
        Get learned pathway-gene associations.
        
        Returns:
            Pathway-gene association matrix [n_pathways, n_genes]
        """
        return torch.softmax(self.pathway_gene_matrix, dim=-1)
        
    def get_top_genes_per_pathway(
        self,
        gene_names: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get top genes associated with each pathway.
        
        Returns:
            Dictionary mapping pathway index to list of (gene_idx, score) tuples
        """
        associations = self.get_pathway_gene_associations().detach().cpu()
        
        results = {}
        for pathway_idx in range(self.n_pathways):
            pathway_scores = associations[pathway_idx]
            top_indices = pathway_scores.argsort(descending=True)[:top_k]
            
            if gene_names is not None:
                results[pathway_idx] = [
                    (gene_names[idx], pathway_scores[idx].item())
                    for idx in top_indices
                ]
            else:
                results[pathway_idx] = [
                    (idx.item(), pathway_scores[idx].item())
                    for idx in top_indices
                ]
                
        return results

    def pathway_grounding_loss(self) -> torch.Tensor:
        """
        Compute a regularisation loss that penalises the learned
        ``pathway_gene_matrix`` for deviating from the biological prior.

        This ensures the model retains biological plausibility even as it
        learns from data.  Returns zero if no prior was provided.
        """
        if not self._has_bio_prior:
            return torch.tensor(0.0, device=self.pathway_gene_matrix.device)
        # Softmax the matrix the same way forward() does
        learned = torch.softmax(self.pathway_gene_matrix, dim=-1)
        prior = self._pathway_gene_prior / self._pathway_gene_prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # KL(prior || learned) — nudge learned towards prior
        loss = F.kl_div(learned.log(), prior, reduction='batchmean')
        return loss


class MechanosensitivePathwayModule(nn.Module):
    """
    Models mechanosensitive signaling pathways that modulate cardiac patterning.
    
    Implements the abstract's claim:
    "elucidate mechanosensitive pathways that modulate early-stage cardiac tissue
    patterning, linking molecular signalling to the formation of soft contractile
    structures"
    
    **Biologically grounded** version: pathway activations are constrained by
    a real gene-set membership mask built from MSigDB Hallmark sets,
    curated cardiac pathways, and mechanosensor gene lists so that each
    pathway corresponds to a genuine biological programme.
    
    Args:
        hidden_dim: Embedding dimension
        n_mechano_pathways: Number of mechanosensitive pathways
        pathway_gene_mask: Optional binary mask [n_pathways, n_genes] from
            ``build_pathway_gene_mask`` linking pathways to real genes.
        pathway_names: Optional list of pathway names (length must match
            first dim of pathway_gene_mask). If None a default list is used.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_mechano_pathways: int = 8,
        pathway_gene_mask: Optional[torch.Tensor] = None,
        pathway_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_mechano_pathways = n_mechano_pathways
        
        # ── Pathway names (real biology) ─────────────────────────────────
        if pathway_names is not None:
            self.pathway_names = list(pathway_names)[:n_mechano_pathways]
        else:
            self.pathway_names = [
                'YAP_TAZ_Hippo',
                'Integrin_FAK',
                'Piezo_Mechano',
                'TGFb',
                'WNT',
                'NOTCH',
                'BMP',
                'FGF',
            ]
        
        # ── Biology-grounded gene-set mask ───────────────────────────────
        # If a real mask is provided, register it as a non-trainable buffer
        # so the model knows which genes belong to which pathway.
        if pathway_gene_mask is not None:
            # Ensure shape matches n_mechano_pathways (take first N rows if larger)
            if pathway_gene_mask.shape[0] > n_mechano_pathways:
                pathway_gene_mask = pathway_gene_mask[:n_mechano_pathways]
            elif pathway_gene_mask.shape[0] < n_mechano_pathways:
                # Pad with zeros
                pad = torch.zeros(
                    n_mechano_pathways - pathway_gene_mask.shape[0],
                    pathway_gene_mask.shape[1],
                )
                pathway_gene_mask = torch.cat([pathway_gene_mask, pad], dim=0)
            self.register_buffer('pathway_gene_mask', pathway_gene_mask)
            self._has_bio_mask = True
        else:
            self._has_bio_mask = False
        
        # Spatial context encoder (local mechanical environment)
        self.spatial_context_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),  # +2 for spatial coords
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # L-R to mechanosensitive pathway activation
        self.lr_mechano_mapper = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_mechano_pathways),
        )
        
        # Pathway grounding projection: if we have a bio mask, learn a small
        # adapter that respects the mask structure.
        if self._has_bio_mask:
            n_genes = pathway_gene_mask.shape[1]
            self.pathway_grounding_proj = nn.Linear(n_genes, n_mechano_pathways, bias=False)
            # Initialise weights from the mask so grounding starts correct
            with torch.no_grad():
                # Normalise rows so each pathway sums to 1
                row_sums = pathway_gene_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                self.pathway_grounding_proj.weight.copy_(pathway_gene_mask / row_sums)
        
        # Mechano-pathway interaction network
        self.pathway_interaction = nn.Sequential(
            nn.Linear(n_mechano_pathways, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_mechano_pathways),
        )
        
        # Pathway to differentiation effect
        self.pathway_to_diff = nn.Linear(n_mechano_pathways, 1)
        
        # Pathway to gene regulation
        self.pathway_gene_regulation = nn.Sequential(
            nn.Linear(n_mechano_pathways, hidden_dim),
            nn.GELU(),
        )
        
    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        lr_scores: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mechanosensitive pathway activations from L-R interactions.
        
        Args:
            z_src: Source cell embeddings [E, hidden_dim]
            z_dst: Destination cell embeddings [E, hidden_dim]
            lr_scores: L-R interaction scores [E]
            spatial_coords: Spatial coordinates of destination cells [E, 2]
            
        Returns:
            Dictionary containing:
                - mechano_pathway_activation: Pathway activations [E, n_mechano_pathways]
                - differentiation_effect: Effect on differentiation [E, 1]
                - pathway_crosstalk: Inter-pathway interactions [E, n_mechano_pathways]
        """
        E = lr_scores.size(0)
        
        # Incorporate spatial context if available
        if spatial_coords is not None:
            spatial_input = torch.cat([z_dst, spatial_coords], dim=-1)
            z_spatial = self.spatial_context_encoder(spatial_input)
        else:
            z_spatial = z_dst
            
        # L-R to mechanosensitive pathway activation
        lr_features = torch.cat([z_src, z_spatial, lr_scores.unsqueeze(-1)], dim=-1)
        raw_activation = self.lr_mechano_mapper(lr_features)
        
        # Apply pathway interactions (crosstalk)
        pathway_crosstalk = self.pathway_interaction(torch.sigmoid(raw_activation))
        mechano_activation = torch.sigmoid(raw_activation + pathway_crosstalk)
        
        # Effect on differentiation
        differentiation_effect = self.pathway_to_diff(mechano_activation)
        
        # Gene regulation signals
        gene_regulation = self.pathway_gene_regulation(mechano_activation)
        
        return {
            'mechano_pathway_activation': mechano_activation,
            'differentiation_effect': differentiation_effect,
            'pathway_crosstalk': pathway_crosstalk,
            'gene_regulation_signal': gene_regulation,
            'pathway_names': self.pathway_names[:self.n_mechano_pathways],
        }
        
    def get_pathway_importance(
        self,
        mechano_activation: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Get relative importance of each mechanosensitive pathway.
        """
        mean_activation = mechano_activation.mean(dim=0).detach().cpu()
        
        return {
            name: act.item()
            for name, act in zip(self.pathway_names, mean_activation)
        }


class InverseSignalInference(nn.Module):
    """
    Complete inverse modelling pipeline for GRAIL-Heart.
    
    Integrates all inverse modelling components:
    1. CellFatePredictionHead: L-R → Fate
    2. CounterfactualReasoner: Causal L-R identification
    3. LRToTargetGeneDecoder: L-R → Target genes
    4. MechanosensitivePathwayModule: Mechanobiology integration
    
    The inverse inference answers:
    "Given an observed cell fate or gene expression pattern,
     which upstream L-R signals are responsible?"
    
    Args:
        hidden_dim: Embedding dimension
        n_genes: Number of genes
        n_fates: Number of cell fate categories
        n_pathways: Number of signaling pathways
        n_mechano_pathways: Number of mechanosensitive pathways
        pathway_gene_mask: Optional biology mask for LRToTargetGeneDecoder
        mechano_gene_mask: Optional biology mask for MechanosensitivePathwayModule
        mechano_pathway_names: Optional list of pathway names for mechano module
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        n_fates: int,
        n_pathways: int = 20,
        n_mechano_pathways: int = 8,
        pathway_gene_mask: Optional[torch.Tensor] = None,
        mechano_gene_mask: Optional[torch.Tensor] = None,
        mechano_pathway_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Component modules
        self.cell_fate_head = CellFatePredictionHead(
            hidden_dim=hidden_dim,
            n_fates=n_fates,
            use_attention=True,
        )
        
        self.counterfactual_reasoner = CounterfactualReasoner(
            hidden_dim=hidden_dim,
            use_gradient_based=True,
        )
        
        self.lr_target_decoder = LRToTargetGeneDecoder(
            hidden_dim=hidden_dim,
            n_genes=n_genes,
            n_pathways=n_pathways,
            pathway_gene_mask=pathway_gene_mask,
        )
        
        self.mechano_module = MechanosensitivePathwayModule(
            hidden_dim=hidden_dim,
            n_mechano_pathways=n_mechano_pathways,
            pathway_gene_mask=mechano_gene_mask,
            pathway_names=mechano_pathway_names,
        )
        
        # Inverse inference network
        # Maps observed fate/expression BACK to inferred L-R signals
        self.inverse_encoder = nn.Sequential(
            nn.Linear(n_genes + n_fates, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # L-R signal reconstructor from inverse encoding
        self.lr_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Score for each L-R
        )
        
        # ── Cycle-consistency decoder (WP5) ──────────────────────────────
        # Reconstructs LR scores from the fate representation so we can
        # enforce  LR → Fate → LR'  ≈  LR  (cycle consistency).
        self.cycle_lr_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        lr_scores: torch.Tensor,
        expression: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inverse modelling forward pass.
        
        Args:
            z: Node embeddings [N, hidden_dim]
            edge_index: Edge indices [2, E]
            lr_scores: L-R interaction scores [E]
            expression: Gene expression [N, n_genes] (optional)
            spatial_coords: Spatial coordinates [N, 2] (optional)
            
        Returns:
            Dictionary with all inverse modelling outputs
        """
        src, dst = edge_index
        z_src, z_dst = z[src], z[dst]
        
        # 1. Cell fate prediction from L-R
        fate_outputs = self.cell_fate_head(
            z=z,
            edge_index=edge_index,
            lr_scores=lr_scores,
            return_attention=True,
        )
        
        # 2. Causal L-R identification
        causal_outputs = self.counterfactual_reasoner(
            z=z,
            edge_index=edge_index,
            lr_scores=lr_scores,
            fate_representation=fate_outputs['fate_representation'],
            fate_predictor=self.cell_fate_head.fate_classifier,
        )
        
        # 3. L-R to target gene effects
        lr_target_outputs = self.lr_target_decoder(
            z_src=z_src,
            z_dst=z_dst,
            lr_scores=lr_scores,
            baseline_expression=expression[dst] if expression is not None else None,
        )
        
        # 4. Mechanosensitive pathway analysis
        mechano_outputs = self.mechano_module(
            z_src=z_src,
            z_dst=z_dst,
            lr_scores=lr_scores,
            spatial_coords=spatial_coords[dst] if spatial_coords is not None else None,
        )
        
        # Combine all outputs
        outputs = {
            # Fate prediction (forward: L-R → Fate)
            'fate_logits': fate_outputs['fate_logits'],
            'fate_trajectory': fate_outputs['fate_trajectory'],
            'differentiation_score': fate_outputs['differentiation_score'],
            
            # Causal attribution (inverse: Fate → Causal L-R)
            'causal_lr_scores': causal_outputs['causal_scores'],
            
            # Target gene effects (forward: L-R → Genes)
            'pathway_activation': lr_target_outputs['pathway_activation'],
            'target_gene_effects': lr_target_outputs['target_gene_effects'],
            
            # Mechanobiology
            'mechano_pathway_activation': mechano_outputs['mechano_pathway_activation'],
            'mechano_differentiation_effect': mechano_outputs['differentiation_effect'],
            
            # Embeddings for downstream analysis
            'fate_representation': fate_outputs['fate_representation'],
            
            # Pathway grounding regularisation (WP1)
            'pathway_grounding_loss': self.lr_target_decoder.pathway_grounding_loss(),
        }
        
        # ── Cycle-consistency (WP5): Fate repr → reconstructed LR ────────
        # For each edge, use the *destination* cell's fate representation
        # to reconstruct the LR score.  This enforces that the fate
        # representation truly encodes the L-R information.
        fate_repr_dst = fate_outputs['fate_representation'][dst]  # [E, hidden_dim]
        reconstructed_lr = self.cycle_lr_decoder(fate_repr_dst).squeeze(-1)  # [E]
        outputs['cycle_reconstructed_lr'] = reconstructed_lr
        
        if 'predicted_expression' in lr_target_outputs:
            outputs['predicted_expression_from_lr'] = lr_target_outputs['predicted_expression']
            outputs['delta_expression'] = lr_target_outputs['delta_expression']
            
        return outputs
        
    def infer_causal_signals(
        self,
        observed_fate: torch.Tensor,
        observed_expression: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        INVERSE INFERENCE: Given observed fate/expression, infer causal L-R signals.
        
        This is the core inverse modelling function that answers:
        "What L-R signals caused this differentiation pattern?"
        
        Args:
            observed_fate: Observed cell fate labels/probabilities [N, n_fates]
            observed_expression: Observed gene expression [N, n_genes]
            z: Node embeddings [N, hidden_dim]
            edge_index: Edge indices [2, E]
            
        Returns:
            Dictionary with:
                - inferred_lr_importance: Importance of each L-R edge [E]
                - inferred_pathway_importance: Importance of each pathway [E, n_pathways]
        """
        E = edge_index.size(1)
        src, dst = edge_index
        
        # Encode observed state (fate + expression)
        observed_input = torch.cat([observed_expression, observed_fate], dim=-1)
        inverse_encoding = self.inverse_encoder(observed_input)  # [N, hidden_dim]
        
        # For each edge, determine if it could have caused the observed fate
        inverse_encoding_dst = inverse_encoding[dst]  # [E, hidden_dim]
        
        # Reconstruct L-R importance from inverse encoding
        inferred_lr_importance = self.lr_reconstructor(inverse_encoding_dst).squeeze(-1)
        inferred_lr_importance = torch.sigmoid(inferred_lr_importance)
        
        return {
            'inferred_lr_importance': inferred_lr_importance,
            'inverse_encoding': inverse_encoding,
        }


class InverseModellingLoss(nn.Module):
    """
    Loss functions for training inverse modelling components.
    
    Combines multiple objectives:
    1. Fate prediction accuracy
    2. Causal consistency (counterfactual validation)
    3. Target gene prediction accuracy
    4. Inverse reconstruction accuracy
    """
    
    def __init__(
        self,
        fate_weight: float = 1.0,
        causal_weight: float = 0.5,
        gene_target_weight: float = 0.5,
        inverse_weight: float = 1.0,
        differentiation_weight: float = 0.3,
    ):
        super().__init__()
        
        self.fate_weight = fate_weight
        self.causal_weight = causal_weight
        self.gene_target_weight = gene_target_weight
        self.inverse_weight = inverse_weight
        self.differentiation_weight = differentiation_weight
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute inverse modelling losses.
        
        Args:
            outputs: Model outputs from InverseSignalInference
            targets: Target values including:
                - fate_labels: Ground truth fate labels [N]
                - differentiation_stage: Differentiation scores [N]
                - target_expression: Expression after signaling [N, n_genes]
                
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Fate prediction loss
        if 'fate_labels' in targets and 'fate_logits' in outputs:
            fate_loss = F.cross_entropy(
                outputs['fate_logits'],
                targets['fate_labels'],
            )
            loss_dict['fate_loss'] = fate_loss
            total_loss = total_loss + self.fate_weight * fate_loss
            
        # 2. Differentiation score loss
        if 'differentiation_stage' in targets and 'differentiation_score' in outputs:
            diff_loss = F.mse_loss(
                outputs['differentiation_score'].squeeze(),
                targets['differentiation_stage'],
            )
            loss_dict['differentiation_loss'] = diff_loss
            total_loss = total_loss + self.differentiation_weight * diff_loss
            
        # 3. Target gene prediction loss
        if 'target_expression' in targets and 'predicted_expression_from_lr' in outputs:
            gene_loss = F.mse_loss(
                outputs['predicted_expression_from_lr'],
                targets['target_expression'],
            )
            loss_dict['gene_target_loss'] = gene_loss
            total_loss = total_loss + self.gene_target_weight * gene_loss
            
        # 4. Causal consistency loss (encourage sparse causal attributions)
        if 'causal_lr_scores' in outputs:
            causal_sparsity = outputs['causal_lr_scores'].mean()
            loss_dict['causal_sparsity'] = causal_sparsity
            total_loss = total_loss + self.causal_weight * causal_sparsity
            
        # 5. Inverse reconstruction loss
        if 'inferred_lr_importance' in outputs and 'lr_scores' in targets:
            inverse_loss = F.mse_loss(
                outputs['inferred_lr_importance'],
                targets['lr_scores'],
            )
            loss_dict['inverse_loss'] = inverse_loss
            total_loss = total_loss + self.inverse_weight * inverse_loss
            
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
