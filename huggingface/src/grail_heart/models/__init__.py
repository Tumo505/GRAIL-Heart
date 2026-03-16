"""
GRAIL-Heart Models Module

Neural network architectures for cardiac spatial transcriptomics analysis.

Includes both FORWARD and INVERSE modelling components:
- FORWARD: Expression → L-R predictions, cell fates
- INVERSE: Observed fate/phenotype → Inferred causal L-R signals
"""

from .encoders import (
    GeneEncoder,
    VariationalGeneEncoder,
    SpatialPositionEncoder,
    CellTypeEncoder,
    MultiModalEncoder,
)

from .gat_layers import (
    EdgeTypeGATConv,
    MultiHeadGATLayer,
    GATStack,
)

from .predictors import (
    LRInteractionPredictor,
    SignalingNetworkPredictor,
    CellTypePredictor,
    GeneExpressionDecoder,
    MultiTaskHead,
)

from .grail_heart import (
    GRAILHeart,
    GRAILHeartLite,
    create_grail_heart,
)

# Inverse Modelling Components (NEW)
from .inverse_modelling import (
    InverseSignalInference,
    CellFatePredictionHead,
    CounterfactualReasoner,
    LRToTargetGeneDecoder,
    MechanosensitivePathwayModule,
    InverseModellingLoss,
)


__all__ = [
    # Encoders
    'GeneEncoder',
    'VariationalGeneEncoder',
    'SpatialPositionEncoder',
    'CellTypeEncoder',
    'MultiModalEncoder',
    # GAT Layers
    'EdgeTypeGATConv',
    'MultiHeadGATLayer',
    'GATStack',
    # Predictors
    'LRInteractionPredictor',
    'SignalingNetworkPredictor',
    'CellTypePredictor',
    'GeneExpressionDecoder',
    'MultiTaskHead',
    # Main Models
    'GRAILHeart',
    'GRAILHeartLite',
    'create_grail_heart',
    # Inverse Modelling (NEW)
    'InverseSignalInference',
    'CellFatePredictionHead',
    'CounterfactualReasoner',
    'LRToTargetGeneDecoder',
    'MechanosensitivePathwayModule',
    'InverseModellingLoss',
]
