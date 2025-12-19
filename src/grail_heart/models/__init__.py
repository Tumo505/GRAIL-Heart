"""
GRAIL-Heart Models Module

Neural network architectures for cardiac spatial transcriptomics analysis.
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
]
