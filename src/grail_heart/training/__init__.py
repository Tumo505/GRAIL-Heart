"""
GRAIL-Heart Training Module

Training utilities for the GRAIL-Heart framework.
"""

from .losses import (
    LRInteractionLoss,
    ReconstructionLoss,
    CellTypeLoss,
    SignalingNetworkLoss,
    GRAILHeartLoss,
    create_lr_targets,
    create_negative_samples,
)

from .metrics import (
    compute_lr_metrics,
    compute_reconstruction_metrics,
    compute_cell_type_metrics,
    compute_signaling_metrics,
    MetricTracker,
    print_metrics,
)

from .trainer import (
    GRAILHeartTrainer,
    create_optimizer,
    create_scheduler,
)

from .contrastive import (
    ContrastiveLoss,
    ProjectionHead,
    AugmentationModule,
    SimCLRLoss,
)


__all__ = [
    # Losses
    'LRInteractionLoss',
    'ReconstructionLoss',
    'CellTypeLoss',
    'SignalingNetworkLoss',
    'GRAILHeartLoss',
    'create_lr_targets',
    'create_negative_samples',
    # Metrics
    'compute_lr_metrics',
    'compute_reconstruction_metrics',
    'compute_cell_type_metrics',
    'compute_signaling_metrics',
    'MetricTracker',
    'print_metrics',
    # Trainer
    'GRAILHeartTrainer',
    'create_optimizer',
    'create_scheduler',
]
