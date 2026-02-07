# GRAIL-Heart: Graph-based Reconstruction of Artificial Intercellular Links
"""
A Graph Neural Network framework for inferring functional signalling networks 
from spatial transcriptomics data in cardiac tissue.

Features:
- Forward Modeling: Expression → L-R interaction predictions
- Inverse Modeling: Observed fates → Causal L-R signals
- Mechanosensitive pathway analysis
- Interactive visualization

Example:
    >>> from grail_heart import GRAILHeart, load_pretrained
    >>> model = load_pretrained()
    >>> predictions = model.predict("my_cardiac_data.h5ad")
    >>> print(predictions.top_lr_pairs)
"""

__version__ = "1.0.0"
__author__ = "GRAIL-Heart Team"

# Lazy imports to avoid circular dependencies
# Submodules are loaded on first access
def __getattr__(name):
    if name == "data":
        from . import data
        return data
    if name == "models":
        from . import models
        return models
    if name == "training":
        from . import training
        return training
    if name == "evaluation":
        from . import evaluation
        return evaluation
    if name == "utils":
        from . import utils
        return utils
    if name == "GRAILHeart":
        from .models import GRAILHeart
        return GRAILHeart
    if name == "GRAILHeartPredictor":
        from .inference import GRAILHeartPredictor
        return GRAILHeartPredictor
    if name == "load_pretrained":
        from .inference import load_pretrained
        return load_pretrained
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GRAILHeart",
    "GRAILHeartPredictor",
    "load_pretrained",
    "data",
    "models",
    "training",
    "evaluation",
    "utils",
]
