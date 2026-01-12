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

from . import data
from . import models
from . import training
from . import evaluation
from . import utils

# Convenient imports
from .models import GRAILHeart
from .inference import GRAILHeartPredictor, load_pretrained

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
