"""
GRAIL-Heart Data Module

Data loading and preprocessing for spatial transcriptomics.
"""

__all__ = [
    # Datasets
    'SpatialTranscriptomicsDataset',
    'CardiacDataModule',
    'load_heart_cell_atlas',
    'load_nature_genetics_visium',
    # Graph Builders
    'SpatialGraphBuilder',
    'MultiResolutionGraphBuilder',
    'LRGraphBuilder',
    'build_graph_batch',
    # L-R Database
    'LigandReceptorDatabase',
    'CellPhoneDBRunner',
    # Expanded L-R Database
    'get_expanded_lr_database',
    'filter_to_expressed_genes',
]


def __getattr__(name):
    """Lazy import mechanism to avoid loading all submodules at import time."""
    if name in ('SpatialTranscriptomicsDataset', 'CardiacDataModule', 
                'load_heart_cell_atlas', 'load_nature_genetics_visium'):
        from .datasets import (
            SpatialTranscriptomicsDataset, CardiacDataModule,
            load_heart_cell_atlas, load_nature_genetics_visium
        )
        return locals()[name]
    
    if name in ('SpatialGraphBuilder', 'MultiResolutionGraphBuilder',
                'LRGraphBuilder', 'build_graph_batch'):
        from .graph_builder import (
            SpatialGraphBuilder, MultiResolutionGraphBuilder,
            LRGraphBuilder, build_graph_batch
        )
        return locals()[name]
    
    if name in ('LigandReceptorDatabase', 'CellPhoneDBRunner'):
        from .lr_database import LigandReceptorDatabase, CellPhoneDBRunner
        return locals()[name]
    
    if name in ('get_expanded_lr_database', 'filter_to_expressed_genes'):
        from .expanded_lr_database import get_expanded_lr_database, filter_to_expressed_genes
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
