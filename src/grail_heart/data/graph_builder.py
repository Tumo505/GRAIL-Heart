"""
Graph Builder for Spatial Transcriptomics

Constructs graphs from spatial coordinates where cells are nodes 
and edges represent spatial proximity or ligand-receptor interactions.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class SpatialGraphBuilder:
    """
    Builds graphs from spatial transcriptomics data.
    
    Supports multiple graph construction methods:
    - k-nearest neighbors (kNN)
    - radius-based neighbors
    - Delaunay triangulation
    - Combined spatial + L-R edges
    
    Args:
        method: Graph construction method ('knn', 'radius', 'delaunay')
        k: Number of neighbors for kNN
        radius: Radius threshold for radius-based graph
        self_loops: Whether to include self-loops
        symmetric: Whether to make edges bidirectional
    """
    
    def __init__(
        self,
        method: str = 'knn',
        k: int = 6,
        radius: float = 0.1,
        self_loops: bool = False,
        symmetric: bool = True,
    ):
        self.method = method
        self.k = k
        self.radius = radius
        self.self_loops = self_loops
        self.symmetric = symmetric
        
    def build_graph(
        self,
        expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        cell_types: Optional[torch.Tensor] = None,
        lr_edges: Optional[torch.Tensor] = None,
    ) -> Data:
        """
        Build a PyTorch Geometric graph from spatial transcriptomics data.
        
        Args:
            expression: Gene expression matrix [n_cells, n_genes]
            spatial_coords: Spatial coordinates [n_cells, 2]
            cell_types: Cell type labels [n_cells]
            lr_edges: Optional ligand-receptor edges [2, n_lr_edges]
            
        Returns:
            PyTorch Geometric Data object
        """
        n_cells = expression.shape[0]
        
        # Build spatial edges
        if self.method == 'knn':
            edge_index = self._build_knn_graph(spatial_coords)
        elif self.method == 'radius':
            edge_index = self._build_radius_graph(spatial_coords)
        elif self.method == 'delaunay':
            edge_index = self._build_delaunay_graph(spatial_coords)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Compute edge weights based on distance
        edge_weight = self._compute_edge_weights(spatial_coords, edge_index)
        
        # Create edge type labels (0 = spatial, 1 = L-R)
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
        
        # Add L-R edges if provided
        if lr_edges is not None:
            edge_index = torch.cat([edge_index, lr_edges], dim=1)
            lr_weights = torch.ones(lr_edges.shape[1])
            edge_weight = torch.cat([edge_weight, lr_weights])
            lr_types = torch.ones(lr_edges.shape[1], dtype=torch.long)
            edge_type = torch.cat([edge_type, lr_types])
        
        # Build PyG Data object
        data = Data(
            x=expression,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_type=edge_type,
            pos=spatial_coords,
            num_nodes=n_cells,
        )
        
        if cell_types is not None:
            data.y = cell_types
            
        return data
    
    def build_from_dataset(
        self,
        dataset,
        lr_edges: Optional[torch.Tensor] = None,
    ) -> Data:
        """
        Build graph from SpatialTranscriptomicsDataset.
        
        This method extracts all available information from the dataset,
        including differentiation stage for inverse modelling.
        
        Args:
            dataset: SpatialTranscriptomicsDataset instance
            lr_edges: Optional ligand-receptor interaction edges
            
        Returns:
            PyG Data object with all available attributes
        """
        data = self.build_graph(
            expression=dataset.expression,
            spatial_coords=dataset.spatial_coords,
            cell_types=dataset.cell_types,
            lr_edges=lr_edges,
        )
        
        # Add differentiation stage if available
        if hasattr(dataset, 'differentiation_stage') and dataset.differentiation_stage is not None:
            data.differentiation_stage = dataset.differentiation_stage
            
        # Add gene names for interpretability
        if hasattr(dataset, 'gene_names'):
            data.gene_names = dataset.gene_names
            
        # Add cell type names for visualization
        if hasattr(dataset, 'cell_type_names'):
            data.cell_type_names = dataset.cell_type_names

        # Compute soft neighbourhood cell-type composition as fate target
        if data.y is not None:
            data.neighborhood_composition = self._compute_neighborhood_composition(
                edge_index=data.edge_index,
                cell_types=data.y,
                n_nodes=data.num_nodes,
            )
            
        return data

    # ------------------------------------------------------------------
    # Neighbourhood composition (soft fate labels)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_neighborhood_composition(
        edge_index: torch.Tensor,
        cell_types: torch.Tensor,
        n_nodes: int,
        include_self: bool = True,
    ) -> torch.Tensor:
        """
        Compute cell-type distribution in the 1-hop neighbourhood of every cell.

        For each cell *i* the composition vector is a normalised histogram over
        cell types among its immediate graph neighbours (and optionally itself).
        This serves as a **soft** fate label that decouples fate prediction from
        hard cell-type classification.

        Args:
            edge_index: ``[2, E]`` directed edges (assumed symmetric).
            cell_types: ``[N]`` integer cell-type labels.
            n_nodes:    total number of nodes.
            include_self: whether to include the cell's own type.

        Returns:
            ``[N, C]`` float tensor where *C* = number of unique cell types.
        """
        n_types = int(cell_types.max().item()) + 1
        comp = torch.zeros(n_nodes, n_types, dtype=torch.float32)

        dst = edge_index[1]  # destination nodes
        src_types = cell_types[edge_index[0]]  # type of each source

        # Build one-hot vectors for source cell types and scatter to destinations
        one_hot = torch.zeros(edge_index.size(1), n_types, dtype=torch.float32)
        one_hot.scatter_(1, src_types.unsqueeze(1), 1.0)
        comp.index_add_(0, dst, one_hot)

        if include_self:
            self_one_hot = torch.zeros(n_nodes, n_types, dtype=torch.float32)
            self_one_hot.scatter_(1, cell_types.unsqueeze(1), 1.0)
            comp = comp + self_one_hot

        # Normalise to probability distribution per cell
        row_sum = comp.sum(dim=1, keepdim=True).clamp(min=1.0)
        comp = comp / row_sum
        return comp
    
    def _build_knn_graph(self, coords: torch.Tensor) -> torch.Tensor:
        """Build k-nearest neighbors graph using scipy KDTree."""
        coords_np = coords.numpy()
        
        # Build KDTree
        tree = KDTree(coords_np)
        
        # Query k+1 neighbors (includes self)
        k_query = self.k + 1 if not self.self_loops else self.k
        distances, indices = tree.query(coords_np, k=k_query)
        
        # Build edge list
        n_nodes = coords_np.shape[0]
        src = []
        dst = []
        
        for i in range(n_nodes):
            for j in range(k_query):
                neighbor = indices[i, j]
                if neighbor != i or self.self_loops:  # Skip self-loops unless requested
                    if neighbor != i:  # Don't add self-loop twice
                        src.append(i)
                        dst.append(neighbor)
                    elif self.self_loops:
                        src.append(i)
                        dst.append(neighbor)
                        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        if self.symmetric:
            edge_index = self._make_symmetric(edge_index)
            
        return edge_index
    
    def _build_radius_graph(self, coords: torch.Tensor) -> torch.Tensor:
        """Build radius-based graph using scipy KDTree."""
        coords_np = coords.numpy()
        
        # Build KDTree
        tree = KDTree(coords_np)
        
        # Query all pairs within radius
        pairs = tree.query_pairs(r=self.radius, output_type='ndarray')
        
        if len(pairs) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # Convert to edge_index (bidirectional)
        edge_index = torch.tensor(pairs.T, dtype=torch.long)
        
        if self.symmetric:
            edge_index = self._make_symmetric(edge_index)
        
        return edge_index
    
    def _build_delaunay_graph(self, coords: torch.Tensor) -> torch.Tensor:
        """Build Delaunay triangulation graph."""
        coords_np = coords.numpy()
        
        # Compute Delaunay triangulation
        tri = Delaunay(coords_np)
        
        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(edge)
        
        # Convert to edge_index format
        edge_list = list(edges)
        if not edge_list:
            return torch.zeros((2, 0), dtype=torch.long)
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        
        if self.symmetric:
            edge_index = self._make_symmetric(edge_index)
            
        return edge_index
    
    def _make_symmetric(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Make edge_index symmetric (bidirectional)."""
        reverse = edge_index.flip(0)
        edge_index = torch.cat([edge_index, reverse], dim=1)
        
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index
    
    def _compute_edge_weights(
        self, 
        coords: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge weights based on spatial distance."""
        src, dst = edge_index
        
        # Compute Euclidean distances
        diff = coords[src] - coords[dst]
        dist = torch.norm(diff, dim=1)
        
        # Convert distances to weights (inverse distance with softmax-like normalization)
        # Using Gaussian kernel: w = exp(-d^2 / (2 * sigma^2))
        sigma = dist.std() + 1e-8
        weights = torch.exp(-dist.pow(2) / (2 * sigma.pow(2)))
        
        return weights


class MultiResolutionGraphBuilder:
    """
    Builds multi-resolution graphs for hierarchical learning.
    
    Creates graphs at multiple spatial scales to capture both
    local and global cellular interactions.
    """
    
    def __init__(
        self,
        scales: List[int] = [4, 8, 16],
        base_method: str = 'knn'
    ):
        self.scales = scales
        self.base_method = base_method
        self.builders = [
            SpatialGraphBuilder(method=base_method, k=k)
            for k in scales
        ]
        
    def build_graphs(
        self,
        expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        cell_types: Optional[torch.Tensor] = None,
    ) -> List[Data]:
        """Build graphs at multiple resolutions."""
        graphs = []
        
        for builder in self.builders:
            graph = builder.build_graph(expression, spatial_coords, cell_types)
            graphs.append(graph)
            
        return graphs


class LRGraphBuilder(SpatialGraphBuilder):
    """
    Graph builder that incorporates ligand-receptor interactions.
    
    Extends SpatialGraphBuilder to add edges based on ligand-receptor
    co-expression between neighboring cells.
    """
    
    def __init__(
        self,
        lr_database: 'LigandReceptorDatabase',
        lr_threshold: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lr_database = lr_database
        self.lr_threshold = lr_threshold
        
    def build_graph_with_lr(
        self,
        expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        gene_names: List[str],
        cell_types: Optional[torch.Tensor] = None,
    ) -> Data:
        """
        Build graph with both spatial and L-R edges.
        
        Args:
            expression: Gene expression matrix [n_cells, n_genes]
            spatial_coords: Spatial coordinates [n_cells, 2]
            gene_names: List of gene names corresponding to expression columns
            cell_types: Optional cell type labels
            
        Returns:
            PyG Data object with spatial and L-R edges
        """
        # Build base spatial graph
        base_graph = self.build_graph(expression, spatial_coords, cell_types)
        
        # Identify L-R edges
        lr_edges = self._identify_lr_edges(
            expression, 
            gene_names,
            base_graph.edge_index
        )
        
        if lr_edges is not None and lr_edges.shape[1] > 0:
            # Add L-R edge attributes
            base_graph.lr_edge_index = lr_edges
            
            # Create combined edge index with edge type labels
            n_spatial = base_graph.edge_index.shape[1]
            combined_edges = torch.cat([base_graph.edge_index, lr_edges], dim=1)
            edge_types = torch.cat([
                torch.zeros(n_spatial, dtype=torch.long),
                torch.ones(lr_edges.shape[1], dtype=torch.long)
            ])
            
            base_graph.edge_index = combined_edges
            base_graph.edge_type = edge_types
            
        return base_graph
    
    def _identify_lr_edges(
        self,
        expression: torch.Tensor,
        gene_names: List[str],
        spatial_edges: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Identify ligand-receptor edges based on co-expression."""
        
        # Get L-R pairs from database
        lr_pairs = self.lr_database.get_pairs()
        
        if lr_pairs.empty:
            return None
            
        # Create gene name to index mapping
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        lr_edge_list = []
        
        for _, pair in lr_pairs.iterrows():
            ligand = pair['ligand']
            receptor = pair['receptor']
            
            # Skip if genes not in our dataset
            if ligand not in gene_to_idx or receptor not in gene_to_idx:
                continue
                
            ligand_idx = gene_to_idx[ligand]
            receptor_idx = gene_to_idx[receptor]
            
            # Get expression levels
            ligand_expr = expression[:, ligand_idx]
            receptor_expr = expression[:, receptor_idx]
            
            # For each spatial edge, check L-R co-expression
            src, dst = spatial_edges
            
            # L-R edge: source expresses ligand, target expresses receptor
            lr_mask = (ligand_expr[src] > self.lr_threshold) & \
                      (receptor_expr[dst] > self.lr_threshold)
            
            if lr_mask.any():
                lr_edge_list.append(spatial_edges[:, lr_mask])
                
        if not lr_edge_list:
            return None
            
        return torch.cat(lr_edge_list, dim=1).unique(dim=1)


def build_graph_batch(
    datasets: Dict[str, 'SpatialTranscriptomicsDataset'],
    builder: SpatialGraphBuilder,
) -> Batch:
    """
    Build a batch of graphs from multiple datasets.
    
    Args:
        datasets: Dictionary of datasets
        builder: Graph builder instance
        
    Returns:
        Batched PyG Data object
    """
    graphs = []
    
    for name, ds in datasets.items():
        if ds.has_spatial:
            graph = builder.build_graph(
                ds.expression,
                ds.spatial_coords,
                ds.cell_types
            )
            graph.dataset_name = name
            graphs.append(graph)
        else:
            print(f"Warning: {name} has no spatial coordinates, skipping")
            
    return Batch.from_data_list(graphs)
