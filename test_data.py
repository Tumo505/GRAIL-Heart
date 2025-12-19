"""
Test GRAIL-Heart with actual cardiac data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from torch_geometric.loader import DataLoader

from grail_heart.data import (
    SpatialTranscriptomicsDataset, 
    SpatialGraphBuilder, 
    LigandReceptorDatabase
)
from grail_heart.models import create_grail_heart
from grail_heart.training import GRAILHeartLoss, create_optimizer

print("=== GRAIL-Heart Data Test ===\n")

# Check for available data
data_dir = Path("data")
print(f"Looking for data in: {data_dir.absolute()}")

# Find h5ad files
h5ad_files = list(data_dir.rglob("*.h5ad"))
print(f"Found {len(h5ad_files)} h5ad files")

if h5ad_files:
    # List first few files
    for f in h5ad_files[:5]:
        print(f"  - {f.relative_to(data_dir)}")
    if len(h5ad_files) > 5:
        print(f"  ... and {len(h5ad_files) - 5} more")
        
    # Try loading the first Visium file
    visium_files = [f for f in h5ad_files if 'visium' in f.name.lower()]
    
    if visium_files:
        print(f"\n=== Loading {visium_files[0].name} ===")
        
        try:
            ds = SpatialTranscriptomicsDataset(
                data_path=visium_files[0],
                n_top_genes=1000,  # Smaller for quick test
                normalize=True,
                log_transform=True,
            )
            
            print(f"Cells: {ds.n_cells}")
            print(f"Genes: {ds.n_genes}")
            print(f"Has spatial: {ds.has_spatial}")
            print(f"Cell types: {ds.n_cell_types}")
            
            # Build graph
            print("\n=== Building Graph ===")
            builder = SpatialGraphBuilder(method='knn', k=6)
            
            if ds.has_spatial:
                graph = builder.build_graph(
                    expression=ds.expression,
                    spatial_coords=ds.spatial_coords,
                    cell_types=ds.cell_types,
                )
                print(f"Nodes: {graph.num_nodes}")
                print(f"Edges: {graph.num_edges}")
                print(f"Edge weight shape: {graph.edge_weight.shape}")
                
                # Add edge type
                graph.edge_type = torch.zeros(graph.num_edges, dtype=torch.long)
                
                # Create model
                print("\n=== Creating Model ===")
                model = create_grail_heart(
                    n_genes=ds.n_genes,
                    config={
                        'hidden_dim': 128,
                        'n_gat_layers': 2,
                        'n_heads': 4,
                        'n_cell_types': ds.n_cell_types,
                    }
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Device: {device}")
                
                model = model.to(device)
                graph = graph.to(device)
                
                # Forward pass
                print("\n=== Forward Pass ===")
                with torch.no_grad():
                    outputs = model(graph)
                    
                for key, val in outputs.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: {val.shape}")
                    elif isinstance(val, dict):
                        print(f"  {key}: dict with keys {list(val.keys())}")
                        
                # Test training step
                print("\n=== Training Step Test ===")
                model.train()
                optimizer = create_optimizer(model, lr=1e-4)
                loss_fn = GRAILHeartLoss(n_cell_types=ds.n_cell_types)
                
                optimizer.zero_grad()
                outputs = model(graph)
                
                targets = {
                    'expression': graph.x,
                    'lr_targets': torch.zeros(graph.num_edges, device=device),
                }
                if graph.y is not None:
                    targets['cell_types'] = graph.y
                    
                loss, loss_dict = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                print(f"Loss: {loss.item():.4f}")
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.item():.4f}")
                        
                print("\n=== SUCCESS! Model can train on real data ===")
                
            else:
                print("No spatial coordinates found in dataset")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No Visium files found")
else:
    print("No h5ad files found. Creating synthetic test...")
    
    # Test with synthetic data
    print("\n=== Synthetic Data Test ===")
    
    n_cells, n_genes = 500, 1000
    x = torch.randn(n_cells, n_genes)
    pos = torch.rand(n_cells, 2)
    
    builder = SpatialGraphBuilder(method='knn', k=6)
    from torch_geometric.data import Data
    
    edge_index = builder._build_knn_graph(pos)
    edge_weight = builder._compute_edge_weights(pos, edge_index)
    
    graph = Data(
        x=x, 
        edge_index=edge_index, 
        edge_weight=edge_weight,
        pos=pos,
        edge_type=torch.zeros(edge_index.shape[1], dtype=torch.long)
    )
    
    model = create_grail_heart(n_genes=n_genes, config={'hidden_dim': 64})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    graph = graph.to(device)
    
    with torch.no_grad():
        outputs = model(graph)
        
    print(f"Forward pass successful!")
    print(f"Output keys: {list(outputs.keys())}")
