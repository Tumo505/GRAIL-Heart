"""Evaluate the trained GRAIL-Heart model on the test set."""

import torch
import yaml
from pathlib import Path
from src.grail_heart.training.trainer import GRAILHeartTrainer
from src.grail_heart.models.grail_heart import GRAILHeart
from src.grail_heart.data.datasets import SpatialTranscriptomicsDataset
from src.grail_heart.data.graph_builder import SpatialGraphBuilder
from src.grail_heart.data.lr_database import LigandReceptorDatabase
from torch_geometric.loader import DataLoader


def main():
    # Load config
    with open('outputs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    
    # Data directory
    data_dir = Path('data/HeartCellAtlasv2/visium-OCT_LV_raw.h5ad').parent
    h5ad_files = sorted(data_dir.glob('*.h5ad'))
    
    # Load L-R database
    lr_db = LigandReceptorDatabase()
    print(f"Loaded L-R database with {len(lr_db.lr_pairs)} pairs")
    
    # Load datasets
    datasets = []
    for f in h5ad_files[:6]:
        try:
            print(f"Loading {f.name}...")
            ds = SpatialTranscriptomicsDataset(
                data_path=f,
                n_top_genes=data_config['n_top_genes'],
                normalize=data_config['normalize'],
                log_transform=data_config['log_transform'],
                min_cells=data_config['min_cells'],
                min_genes=data_config['min_genes'],
            )
            if ds.has_spatial:
                datasets.append(ds)
                print(f"  Loaded: {ds.n_cells} cells, {ds.n_genes} genes")
        except Exception as e:
            print(f"  Failed to load: {e}")
    
    # Build graphs
    print("\nBuilding graphs...")
    graph_builder = SpatialGraphBuilder(
        method=data_config['graph_method'],
        k=data_config['k_neighbors'],
    )
    
    graphs = []
    for i, ds in enumerate(datasets):
        print(f"Building graph {i+1}/{len(datasets)}...")
        lr_edge_dict = lr_db.to_edge_dict({g: i for i, g in enumerate(ds.gene_names)})
        
        # Build graph
        graph = graph_builder.build_graph(
            expression=ds.expression,
            spatial_coords=ds.spatial_coords,
            cell_types=ds.cell_types,
        )
        
        # Add L-R edge information
        if lr_edge_dict['n_edges'] > 0:
            edge_type = torch.zeros(graph.edge_index.shape[1], dtype=torch.long)
            graph.edge_type = edge_type
            
        graphs.append(graph)
        print(f"  Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    
    # Get dimensions from first graph
    sample_graph = graphs[0]
    n_genes = sample_graph.x.shape[1]
    # The training was done with n_cell_types=None, so we use None here too
    n_cell_types = None

    # Create model
    model_config = config['model']
    model = GRAILHeart(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        hidden_dim=model_config['hidden_dim'],
        n_gat_layers=model_config['n_gat_layers'],
        n_heads=model_config['n_heads'],
        n_edge_types=model_config.get('n_edge_types', 2),
        encoder_dims=model_config.get('encoder_dims', [512, 256]),
        dropout=model_config['dropout'],
        use_spatial=model_config.get('use_spatial', True),
        use_variational=model_config.get('use_variational', False),
        tasks=model_config.get('tasks', ['lr', 'reconstruction']),
        n_lr_pairs=2,  # Match checkpoint
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load best checkpoint
    checkpoint = torch.load('outputs/checkpoints/best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}" if checkpoint['best_val_loss'] != float('inf') else "Best validation loss: (not tracked)")

    # Prepare test loader (last graph)
    test_graphs = [graphs[-1]]  # Use last graph as test
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # Evaluate on test set
    print('\n=== Test Set Evaluation ===')
    model.eval()
    
    from src.grail_heart.training.metrics import compute_reconstruction_metrics
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Reconstruction predictions
            x_recon = outputs['reconstruction']
            x_true = batch.x
            
            all_preds.append(x_recon.cpu())
            all_targets.append(x_true.cpu())
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute reconstruction metrics (expects tensors)
    test_metrics = compute_reconstruction_metrics(all_preds, all_targets)
    
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")

    # Save test metrics
    with open('outputs/test_metrics.yaml', 'w') as f:
        yaml.dump({f'test_{k}': float(v) for k, v in test_metrics.items()}, f, default_flow_style=False)
    print('\nTest metrics saved to outputs/test_metrics.yaml')


if __name__ == '__main__':
    main()
