"""Evaluate the trained GRAIL-Heart model on the test set."""

import torch
import yaml
from pathlib import Path
from src.grail_heart.training.trainer import GRAILHeartTrainer
from src.grail_heart.models.grail_heart import GRAILHeart
from src.grail_heart.data.datasets import SpatialTranscriptomicsDataset
from src.grail_heart.data.graph_builder import SpatialGraphBuilder
from src.grail_heart.data.cellchat_database import get_omnipath_lr_database
from torch_geometric.loader import DataLoader


def main():
    # Load config
    with open('outputs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    
    # Data directory
    data_dir = Path('data/HeartCellAtlasv2/visium-OCT_LV_raw.h5ad').parent
    h5ad_files = sorted(data_dir.glob('*.h5ad'))
    
    # Load L-R database from OmniPath (CellPhoneDB + CellChat + more)
    cache_path = Path('data/lr_database_cache.csv')
    lr_pairs = get_omnipath_lr_database(cache_path=cache_path)
    print(f"Loaded L-R database with {len(lr_pairs)} pairs from OmniPath")
    
    # Load datasets (no limit - use all files)
    datasets = []
    for f in h5ad_files:
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
    
    # Build graphs with proper L-R edge labeling (must match training)
    print("\nBuilding graphs...")
    graph_builder = SpatialGraphBuilder(
        method=data_config['graph_method'],
        k=data_config['k_neighbors'],
    )
    
    graphs = []
    for i, ds in enumerate(datasets):
        print(f"Building graph {i+1}/{len(datasets)}...")
        
        # Build gene name to index mapping
        gene_to_idx = {g: idx for idx, g in enumerate(ds.gene_names)}
        
        # Build graph
        graph = graph_builder.build_graph(
            expression=ds.expression,
            spatial_coords=ds.spatial_coords,
            cell_types=ds.cell_types,
        )
        
        # Label edges as L-R based on real database (must match training)
        edge_type = torch.zeros(graph.edge_index.shape[1], dtype=torch.long)
        expression_threshold = 0.0
        
        for _, row in lr_pairs.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            if ligand in gene_to_idx and receptor in gene_to_idx:
                lig_idx = gene_to_idx[ligand]
                rec_idx = gene_to_idx[receptor]
                
                lig_expr = ds.expression[:, lig_idx]
                rec_expr = ds.expression[:, rec_idx]
                
                src_nodes = graph.edge_index[0]
                dst_nodes = graph.edge_index[1]
                
                src_has_ligand = (lig_expr[src_nodes] > expression_threshold)
                dst_has_receptor = (rec_expr[dst_nodes] > expression_threshold)
                
                lr_mask = src_has_ligand & dst_has_receptor
                edge_type[lr_mask] = 1
        
        graph.edge_type = edge_type
        n_lr_edges = (edge_type == 1).sum().item()
            
        graphs.append(graph)
        print(f"  Nodes: {graph.num_nodes}, Edges: {graph.num_edges}, L-R edges: {n_lr_edges}")
    
    # Get dimensions from first graph
    sample_graph = graphs[0]
    n_genes = sample_graph.x.shape[1]
    
    # Compute max cell types across all datasets (must match training)
    max_cell_types = max(ds.n_cell_types for ds in datasets if ds.n_cell_types is not None)
    n_cell_types = max_cell_types

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint first to get architecture info
    checkpoint = torch.load('outputs/checkpoints/best.pt', map_location=device, weights_only=False)
    
    # Create model with matching architecture
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
        n_lr_pairs=len(lr_pairs),  # Must match training
    )

    model = model.to(device)

    # Load checkpoint weights
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
