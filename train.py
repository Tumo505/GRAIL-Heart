"""
GRAIL-Heart Training Script

Main entry point for training the GRAIL-Heart model on cardiac
spatial transcriptomics data.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --data_dir path/to/data
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from grail_heart.data import (
    SpatialTranscriptomicsDataset,
    CardiacDataModule,
    SpatialGraphBuilder,
    LigandReceptorDatabase,
)
from grail_heart.data.cellchat_database import (
    get_omnipath_lr_database,
    filter_for_cardiac,
    get_mechanosensitive_gene_sets,
    build_pathway_gene_mask,
    CARDIAC_PATHWAYS,
)
from grail_heart.models import GRAILHeart, create_grail_heart
from grail_heart.training import (
    GRAILHeartLoss,
    GRAILHeartTrainer,
    create_optimizer,
    create_scheduler,
)

from torch_geometric.loader import DataLoader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict, data_dir: Path):
    """
    Prepare datasets and data loaders.
    
    Args:
        config: Configuration dictionary
        data_dir: Path to data directory
        
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    print("\n=== Preparing Data ===")
    
    data_config = config['data']
    
    # Initialize L-R database from OmniPath (CellPhoneDB + CellChat + more)
    print("Loading L-R database from OmniPath (CellPhoneDB, CellChat, ICELLNET)...")
    cache_path = data_dir / 'lr_database_cache.csv'
    lr_pairs_df = get_omnipath_lr_database(cache_path=cache_path)
    print(f"Loaded {len(lr_pairs_df)} L-R pairs from OmniPath database")
    
    # Create a simple object to hold the pairs for compatibility
    class LRDatabase:
        def __init__(self, pairs_df):
            self.lr_pairs = pairs_df
        def get_pairs(self):
            return self.lr_pairs
    
    lr_db = LRDatabase(lr_pairs_df)
    
    # Find all h5ad files
    h5ad_files = list(data_dir.rglob('*.h5ad'))
    print(f"Found {len(h5ad_files)} h5ad files")
    
    if not h5ad_files:
        # Try loading from HeartCellAtlasv2
        atlas_dir = data_dir / 'HeartCellAtlasv2'
        if atlas_dir.exists():
            h5ad_files = list(atlas_dir.glob('visium*.h5ad'))
            
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {data_dir}")
    
    # Get max files from config (None = no limit, use all files)
    max_files = data_config.get('max_files', None)
    files_to_load = h5ad_files if max_files is None else h5ad_files[:max_files]
    print(f"Loading {len(files_to_load)} of {len(h5ad_files)} available files")
    
    # Load datasets
    datasets = []
    for f in files_to_load:
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
            else:
                print(f"  Skipped (no spatial data)")
        except Exception as e:
            print(f"  Failed to load: {e}")
            
    if not datasets:
        raise ValueError("No valid datasets loaded")
        
    # Build graphs
    print("\nBuilding graphs...")
    graph_builder = SpatialGraphBuilder(
        method=data_config['graph_method'],
        k=data_config['k_neighbors'],
    )
    
    # Get L-R pairs from database for edge labeling
    lr_pairs = lr_db.get_pairs()
    print(f"L-R database has {len(lr_pairs)} pairs")
    
    graphs = []
    for i, ds in enumerate(datasets):
        print(f"Building graph {i+1}/{len(datasets)}...")
        
        # Build gene name to index mapping
        gene_to_idx = {g: idx for idx, g in enumerate(ds.gene_names)}
        
        # Build graph with L-R edges and differentiation stage
        graph = graph_builder.build_from_dataset(ds)
        
        # Label edges as L-R (type=1) based on real L-R database
        # An edge (i,j) is an L-R edge if cell i expresses a ligand and cell j expresses its receptor
        edge_type = torch.zeros(graph.edge_index.shape[1], dtype=torch.long)
        
        # Find expressed L-R pairs in this dataset
        lr_edge_count = 0
        expression_threshold = 0.0  # Minimum expression to consider gene "expressed"
        
        for _, row in lr_pairs.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            # Check if both genes are in the dataset
            if ligand in gene_to_idx and receptor in gene_to_idx:
                lig_idx = gene_to_idx[ligand]
                rec_idx = gene_to_idx[receptor]
                
                # Get expression vectors
                lig_expr = ds.expression[:, lig_idx]
                rec_expr = ds.expression[:, rec_idx]
                
                # Find cells expressing ligand and receptor
                lig_cells = (lig_expr > expression_threshold).nonzero(as_tuple=True)[0]
                rec_cells = (rec_expr > expression_threshold).nonzero(as_tuple=True)[0]
                
                if len(lig_cells) > 0 and len(rec_cells) > 0:
                    # For each edge, check if source expresses ligand and target expresses receptor
                    src_nodes = graph.edge_index[0]
                    dst_nodes = graph.edge_index[1]
                    
                    # Create masks for ligand/receptor expression
                    src_has_ligand = (lig_expr[src_nodes] > expression_threshold)
                    dst_has_receptor = (rec_expr[dst_nodes] > expression_threshold)
                    
                    # Mark edges where source has ligand AND target has receptor
                    lr_mask = src_has_ligand & dst_has_receptor
                    edge_type[lr_mask] = 1
                    lr_edge_count += lr_mask.sum().item()
        
        graph.edge_type = edge_type
        n_lr_edges = (edge_type == 1).sum().item()
        
        graphs.append(graph)
        print(f"  Nodes: {graph.num_nodes}, Edges: {graph.num_edges}, L-R edges: {n_lr_edges}")
        
    # Split into train/val/test
    n_graphs = len(graphs)
    n_train = max(1, int(0.7 * n_graphs))
    n_val = max(1, int(0.15 * n_graphs))
    
    # Shuffle
    indices = torch.randperm(n_graphs).tolist()
    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test_graphs = [graphs[i] for i in indices[n_train + n_val:]]
    
    # Ensure at least one graph in each split
    if not val_graphs:
        val_graphs = train_graphs[:1]
    if not test_graphs:
        test_graphs = val_graphs[:1]
        
    print(f"\nData splits: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=data_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=data_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=data_config['batch_size'], shuffle=False)
    
    # Metadata - compute max n_cell_types across all datasets for consistent model
    max_cell_types = max(ds.n_cell_types for ds in datasets if ds.n_cell_types is not None)
    metadata = {
        'n_genes': datasets[0].n_genes,
        'n_cell_types': max_cell_types,  # Use max across all regions
        'gene_names': datasets[0].gene_names,
        'n_lr_pairs': len(lr_pairs),  # Number of L-R pairs from database
        'data_dir': str(data_dir),     # For resolving msigdb path later
    }
    
    return train_loader, val_loader, test_loader, metadata


def create_model(config: dict, metadata: dict) -> GRAILHeart:
    """Create GRAIL-Heart model from config."""
    
    model_config = config['model'].copy()
    model_config['n_genes'] = metadata['n_genes']
    model_config['n_cell_types'] = metadata['n_cell_types']
    # NOTE: Do NOT pass n_lr_pairs to the model — it creates one Linear(hidden, hidden)
    # per LR pair which is 22K×65K = 1.46B params.  The MLP/bilinear scorer works
    # without per-pair projections.
    model_config['n_lr_pairs'] = None
    
    # Extract n_genes before passing to create_grail_heart
    n_genes = model_config.pop('n_genes')
    
    # Handle inverse modelling configuration
    if model_config.get('use_inverse_modelling', False):
        # Use n_cell_types as n_fates if not specified
        if model_config.get('n_fates') is None:
            model_config['n_fates'] = metadata['n_cell_types']

    # ── Build biologically-grounded pathway masks (WP1) ──────────────
    pathway_gene_mask = None
    mechano_gene_mask = None
    mechano_pathway_names = None

    if model_config.get('use_inverse_modelling', False):
        data_config = config.get('data', {})
        msigdb_path = data_config.get('msigdb_path', None)
        include_hallmark = data_config.get('include_hallmark', True)

        gene_sets = get_mechanosensitive_gene_sets(
            msigdb_path=msigdb_path,
            include_hallmark=include_hallmark,
        )
        gene_names = metadata.get('gene_names', [])
        if gene_sets and gene_names:
            mask_np = build_pathway_gene_mask(gene_sets, gene_names)
            pathway_gene_mask = torch.tensor(mask_np, dtype=torch.float32)
            # Mechanosensitive subset: curated + Hippo + Piezo + Integrin
            mechano_keys = [k for k in sorted(gene_sets.keys())
                           if k in ('YAP_TAZ_Hippo', 'Piezo_Mechano', 'Integrin_FAK')
                           or k in CARDIAC_PATHWAYS]
            mechano_sets = {k: gene_sets[k] for k in mechano_keys if k in gene_sets}
            if mechano_sets:
                mechano_np = build_pathway_gene_mask(mechano_sets, gene_names)
                mechano_gene_mask = torch.tensor(mechano_np, dtype=torch.float32)
                mechano_pathway_names = sorted(mechano_sets.keys())
                model_config['n_mechano_pathways'] = len(mechano_pathway_names)
            model_config['n_pathways'] = mask_np.shape[0]
            print(f"Pathway masks: {mask_np.shape[0]} total pathways, "
                  f"{len(mechano_pathway_names) if mechano_pathway_names else 0} mechano pathways")

    model_config['pathway_gene_mask'] = pathway_gene_mask
    model_config['mechano_gene_mask'] = mechano_gene_mask
    model_config['mechano_pathway_names'] = mechano_pathway_names

    model = create_grail_heart(n_genes=n_genes, config=model_config)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train GRAIL-Heart model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.device:
        config['hardware']['device'] = args.device
    if args.seed:
        config['hardware']['seed'] = args.seed
        
    # Set up paths
    data_dir = Path(config['paths']['data_dir'])
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # Set seed
    set_seed(config['hardware']['seed'])
    
    # Set device
    device = torch.device(config['hardware']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    # Prepare data
    train_loader, val_loader, test_loader, metadata = prepare_data(config, data_dir)
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(config, metadata)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer and scheduler
    train_config = config['training']
    optimizer = create_optimizer(
        model,
        optimizer_name=train_config['optimizer'],
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=train_config['scheduler'],
        n_epochs=train_config['n_epochs'],
        warmup_epochs=train_config['warmup_epochs'],
    )
    
    # Create loss function
    loss_config = config['loss']
    loss_fn = GRAILHeartLoss(
        lr_weight=loss_config['lr_weight'],
        recon_weight=loss_config['recon_weight'],
        cell_type_weight=loss_config['cell_type_weight'],
        signaling_weight=loss_config['signaling_weight'],
        kl_weight=loss_config['kl_weight'],
        contrastive_weight=loss_config.get('contrastive_weight', 0.5),
        n_cell_types=metadata['n_cell_types'],
        use_contrastive=loss_config.get('use_contrastive', True),
        recon_loss_type=loss_config.get('recon_loss_type', 'combined'),
        # Inverse modelling loss parameters
        use_inverse_losses=loss_config.get('use_inverse_losses', True),
        fate_weight=loss_config.get('fate_weight', 0.5),
        causal_weight=loss_config.get('causal_weight', 0.3),
        differentiation_weight=loss_config.get('differentiation_weight', 0.5),
        gene_target_weight=loss_config.get('gene_target_weight', 0.3),
        cycle_weight=loss_config.get('cycle_weight', 0.3),
        pathway_grounding_weight=loss_config.get('pathway_grounding_weight', 0.1),
    )
    
    # Create trainer
    trainer = GRAILHeartTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        scheduler=scheduler,
        grad_clip=train_config['grad_clip'],
        grad_accum_steps=train_config['grad_accum_steps'],
        mixed_precision=train_config['mixed_precision'],
        log_interval=config['logging']['log_interval'],
        save_interval=config['checkpoint']['save_interval'],
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Initialize WandB if enabled
    if config['logging']['use_wandb']:
        trainer.init_wandb(
            project=config['logging']['wandb_project'],
            config=config,
        )
        
    # Train
    print("\n=== Starting Training ===")
    early_config = config['early_stopping']
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=train_config['n_epochs'],
        patience=early_config['patience'],
        monitor=early_config['monitor'],
        mode=early_config['mode'],
    )
    
    # Final evaluation on test set
    print("\n=== Final Evaluation ===")
    trainer.load_checkpoint(output_dir / 'checkpoints' / 'best.pt')
    test_metrics = trainer.validate(test_loader)
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # Save test metrics
    with open(output_dir / 'test_metrics.yaml', 'w') as f:
        yaml.dump({k: float(v) for k, v in test_metrics.items()}, f)
        
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
