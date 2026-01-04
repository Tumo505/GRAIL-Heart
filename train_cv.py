"""
GRAIL-Heart Cross-Validation Training Script

Implements Leave-One-Region-Out (LORO) cross-validation across 6 cardiac regions:
- AX (Apex)
- LA (Left Atrium)
- LV (Left Ventricle)
- RA (Right Atrium)
- RV (Right Ventricle)
- SP (Septum)

For each fold, train on 5 regions and validate on the held-out region.
Reports mean ± std metrics across all folds.

Usage:
    python train_cv.py --config configs/default.yaml
    python train_cv.py --config configs/default.yaml --n_epochs 50
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from grail_heart.data import (
    SpatialTranscriptomicsDataset,
    SpatialGraphBuilder,
)
from grail_heart.data.cellchat_database import get_omnipath_lr_database
from grail_heart.models import create_grail_heart
from grail_heart.training import (
    GRAILHeartLoss,
    GRAILHeartTrainer,
    create_optimizer,
    create_scheduler,
)

from torch_geometric.loader import DataLoader


# Cardiac region identifiers
CARDIAC_REGIONS = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']


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


def get_region_from_filename(filename: str) -> Optional[str]:
    """Extract cardiac region from filename."""
    filename_upper = filename.upper()
    for region in CARDIAC_REGIONS:
        if f'_{region}_' in filename_upper or f'-{region}_' in filename_upper or filename_upper.startswith(f'VISIUM-OCT_{region}'):
            return region
    return None


def load_region_graphs(
    config: dict, 
    data_dir: Path,
    lr_pairs_df,
) -> Dict[str, 'Data']:
    """
    Load all cardiac region datasets and build graphs.
    
    Returns:
        Dictionary mapping region name to graph Data object
    """
    print("\n=== Loading Region Data ===")
    
    data_config = config['data']
    
    # Find all h5ad files
    h5ad_files = list(data_dir.rglob('*.h5ad'))
    print(f"Found {len(h5ad_files)} h5ad files")
    
    if not h5ad_files:
        atlas_dir = data_dir / 'HeartCellAtlasv2'
        if atlas_dir.exists():
            h5ad_files = list(atlas_dir.glob('visium*.h5ad'))
    
    # Build graph builder
    graph_builder = SpatialGraphBuilder(
        method=data_config['graph_method'],
        k=data_config['k_neighbors'],
    )
    
    region_graphs = {}
    region_datasets = {}
    
    for f in h5ad_files:
        region = get_region_from_filename(f.name)
        if region is None:
            print(f"  Skipping {f.name} (not a cardiac region file)")
            continue
            
        try:
            print(f"Loading {f.name} (region: {region})...")
            ds = SpatialTranscriptomicsDataset(
                data_path=f,
                n_top_genes=data_config['n_top_genes'],
                normalize=data_config['normalize'],
                log_transform=data_config['log_transform'],
                min_cells=data_config['min_cells'],
                min_genes=data_config['min_genes'],
            )
            
            if not ds.has_spatial:
                print(f"  Skipped (no spatial data)")
                continue
                
            # Build graph with differentiation stage
            graph = graph_builder.build_from_dataset(ds)
            
            # Label L-R edges
            gene_to_idx = {g: idx for idx, g in enumerate(ds.gene_names)}
            edge_type = torch.zeros(graph.edge_index.shape[1], dtype=torch.long)
            expression_threshold = 0.0
            
            for _, row in lr_pairs_df.iterrows():
                ligand = row['ligand']
                receptor = row['receptor']
                
                if ligand in gene_to_idx and receptor in gene_to_idx:
                    lig_idx = gene_to_idx[ligand]
                    rec_idx = gene_to_idx[receptor]
                    
                    lig_expr = ds.expression[:, lig_idx]
                    rec_expr = ds.expression[:, rec_idx]
                    
                    lig_cells = (lig_expr > expression_threshold).nonzero(as_tuple=True)[0]
                    rec_cells = (rec_expr > expression_threshold).nonzero(as_tuple=True)[0]
                    
                    if len(lig_cells) > 0 and len(rec_cells) > 0:
                        src_nodes = graph.edge_index[0]
                        dst_nodes = graph.edge_index[1]
                        
                        src_has_ligand = (lig_expr[src_nodes] > expression_threshold)
                        dst_has_receptor = (rec_expr[dst_nodes] > expression_threshold)
                        
                        lr_mask = src_has_ligand & dst_has_receptor
                        edge_type[lr_mask] = 1
            
            graph.edge_type = edge_type
            n_lr_edges = (edge_type == 1).sum().item()
            
            region_graphs[region] = graph
            region_datasets[region] = ds
            
            print(f"  Loaded: {ds.n_cells} cells, {ds.n_genes} genes, {n_lr_edges} L-R edges")
            
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}")
    
    return region_graphs, region_datasets


def create_cv_splits(
    region_graphs: Dict[str, 'Data'],
) -> List[Dict[str, any]]:
    """
    Create leave-one-region-out cross-validation splits.
    
    Returns:
        List of fold dictionaries with train_regions, val_region, train_graphs, val_graph
    """
    folds = []
    regions = list(region_graphs.keys())
    
    for val_region in regions:
        train_regions = [r for r in regions if r != val_region]
        
        fold = {
            'val_region': val_region,
            'train_regions': train_regions,
            'val_graph': region_graphs[val_region],
            'train_graphs': [region_graphs[r] for r in train_regions],
        }
        folds.append(fold)
    
    return folds


def train_fold(
    fold_idx: int,
    fold: Dict,
    config: dict,
    metadata: dict,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train a single cross-validation fold.
    
    Returns:
        Dictionary of validation metrics for this fold
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}: Validating on {fold['val_region']}")
    print(f"Training on: {', '.join(fold['train_regions'])}")
    print(f"{'='*60}")
    
    fold_dir = output_dir / f"fold_{fold_idx}_{fold['val_region']}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader = DataLoader(
        fold['train_graphs'], 
        batch_size=config['data']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        [fold['val_graph']], 
        batch_size=1, 
        shuffle=False
    )
    
    # Create fresh model for this fold
    model_config = config['model'].copy()
    model_config['n_cell_types'] = metadata['n_cell_types']
    model_config['n_lr_pairs'] = metadata['n_lr_pairs']
    n_genes = model_config.pop('n_genes', metadata['n_genes'])
    
    # Handle inverse modelling configuration
    if model_config.get('use_inverse_modelling', False):
        # Use n_cell_types as n_fates if not specified
        if model_config.get('n_fates') is None:
            model_config['n_fates'] = metadata['n_cell_types']
    
    model = create_grail_heart(n_genes=n_genes, config=model_config)
    
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
        differentiation_weight=loss_config.get('differentiation_weight', 0.2),
        gene_target_weight=loss_config.get('gene_target_weight', 0.3),
    )
    
    # Create trainer
    trainer = GRAILHeartTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=fold_dir,
        scheduler=scheduler,
        grad_clip=train_config['grad_clip'],
        grad_accum_steps=train_config['grad_accum_steps'],
        mixed_precision=train_config['mixed_precision'],
        log_interval=config['logging']['log_interval'],
        save_interval=config['checkpoint']['save_interval'],
    )
    
    # Train
    early_config = config['early_stopping']
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=train_config['n_epochs'],
        patience=early_config['patience'],
        monitor=early_config['monitor'],
        mode=early_config['mode'],
    )
    
    # Clear GPU memory before loading checkpoint
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load best model and get final validation metrics
    best_checkpoint = fold_dir / 'checkpoints' / 'best.pt'
    if best_checkpoint.exists():
        # Load to CPU first to avoid OOM, then move to GPU
        trainer.load_checkpoint(best_checkpoint, map_location='cpu')
        trainer.model.to(device)
    
    val_metrics = trainer.validate(val_loader)
    
    # Add region info
    val_metrics['region'] = fold['val_region']
    
    # Save fold metrics
    with open(fold_dir / 'val_metrics.yaml', 'w') as f:
        yaml.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in val_metrics.items()}, f)
    
    print(f"\nFold {fold_idx + 1} ({fold['val_region']}) Validation Metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k}: {v:.4f}")
    
    # Clean up to free memory - more aggressive cleanup
    del model, trainer, optimizer, scheduler, loss_fn
    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return val_metrics


def aggregate_cv_results(
    fold_metrics: List[Dict[str, float]],
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate cross-validation results across all folds.
    
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    # Collect all numeric metrics
    all_metrics = defaultdict(list)
    for fold in fold_metrics:
        for k, v in fold.items():
            if isinstance(v, (int, float, np.floating)) and k != 'region':
                all_metrics[k].append(float(v))
    
    # Compute statistics
    summary = {}
    for metric, values in all_metrics.items():
        values = np.array(values)
        summary[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': [float(v) for v in values],
        }
    
    # Print summary
    print(f"\n{'Metric':<25} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    
    key_metrics = ['val_loss', 'lr_auroc', 'lr_auprc', 'recon_r2', 'cell_type_accuracy']
    for metric in key_metrics:
        if metric in summary:
            s = summary[metric]
            print(f"{metric:<25} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
    
    print("\nAll metrics:")
    for metric, s in sorted(summary.items()):
        if metric not in key_metrics:
            print(f"  {metric}: {s['mean']:.4f} ± {s['std']:.4f}")
    
    # Per-region breakdown
    print("\nPer-Region Results:")
    print(f"{'Region':<10} {'Loss':>10} {'LR-AUROC':>10} {'Recon-R²':>10}")
    print("-" * 40)
    for fold in fold_metrics:
        region = fold.get('region', 'Unknown')
        loss = fold.get('val_total_loss', fold.get('val_loss', 0))
        auroc = fold.get('val_auroc', fold.get('lr_auroc', 0))
        r2 = fold.get('val_r2', fold.get('recon_r2', 0))
        print(f"{region:<10} {loss:>10.4f} {auroc:>10.4f} {r2:>10.4f}")
    
    # Convert all values to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save comprehensive results
    results = {
        'summary': summary,
        'per_fold': convert_to_native(fold_metrics),
        'n_folds': len(fold_metrics),
    }
    
    with open(output_dir / 'cv_results.yaml', 'w') as f:
        yaml.dump(convert_to_native(results), f, default_flow_style=False)
    
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'cv_results.yaml'}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Train GRAIL-Heart with Cross-Validation')
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
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Override number of epochs per fold')
    parser.add_argument('--folds', type=str, default=None,
                        help='Specific folds to run (e.g., "0,1,2" or "LA,LV")')
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
    if args.n_epochs:
        config['training']['n_epochs'] = args.n_epochs
    
    # Set up paths
    data_dir = Path(config['paths']['data_dir'])
    base_output_dir = Path(config['paths']['output_dir'])
    
    # Create CV-specific output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_output_dir / f'cv_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
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
    
    # Load L-R database
    print("\n=== Loading L-R Database ===")
    cache_path = data_dir / 'lr_database_cache.csv'
    lr_pairs_df = get_omnipath_lr_database(cache_path=cache_path)
    print(f"Loaded {len(lr_pairs_df)} L-R pairs from OmniPath")
    
    # Optionally limit L-R pairs for memory efficiency
    max_lr_pairs = config['data'].get('max_lr_pairs', None)
    if max_lr_pairs and len(lr_pairs_df) > max_lr_pairs:
        # Prioritize pairs with more sources (higher confidence)
        if 'n_sources' in lr_pairs_df.columns:
            lr_pairs_df = lr_pairs_df.sort_values('n_sources', ascending=False)
        lr_pairs_df = lr_pairs_df.head(max_lr_pairs).reset_index(drop=True)
        print(f"Limited to top {max_lr_pairs} L-R pairs (by curation confidence)")
    
    # Load all region data
    region_graphs, region_datasets = load_region_graphs(config, data_dir, lr_pairs_df)
    
    if len(region_graphs) < 2:
        raise ValueError(f"Need at least 2 regions for CV, found {len(region_graphs)}")
    
    print(f"\nLoaded {len(region_graphs)} cardiac regions: {list(region_graphs.keys())}")
    
    # Compute metadata
    max_cell_types = max(ds.n_cell_types for ds in region_datasets.values() if ds.n_cell_types is not None)
    metadata = {
        'n_genes': list(region_datasets.values())[0].n_genes,
        'n_cell_types': max_cell_types,
        'gene_names': list(region_datasets.values())[0].gene_names,
        'n_lr_pairs': len(lr_pairs_df),
    }
    
    # Create CV splits
    folds = create_cv_splits(region_graphs)
    print(f"\nCreated {len(folds)} cross-validation folds (leave-one-region-out)")
    
    # Parse which folds to run
    folds_to_run = list(range(len(folds)))
    if args.folds:
        if ',' in args.folds:
            fold_specs = args.folds.split(',')
            folds_to_run = []
            for spec in fold_specs:
                spec = spec.strip()
                if spec.isdigit():
                    folds_to_run.append(int(spec))
                else:
                    # Match by region name
                    for i, fold in enumerate(folds):
                        if fold['val_region'].upper() == spec.upper():
                            folds_to_run.append(i)
    
    print(f"Running folds: {folds_to_run}")
    
    # Train each fold
    fold_metrics = []
    for fold_idx in folds_to_run:
        fold = folds[fold_idx]
        metrics = train_fold(
            fold_idx=fold_idx,
            fold=fold,
            config=config,
            metadata=metadata,
            output_dir=output_dir,
            device=device,
        )
        fold_metrics.append(metrics)
        
        # Save intermediate results
        with open(output_dir / 'fold_metrics_partial.yaml', 'w') as f:
            yaml.dump(fold_metrics, f)
    
    # Aggregate results
    if len(fold_metrics) > 1:
        summary = aggregate_cv_results(fold_metrics, output_dir)
    else:
        print(f"\nOnly {len(fold_metrics)} fold completed. Run more folds for CV summary.")
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
