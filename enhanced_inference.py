"""
Enhanced Inference Script for GRAIL-Heart with Expanded L-R Database

This script:
1. Loads trained GRAIL-Heart model
2. Runs inference on all cardiac regions
3. Detects L-R interactions using expanded database (500+ pairs)
4. Creates comprehensive spatial visualizations
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from grail_heart.models import GRAILHeart
from grail_heart.data.cellchat_database import (
    get_omnipath_lr_database,
    filter_for_cardiac,
    annotate_cardiac_pathways,
)
from grail_heart.data.expanded_lr_database import (
    filter_to_expressed_genes,
    compute_lr_scores,
    get_lr_genes,
)
from grail_heart.visualization import SpatialVisualizer, PATHWAY_COLORS


class EnhancedInference:
    """
    Enhanced inference pipeline with expanded L-R detection and visualization.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        output_dir: str = 'outputs/enhanced_analysis',
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'networks').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.lr_database = None
        self.visualizer = SpatialVisualizer(
            output_dir=str(self.output_dir / 'figures'),
            fig_format='png',
            dpi=300
        )
        
        # Store results
        self.region_results = {}
        self.all_lr_scores = []
        
    def load_model(self) -> None:
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Get metadata for n_cell_types
        n_cell_types = config.get('n_cell_types', checkpoint.get('n_cell_types', 10))
        n_genes = config.get('n_genes', checkpoint.get('n_genes', 2000))
        
        # Infer n_lr_pairs from checkpoint state dict to ensure match
        state_dict = checkpoint['model_state_dict']
        lr_keys = [k for k in state_dict.keys() if 'lr_projections' in k and k.endswith('.weight')]
        if lr_keys:
            indices = [int(k.split('.')[-2]) for k in lr_keys]
            n_lr_pairs = max(indices) + 1
            print(f"  Detected {n_lr_pairs} L-R pairs from checkpoint")
        else:
            # Fallback to expanded database
            from grail_heart.data.expanded_lr_database import get_expanded_lr_database
            lr_pairs_for_model = get_expanded_lr_database()
            n_lr_pairs = config.get('n_lr_pairs', checkpoint.get('n_lr_pairs', len(lr_pairs_for_model)))
        
        # Initialize model with matching parameters
        tasks = model_config.get('tasks', ['lr', 'reconstruction', 'cell_type'])
        
        self.model = GRAILHeart(
            n_genes=n_genes,
            hidden_dim=model_config.get('hidden_dim', 256),
            n_gat_layers=model_config.get('n_gat_layers', 3),
            n_heads=model_config.get('n_heads', 8),
            n_cell_types=n_cell_types,
            n_edge_types=model_config.get('n_edge_types', 2),
            encoder_dims=model_config.get('encoder_dims', [512, 256]),
            dropout=model_config.get('dropout', 0.1),
            use_spatial=model_config.get('use_spatial', True),
            use_variational=model_config.get('use_variational', False),
            tasks=tasks,
            n_lr_pairs=n_lr_pairs,
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
    
    def load_lr_database(self) -> None:
        """Load L-R database from OmniPath (CellPhoneDB + CellChat + more)."""
        print("Loading L-R database from OmniPath...")
        cache_path = Path(self.data_dir) / 'lr_database_cache.csv'
        self.lr_database = get_omnipath_lr_database(cache_path=cache_path)
        self.lr_database = annotate_cardiac_pathways(self.lr_database)
        print(f"Loaded {len(self.lr_database)} L-R pairs")
        print(f"  Unique ligands: {len(self.lr_database['ligand'].unique())}")
        print(f"  Unique receptors: {len(self.lr_database['receptor'].unique())}")
        print(f"  Pathway categories: {len(self.lr_database['pathway'].unique())}")
    
    def process_region(
        self,
        region: str,
        visualize: bool = True,
    ) -> Dict:
        """
        Process a single cardiac region.
        
        Args:
            region: Region name (AX, LA, LV, RA, RV, SP)
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with region results
        """
        print(f"\n{'='*60}")
        print(f"Processing region: {region}")
        print('='*60)
        
        # Load dataset
        h5ad_path = self.data_dir / 'HeartCellAtlasv2' / f'visium-OCT_{region}_raw.h5ad'
        
        if not h5ad_path.exists():
            print(f"  ⚠ File not found: {h5ad_path}")
            return None
        
        adata = sc.read_h5ad(h5ad_path)
        print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Preprocess
        adata = self._preprocess(adata)
        
        # Get gene names (symbols)
        if 'SYMBOL' in adata.var.columns:
            gene_names = adata.var['SYMBOL'].tolist()
        else:
            gene_names = adata.var_names.tolist()
        
        # Filter L-R database to expressed genes
        expressed_lr = filter_to_expressed_genes(self.lr_database, gene_names)
        print(f"  Matched L-R pairs: {len(expressed_lr)}")
        
        if len(expressed_lr) == 0:
            print("  ⚠ No matching L-R pairs found!")
            # Debug: show which LR genes are in the data
            lr_genes = get_lr_genes()
            found = lr_genes.intersection(set(gene_names))
            print(f"  Found {len(found)} L-R genes in data")
            return None
        
        # Build spatial graph
        coords = self._get_coordinates(adata)
        edge_index = self._build_graph(coords, k=15)
        
        print(f"  Spatial graph: {edge_index.shape[1]} edges")
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = np.array(adata.X)
        
        # Compute L-R scores
        lr_scores = compute_lr_scores(
            expression=expression,
            gene_names=gene_names,
            edge_index=edge_index,
            lr_df=expressed_lr,
            method='product'
        )
        
        print(f"  Detected interactions: {len(lr_scores)}")
        
        if len(lr_scores) > 0:
            print(f"\n  Top 10 L-R interactions:")
            top10 = lr_scores.nlargest(10, 'mean_score')
            for _, row in top10.iterrows():
                print(f"    {row['ligand']} -> {row['receptor']}: "
                      f"score={row['mean_score']:.3f}, "
                      f"pathway={row['pathway']}")
        
        # Compute cell-level communication scores
        cell_comm_scores = self._compute_cell_comm_scores(
            expression, gene_names, edge_index, lr_scores
        )
        
        # Visualize
        if visualize and len(lr_scores) > 0:
            self._visualize_region(
                region=region,
                coords=coords,
                edge_index=edge_index,
                cell_comm_scores=cell_comm_scores,
                lr_scores=lr_scores,
                expression=expression,
                gene_names=gene_names,
            )
        
        # Store results
        result = {
            'region': region,
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'n_edges': edge_index.shape[1],
            'n_lr_pairs_matched': len(expressed_lr),
            'n_interactions_detected': len(lr_scores),
            'lr_scores': lr_scores,
            'coords': coords,
            'edge_index': edge_index,
        }
        
        self.region_results[region] = result
        
        # Save region results
        lr_scores.to_csv(
            self.output_dir / 'tables' / f'{region}_lr_scores.csv',
            index=False
        )
        
        return result
    
    def _preprocess(self, adata: ad.AnnData) -> ad.AnnData:
        """Preprocess AnnData object."""
        adata = adata.copy()
        
        # Basic filtering
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=10)
        
        # Normalize and log
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Select HVGs
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable].copy()
        
        # Scale
        sc.pp.scale(adata, max_value=10)
        
        return adata
    
    def _get_coordinates(self, adata: ad.AnnData) -> np.ndarray:
        """Extract spatial coordinates."""
        if 'spatial' in adata.obsm:
            return np.array(adata.obsm['spatial'])
        elif 'X_spatial' in adata.obsm:
            return np.array(adata.obsm['X_spatial'])
        else:
            # Try to find coordinates in obs
            if 'x' in adata.obs and 'y' in adata.obs:
                return np.column_stack([adata.obs['x'].values, adata.obs['y'].values])
            else:
                raise ValueError("No spatial coordinates found")
    
    def _build_graph(
        self,
        coords: np.ndarray,
        k: int = 15,
    ) -> np.ndarray:
        """Build k-NN spatial graph."""
        from sklearn.neighbors import kneighbors_graph
        
        adj = kneighbors_graph(coords, n_neighbors=k, mode='connectivity')
        adj = adj + adj.T  # Make symmetric
        adj.data = np.ones_like(adj.data)  # Binary
        
        # Convert to edge index
        coo = adj.tocoo()
        edge_index = np.vstack([coo.row, coo.col])
        
        return edge_index
    
    def _compute_cell_comm_scores(
        self,
        expression: np.ndarray,
        gene_names: List[str],
        edge_index: np.ndarray,
        lr_scores: pd.DataFrame,
    ) -> np.ndarray:
        """Compute per-cell communication scores."""
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n_cells = expression.shape[0]
        cell_scores = np.zeros(n_cells)
        
        for _, row in lr_scores.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            if ligand not in gene_to_idx or receptor not in gene_to_idx:
                continue
            
            lig_idx = gene_to_idx[ligand]
            rec_idx = gene_to_idx[receptor]
            
            # For each edge, compute L-R score
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                score = expression[src, lig_idx] * expression[dst, rec_idx]
                cell_scores[dst] += score  # Receiving cell gets the score
        
        # Normalize
        if cell_scores.max() > 0:
            cell_scores = cell_scores / cell_scores.max()
        
        return cell_scores
    
    def _visualize_region(
        self,
        region: str,
        coords: np.ndarray,
        edge_index: np.ndarray,
        cell_comm_scores: np.ndarray,
        lr_scores: pd.DataFrame,
        expression: np.ndarray,
        gene_names: List[str],
    ) -> None:
        """Create visualizations for a region."""
        print(f"  Creating visualizations...")
        
        # 1. Spatial communication network
        self.visualizer.plot_spatial_communication(
            coords=coords,
            edge_index=edge_index,
            cell_colors=cell_comm_scores,
            title=f'{region} - Cell-Cell Communication',
            top_edges=3000,
            save_name=f'{region}_spatial_network'
        )
        plt.close()
        
        # 2. L-R heatmap
        self.visualizer.plot_lr_heatmap(
            lr_scores=lr_scores,
            top_n=25,
            save_name=f'{region}_lr_heatmap'
        )
        plt.close()
        
        # 3. Pathway activity
        self.visualizer.plot_pathway_activity(
            lr_scores=lr_scores,
            save_name=f'{region}_pathway_activity'
        )
        plt.close()
        
        # 4. Specific L-R pairs (top 3)
        if len(lr_scores) >= 3:
            top3 = lr_scores.nlargest(3, 'mean_score')
            for _, row in top3.iterrows():
                try:
                    self.visualizer.plot_specific_lr_spatial(
                        coords=coords,
                        edge_index=edge_index,
                        expression=expression,
                        gene_names=gene_names,
                        ligand=row['ligand'],
                        receptor=row['receptor'],
                        title=f"{region}: {row['ligand']}->{row['receptor']} ({row['pathway']})",
                        save_name=f"{region}_{row['ligand']}_{row['receptor']}"
                    )
                    plt.close()
                except Exception as e:
                    print(f"    Warning: Could not plot {row['ligand']}->{row['receptor']}: {e}")
    
    def run_all_regions(self) -> None:
        """Process all cardiac regions."""
        regions = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']
        
        for region in regions:
            try:
                self.process_region(region, visualize=True)
            except Exception as e:
                print(f"Error processing {region}: {e}")
                import traceback
                traceback.print_exc()
    
    def create_cross_region_comparison(self) -> None:
        """Create comparison across all regions."""
        print("\n" + "="*60)
        print("Creating cross-region comparison")
        print("="*60)
        
        # Collect all L-R scores
        all_pairs = set()
        region_scores = {}
        
        for region, result in self.region_results.items():
            if result is None or len(result.get('lr_scores', [])) == 0:
                continue
            
            lr_df = result['lr_scores']
            for _, row in lr_df.iterrows():
                pair = (row['ligand'], row['receptor'], row['pathway'], row['function'])
                all_pairs.add(pair)
            
            # Create dict for this region
            scores = {}
            for _, row in lr_df.iterrows():
                key = (row['ligand'], row['receptor'])
                scores[key] = row['mean_score']
            region_scores[region] = scores
        
        if not all_pairs:
            print("No L-R pairs to compare!")
            return
        
        # Create comparison DataFrame
        rows = []
        for ligand, receptor, pathway, function in all_pairs:
            row = {
                'ligand': ligand,
                'receptor': receptor,
                'pathway': pathway,
                'function': function,
            }
            for region in self.region_results.keys():
                key = (ligand, receptor)
                row[region] = region_scores.get(region, {}).get(key, 0)
            rows.append(row)
        
        comparison_df = pd.DataFrame(rows)
        
        # Add mean score
        regions = [r for r in self.region_results.keys() if r in comparison_df.columns]
        comparison_df['mean_score'] = comparison_df[regions].mean(axis=1)
        comparison_df = comparison_df.sort_values('mean_score', ascending=False)
        
        # Save
        comparison_df.to_csv(
            self.output_dir / 'tables' / 'cross_region_comparison.csv',
            index=False
        )
        
        print(f"Saved comparison: {len(comparison_df)} L-R pairs across {len(regions)} regions")
        
        # Create visualizations
        if len(regions) > 1:
            # Multi-region heatmap
            self.visualizer.plot_lr_heatmap(
                lr_scores=comparison_df,
                top_n=30,
                save_name='cross_region_lr_heatmap'
            )
            plt.close()
            
            # Region comparison panels
            self.visualizer.plot_region_comparison(
                comparison_df=comparison_df,
                top_n=15,
                save_name='region_comparison_panels'
            )
            plt.close()
        
        # Create summary
        n_cells = {r: res['n_cells'] for r, res in self.region_results.items() if res}
        n_interactions = {r: res['n_interactions_detected'] for r, res in self.region_results.items() if res}
        
        top_pathways = {}
        for region, result in self.region_results.items():
            if result and len(result.get('lr_scores', [])) > 0:
                pathway_scores = result['lr_scores'].groupby('pathway')['total_score'].sum()
                top_pathways[region] = pathway_scores.nlargest(5).index.tolist()
        
        self.visualizer.plot_network_summary(
            n_cells_per_region=n_cells,
            n_interactions_per_region=n_interactions,
            top_pathways_per_region=top_pathways,
            save_name='network_summary_dashboard'
        )
        plt.close()
        
        print("\nTop L-R interactions across all regions:")
        print(comparison_df.head(20)[['ligand', 'receptor', 'pathway', 'mean_score']].to_string())
    
    def generate_report(self) -> str:
        """Generate summary report."""
        report = []
        report.append("="*60)
        report.append("GRAIL-Heart Enhanced Analysis Report")
        report.append("="*60)
        report.append("")
        
        # Overall statistics
        total_cells = sum(r['n_cells'] for r in self.region_results.values() if r)
        total_interactions = sum(r['n_interactions_detected'] for r in self.region_results.values() if r)
        
        report.append(f"Total cells analyzed: {total_cells:,}")
        report.append(f"Total L-R pairs in database: {len(self.lr_database)}")
        report.append(f"Total interactions detected: {total_interactions}")
        report.append("")
        
        # Per-region summary
        report.append("Per-region summary:")
        report.append("-"*40)
        for region, result in self.region_results.items():
            if result:
                report.append(f"  {region}:")
                report.append(f"    Cells: {result['n_cells']:,}")
                report.append(f"    Edges: {result['n_edges']:,}")
                report.append(f"    L-R pairs matched: {result['n_lr_pairs_matched']}")
                report.append(f"    Interactions detected: {result['n_interactions_detected']}")
        
        report.append("")
        report.append("Files saved to:")
        report.append(f"  Tables: {self.output_dir / 'tables'}")
        report.append(f"  Figures: {self.output_dir / 'figures'}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text


def main():
    """Run enhanced inference pipeline."""
    print("="*60)
    print("GRAIL-Heart Enhanced Inference Pipeline")
    print("="*60)
    
    # Find best checkpoint - prefer CV checkpoints with inverse modelling
    import os
    from pathlib import Path
    cv_dirs = sorted([d for d in Path('outputs').iterdir() if d.name.startswith('cv_')])
    if cv_dirs:
        latest_cv = cv_dirs[-1]
        # Use best fold checkpoint (RV typically performs best)
        checkpoint_path = latest_cv / 'fold_4_RV' / 'checkpoints' / 'best.pt'
        if not checkpoint_path.exists():
            # Fallback to fold_0
            checkpoint_path = latest_cv / 'fold_0_AX' / 'checkpoints' / 'best.pt'
        checkpoint_path = str(checkpoint_path)
        print(f"Using CV checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = 'outputs/checkpoints/best.pt'
    
    data_dir = 'data'
    output_dir = 'outputs/enhanced_analysis'
    
    # Initialize pipeline
    pipeline = EnhancedInference(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    
    # Load components
    pipeline.load_model()
    pipeline.load_lr_database()
    
    # Process all regions
    pipeline.run_all_regions()
    
    # Create cross-region comparison
    pipeline.create_cross_region_comparison()
    
    # Generate report
    pipeline.generate_report()
    
    print("\n✓ Enhanced analysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
