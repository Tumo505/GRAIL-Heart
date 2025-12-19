"""
GRAIL-Heart Inference Script

Extract cell-cell communication networks from the trained model using
GAT attention weights and ligand-receptor co-expression patterns.
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from src.grail_heart.models.grail_heart import GRAILHeart
from src.grail_heart.data.datasets import SpatialTranscriptomicsDataset
from src.grail_heart.data.graph_builder import SpatialGraphBuilder
from src.grail_heart.data.lr_database import LigandReceptorDatabase
from torch_geometric.loader import DataLoader


class GRAILHeartInference:
    """
    Inference class for extracting cell-cell communication networks.
    """
    
    def __init__(
        self,
        checkpoint_path: str = 'outputs/checkpoints/best.pt',
        config_path: str = 'outputs/config.yaml',
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load L-R database
        self.lr_db = LigandReceptorDatabase()
        print(f"Loaded {len(self.lr_db.lr_pairs)} L-R pairs")
        
        self.model = None
        self.datasets = []
        self.graphs = []
        self.region_names = []
        
    def load_model(self, n_genes: int = 2000):
        """Load trained model from checkpoint."""
        model_config = self.config['model']
        
        self.model = GRAILHeart(
            n_genes=n_genes,
            n_cell_types=None,
            hidden_dim=model_config['hidden_dim'],
            n_gat_layers=model_config['n_gat_layers'],
            n_heads=model_config['n_heads'],
            n_edge_types=model_config.get('n_edge_types', 2),
            encoder_dims=model_config.get('encoder_dims', [512, 256]),
            dropout=model_config['dropout'],
            use_spatial=model_config.get('use_spatial', True),
            use_variational=model_config.get('use_variational', False),
            tasks=model_config.get('tasks', ['lr', 'reconstruction']),
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        
    def load_data(self, data_dir: str = 'data/HeartCellAtlasv2'):
        """Load all cardiac region datasets."""
        data_dir = Path(data_dir)
        h5ad_files = sorted(data_dir.glob('*.h5ad'))
        
        data_config = self.config['data']
        
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
                    self.datasets.append(ds)
                    # Extract region name from filename
                    region = f.stem.replace('visium-OCT_', '').replace('_raw', '')
                    self.region_names.append(region)
                    print(f"  Loaded: {ds.n_cells} cells ({region})")
            except Exception as e:
                print(f"  Failed: {e}")
        
        # Build graphs
        print("\nBuilding graphs...")
        graph_builder = SpatialGraphBuilder(
            method=data_config['graph_method'],
            k=data_config['k_neighbors'],
        )
        
        for i, ds in enumerate(self.datasets):
            lr_edge_dict = self.lr_db.to_edge_dict({g: i for i, g in enumerate(ds.gene_names)})
            graph = graph_builder.build_graph(
                expression=ds.expression,
                spatial_coords=ds.spatial_coords,
                cell_types=ds.cell_types,
            )
            if lr_edge_dict['n_edges'] > 0:
                edge_type = torch.zeros(graph.edge_index.shape[1], dtype=torch.long)
                graph.edge_type = edge_type
            self.graphs.append(graph)
            print(f"  {self.region_names[i]}: {graph.num_nodes} nodes, {graph.num_edges} edges")
            
    def extract_attention_weights(
        self,
        graph_idx: int = 0,
        layer_idx: int = -1,  # Last layer by default
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention weights from GAT layers.
        
        Returns:
            edge_index: [2, E] edge indices
            attention_weights: [E, n_heads] attention per head
        """
        graph = self.graphs[graph_idx].to(self.device)
        
        # Register hooks to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            # GAT layers return (output, attention) when return_attention=True
            if isinstance(output, tuple) and len(output) == 2:
                attention_weights.append(output[1])
        
        # Get GAT layers
        gat_layers = self.model.gat_encoder.gat_layers
        
        # Register hook on specified layer
        target_layer = gat_layers[layer_idx]
        handle = target_layer.register_forward_hook(attention_hook)
        
        # Forward pass
        with torch.no_grad():
            # We need to modify forward to get attention
            # For now, compute edge-level scores from embeddings
            outputs = self.model(graph)
        
        handle.remove()
        
        # Get edge attention from L-R scores if available
        lr_scores = outputs.get('lr_scores', None)
        
        return graph.edge_index.cpu(), lr_scores.cpu() if lr_scores is not None else None
    
    def compute_cell_communication_scores(
        self,
        graph_idx: int = 0,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Compute cell-cell communication scores based on learned embeddings.
        
        Returns:
            DataFrame with source_cell, target_cell, score columns
        """
        graph = self.graphs[graph_idx].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(graph)
            
            # Get cell embeddings
            embeddings = outputs.get('embeddings', None)
            if embeddings is None:
                # Extract from intermediate representations
                embeddings = self.model.multimodal_encoder(
                    graph.x, 
                    graph.pos if hasattr(graph, 'pos') else None,
                    None
                )
            
            # Compute pairwise similarity for edges
            edge_index = graph.edge_index
            src_emb = embeddings[edge_index[0]]
            dst_emb = embeddings[edge_index[1]]
            
            # Cosine similarity as communication score
            scores = torch.nn.functional.cosine_similarity(src_emb, dst_emb, dim=1)
            
            # Get L-R scores if available
            lr_scores = outputs.get('lr_scores', None)
            if lr_scores is not None:
                # Combine embedding similarity with L-R prediction
                combined_scores = 0.5 * scores + 0.5 * torch.sigmoid(lr_scores.squeeze())
            else:
                combined_scores = scores
        
        # Create DataFrame
        edge_index_np = edge_index.cpu().numpy()
        scores_np = combined_scores.cpu().numpy()
        
        df = pd.DataFrame({
            'source_cell': edge_index_np[0],
            'target_cell': edge_index_np[1],
            'communication_score': scores_np
        })
        
        # Filter by threshold
        df = df[df['communication_score'] > threshold].sort_values(
            'communication_score', ascending=False
        )
        
        return df
    
    def identify_lr_interactions(
        self,
        graph_idx: int = 0,
        top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Identify top ligand-receptor interactions in a region.
        """
        ds = self.datasets[graph_idx]
        graph = self.graphs[graph_idx]
        
        # Get gene expression
        expression = ds.expression.numpy()
        gene_names = ds.gene_names
        
        # Create gene name to index mapping
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        # Compute L-R co-expression scores
        lr_scores = []
        
        for _, row in self.lr_db.lr_pairs.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            if ligand in gene_to_idx and receptor in gene_to_idx:
                lig_idx = gene_to_idx[ligand]
                rec_idx = gene_to_idx[receptor]
                
                # Get expression vectors
                lig_expr = expression[:, lig_idx]
                rec_expr = expression[:, rec_idx]
                
                # For each edge, compute L-R score
                edge_index = graph.edge_index.numpy()
                
                # Source cell expresses ligand, target cell expresses receptor
                lig_src = lig_expr[edge_index[0]]
                rec_dst = rec_expr[edge_index[1]]
                
                # Product as interaction strength
                interaction_strength = lig_src * rec_dst
                total_score = np.mean(interaction_strength[interaction_strength > 0])
                
                if not np.isnan(total_score) and total_score > 0:
                    lr_scores.append({
                        'ligand': ligand,
                        'receptor': receptor,
                        'mean_score': total_score,
                        'n_interactions': np.sum(interaction_strength > 0),
                        'pathway': row.get('pathway', 'Unknown')
                    })
        
        df = pd.DataFrame(lr_scores)
        if len(df) > 0:
            df = df.sort_values('mean_score', ascending=False).head(top_k)
        
        return df
    
    def compare_regions(
        self,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        Compare L-R interactions across cardiac regions.
        """
        region_interactions = {}
        
        for i, region in enumerate(self.region_names):
            print(f"Analyzing {region}...")
            lr_df = self.identify_lr_interactions(graph_idx=i, top_k=top_k)
            if len(lr_df) > 0:
                lr_df['region'] = region
                region_interactions[region] = lr_df
        
        # Combine all regions
        if region_interactions:
            combined = pd.concat(region_interactions.values(), ignore_index=True)
            return combined
        return pd.DataFrame()
    
    def plot_region_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_path: str = 'outputs/region_comparison.png',
        top_k: int = 15,
    ):
        """Plot L-R interaction comparison across regions."""
        if len(comparison_df) == 0:
            print("No data to plot")
            return
            
        # Get top interactions overall
        top_pairs = comparison_df.groupby(['ligand', 'receptor'])['mean_score'].mean()
        top_pairs = top_pairs.nlargest(top_k).index.tolist()
        
        # Filter to top pairs
        plot_df = comparison_df[
            comparison_df.apply(lambda x: (x['ligand'], x['receptor']) in top_pairs, axis=1)
        ]
        
        # Create pivot table
        plot_df['pair'] = plot_df['ligand'] + ' → ' + plot_df['receptor']
        pivot = plot_df.pivot_table(
            index='pair', 
            columns='region', 
            values='mean_score',
            aggfunc='mean'
        ).fillna(0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            pivot,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            ax=ax,
            cbar_kws={'label': 'Mean L-R Score'}
        )
        ax.set_title('Ligand-Receptor Interactions Across Cardiac Regions', fontsize=14)
        ax.set_xlabel('Cardiac Region', fontsize=12)
        ax.set_ylabel('Ligand → Receptor Pair', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to {output_path}")
    
    def extract_signaling_network(
        self,
        graph_idx: int = 0,
        score_threshold: float = 0.3,
    ) -> Dict:
        """
        Extract complete signaling network for a region.
        
        Returns dict with:
        - nodes: cell information
        - edges: communication edges with scores
        - lr_interactions: top L-R pairs
        """
        region = self.region_names[graph_idx]
        ds = self.datasets[graph_idx]
        graph = self.graphs[graph_idx]
        
        # Get cell-cell communication
        comm_df = self.compute_cell_communication_scores(
            graph_idx=graph_idx,
            threshold=score_threshold
        )
        
        # Get L-R interactions
        lr_df = self.identify_lr_interactions(graph_idx=graph_idx)
        
        # Create network structure
        network = {
            'region': region,
            'n_cells': ds.n_cells,
            'n_genes': ds.n_genes,
            'n_edges': len(comm_df),
            'spatial_coords': ds.spatial_coords.numpy().tolist(),
            'top_communications': comm_df.head(100).to_dict('records'),
            'lr_interactions': lr_df.to_dict('records') if len(lr_df) > 0 else [],
        }
        
        return network
    
    def run_full_analysis(
        self,
        output_dir: str = 'outputs/analysis',
    ):
        """Run complete signaling network analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("GRAIL-Heart Signaling Network Analysis")
        print("="*60)
        
        # Analyze each region
        all_networks = {}
        for i, region in enumerate(self.region_names):
            print(f"\n--- {region} ---")
            network = self.extract_signaling_network(graph_idx=i)
            all_networks[region] = network
            
            print(f"  Cells: {network['n_cells']}")
            print(f"  High-confidence edges: {network['n_edges']}")
            print(f"  L-R pairs detected: {len(network['lr_interactions'])}")
            
            if network['lr_interactions']:
                top3 = network['lr_interactions'][:3]
                print("  Top L-R interactions:")
                for lr in top3:
                    print(f"    {lr['ligand']} → {lr['receptor']}: {lr['mean_score']:.3f}")
        
        # Compare regions
        print("\n--- Cross-Region Comparison ---")
        comparison = self.compare_regions(top_k=20)
        
        if len(comparison) > 0:
            # Save comparison
            comparison.to_csv(output_dir / 'lr_comparison.csv', index=False)
            print(f"Saved L-R comparison to {output_dir / 'lr_comparison.csv'}")
            
            # Plot
            self.plot_region_comparison(
                comparison,
                output_path=str(output_dir / 'region_comparison.png')
            )
        
        # Save full analysis
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        for region, network in all_networks.items():
            network_clean = {
                k: convert_numpy(v) if not isinstance(v, list) else v
                for k, v in network.items()
            }
            with open(output_dir / f'{region}_network.json', 'w') as f:
                json.dump(network_clean, f, indent=2, default=convert_numpy)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}")
        
        return all_networks, comparison


def main():
    """Run GRAIL-Heart inference and analysis."""
    
    # Initialize inference
    inference = GRAILHeartInference(
        checkpoint_path='outputs/checkpoints/best.pt',
        config_path='outputs/config.yaml',
        device='cuda'
    )
    
    # Load data
    inference.load_data()
    
    # Load model
    if inference.datasets:
        n_genes = inference.datasets[0].n_genes
        inference.load_model(n_genes=n_genes)
    
        # Run analysis
        networks, comparison = inference.run_full_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Regions analyzed: {len(networks)}")
        print(f"Total L-R interactions found: {len(comparison)}")
        
        if len(comparison) > 0:
            # Top interactions across all regions
            top_overall = comparison.groupby(['ligand', 'receptor'])['mean_score'].mean()
            top_overall = top_overall.nlargest(10)
            
            print("\nTop 10 L-R Interactions (all regions):")
            for (lig, rec), score in top_overall.items():
                print(f"  {lig} → {rec}: {score:.4f}")


if __name__ == '__main__':
    main()
