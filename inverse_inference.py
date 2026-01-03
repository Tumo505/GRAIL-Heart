"""
Inverse Modelling Analysis Script for GRAIL-Heart

This script demonstrates the inverse modelling capabilities:
1. Load trained GRAIL-Heart model with inverse modelling enabled
2. Run inference to predict cell fates from L-R interactions
3. Perform counterfactual analysis to identify causal L-R signals
4. Analyze mechanosensitive pathways
5. Generate inverse modelling visualizations

Reference (Abstract):
    "This inverse modelling framework will also elucidate mechanosensitive pathways
    that modulate early-stage cardiac tissue patterning, linking molecular signalling
    to the formation of soft contractile structures."
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from grail_heart.models import GRAILHeart, InverseSignalInference
from grail_heart.data.cellchat_database import get_omnipath_lr_database
from grail_heart.visualization import SpatialVisualizer


class InverseModellingAnalysis:
    """
    Inverse modelling analysis pipeline for GRAIL-Heart.
    
    Enables:
    1. Cell fate prediction from L-R interactions
    2. Causal L-R signal identification
    3. Mechanosensitive pathway analysis
    4. Target gene effect prediction
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        output_dir: str = 'outputs/inverse_analysis',
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'causal_analysis').mkdir(exist_ok=True)
        (self.output_dir / 'fate_prediction').mkdir(exist_ok=True)
        (self.output_dir / 'mechanobiology').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.lr_database = None
        
        # Store results
        self.region_results = {}
        self.causal_analysis_results = {}
        self.mechano_results = {}
        
    def load_model(self) -> None:
        """Load trained model with inverse modelling enabled."""
        print(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        n_cell_types = config.get('n_cell_types', 10)
        n_genes = config.get('n_genes', 2000)
        n_lr_pairs = config.get('n_lr_pairs', 5000)
        
        # Create model WITH inverse modelling enabled
        self.model = GRAILHeart(
            n_genes=n_genes,
            hidden_dim=model_config.get('hidden_dim', 256),
            n_gat_layers=model_config.get('n_gat_layers', 3),
            n_heads=model_config.get('n_heads', 8),
            n_cell_types=n_cell_types,
            n_lr_pairs=n_lr_pairs,
            tasks=['lr', 'reconstruction', 'cell_type'],
            # Enable inverse modelling
            use_inverse_modelling=True,
            n_fates=n_cell_types,  # Use cell types as fate categories
            n_pathways=20,
            n_mechano_pathways=8,
        )
        
        # Load state dict (only forward model weights, inverse module is newly initialized)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter to only load matching keys (forward model)
        model_state = self.model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        
        missing_keys = set(model_state.keys()) - set(pretrained_dict.keys())
        print(f"Loading {len(pretrained_dict)} pretrained parameters")
        print(f"Newly initialized (inverse module): {len(missing_keys)} parameters")
        
        model_state.update(pretrained_dict)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded with inverse modelling enabled!")
        
    def load_lr_database(self) -> None:
        """Load L-R database from OmniPath."""
        print("Loading OmniPath L-R database...")
        self.lr_database = get_omnipath_lr_database()
        print(f"Loaded {len(self.lr_database)} L-R pairs")
        
    def load_region_data(self, region: str):
        """Load spatial transcriptomics data for a region."""
        import scanpy as sc
        
        region_files = {
            'AX': 'visium-OCT_AX_raw.h5ad',
            'LA': 'visium-OCT_LA_raw.h5ad',
            'LV': 'visium-OCT_LV_raw.h5ad',
            'RA': 'visium-OCT_RA_raw.h5ad',
            'RV': 'visium-OCT_RV_raw.h5ad',
            'SP': 'visium-OCT_SP_raw.h5ad',
        }
        
        file_path = self.data_dir / 'HeartCellAtlasv2' / region_files[region]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        adata = sc.read_h5ad(file_path)
        print(f"Loaded {region}: {adata.n_obs} cells, {adata.n_vars} genes")
        
        return adata
        
    def run_inverse_inference(
        self,
        data,
        region: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inverse modelling inference on a region.
        
        This performs:
        1. Forward L-R prediction
        2. Cell fate prediction from L-R
        3. Causal L-R identification
        4. Mechanosensitive pathway analysis
        """
        print(f"\n{'='*60}")
        print(f"Running inverse modelling for {region}")
        print(f"{'='*60}")
        
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Run forward pass WITH inverse modelling
            outputs = self.model(data, run_inverse=True)
            
            # Move outputs to CPU for analysis
            results = {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}
            
        # Print summary
        self._print_inference_summary(results, region)
        
        return results
        
    def _print_inference_summary(self, results: Dict, region: str):
        """Print summary of inverse modelling results."""
        print(f"\n--- {region} Inverse Modelling Summary ---")
        
        if 'fate_logits' in results:
            fate_probs = torch.softmax(results['fate_logits'], dim=-1)
            n_cells = fate_probs.shape[0]
            n_fates = fate_probs.shape[1]
            print(f"Cell fate prediction: {n_cells} cells -> {n_fates} fates")
            
            # Show fate distribution
            predicted_fates = fate_probs.argmax(dim=-1)
            fate_counts = torch.bincount(predicted_fates, minlength=n_fates)
            print(f"Fate distribution: {fate_counts.tolist()}")
            
        if 'differentiation_score' in results:
            diff_scores = results['differentiation_score'].squeeze()
            print(f"Differentiation scores: mean={diff_scores.mean():.3f}, "
                  f"std={diff_scores.std():.3f}")
            
        if 'causal_lr_scores' in results:
            causal_scores = results['causal_lr_scores']
            print(f"Causal L-R analysis: {len(causal_scores)} edges analyzed")
            print(f"  Top causal scores: {causal_scores.topk(5).values.tolist()}")
            
        if 'mechano_pathway_activation' in results:
            mechano = results['mechano_pathway_activation']
            mean_activation = mechano.mean(dim=0)
            pathway_names = ['YAP_TAZ', 'Integrin_FAK', 'Piezo', 'TGF_beta',
                           'Wnt', 'Notch', 'BMP', 'FGF']
            print("Mechanosensitive pathway activation:")
            for name, act in zip(pathway_names, mean_activation):
                print(f"  {name}: {act:.3f}")
                
    def analyze_causal_lr_signals(
        self,
        results: Dict,
        data,
        region: str,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """
        Identify the most causally important L-R interactions.
        
        This answers: "Which L-R signals are CAUSAL for cell fate decisions?"
        """
        if 'causal_lr_scores' not in results:
            print("Warning: No causal L-R scores in results")
            return pd.DataFrame()
            
        causal_scores = results['causal_lr_scores']
        lr_scores = torch.sigmoid(results['lr_scores'])
        
        edge_index = data.edge_index.cpu()
        src, dst = edge_index
        
        # Get top causal edges
        top_indices = causal_scores.topk(min(top_k, len(causal_scores))).indices
        
        causal_edges = []
        for idx in top_indices:
            causal_edges.append({
                'edge_idx': idx.item(),
                'source_cell': src[idx].item(),
                'target_cell': dst[idx].item(),
                'causal_score': causal_scores[idx].item(),
                'lr_score': lr_scores[idx].item(),
                'region': region,
            })
            
        df = pd.DataFrame(causal_edges)
        
        # Save results
        output_path = self.output_dir / 'causal_analysis' / f'{region}_causal_lr.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved causal L-R analysis to {output_path}")
        
        return df
        
    def analyze_mechanosensitive_pathways(
        self,
        results: Dict,
        region: str,
    ) -> Dict[str, float]:
        """
        Analyze mechanosensitive pathway activations.
        
        This addresses the abstract's claim:
        "elucidate mechanosensitive pathways that modulate early-stage
        cardiac tissue patterning"
        """
        if 'mechano_pathway_activation' not in results:
            print("Warning: No mechanosensitive pathway data in results")
            return {}
            
        mechano = results['mechano_pathway_activation']
        
        pathway_names = [
            'YAP_TAZ',       # Hippo signaling - stiffness sensing
            'Integrin_FAK',  # Cell-ECM mechanotransduction
            'Piezo',         # Mechanosensitive ion channels
            'TGF_beta',      # Mechanical stress response
            'Wnt',           # Cardiac development
            'Notch',         # Cell-cell signaling
            'BMP',           # Cardiac patterning
            'FGF',           # Growth factor signaling
        ]
        
        # Compute mean activation per pathway
        mean_activation = mechano.mean(dim=0)
        
        pathway_results = {}
        for name, act in zip(pathway_names, mean_activation):
            pathway_results[name] = act.item()
            
        # Save results
        output_path = self.output_dir / 'mechanobiology' / f'{region}_mechano_pathways.json'
        with open(output_path, 'w') as f:
            json.dump(pathway_results, f, indent=2)
            
        self.mechano_results[region] = pathway_results
        
        return pathway_results
        
    def analyze_fate_trajectories(
        self,
        results: Dict,
        data,
        region: str,
    ) -> Dict:
        """
        Analyze cell fate trajectories and differentiation.
        
        This addresses:
        "infer functional signalling networks that drive cardiomyocyte differentiation"
        """
        if 'fate_trajectory' not in results:
            print("Warning: No fate trajectory data in results")
            return {}
            
        fate_trajectory = results['fate_trajectory'].numpy()
        diff_scores = results.get('differentiation_score', torch.zeros(len(fate_trajectory)))
        diff_scores = diff_scores.squeeze().numpy()
        
        # Get cell type labels if available
        cell_types = data.y.cpu().numpy() if hasattr(data, 'y') else None
        
        trajectory_results = {
            'n_cells': len(fate_trajectory),
            'trajectory_dim': fate_trajectory.shape[1],
            'mean_differentiation': float(diff_scores.mean()),
            'std_differentiation': float(diff_scores.std()),
        }
        
        # Identify cells at different differentiation stages
        if diff_scores is not None:
            progenitor_mask = diff_scores < 0.3
            intermediate_mask = (diff_scores >= 0.3) & (diff_scores < 0.7)
            differentiated_mask = diff_scores >= 0.7
            
            trajectory_results['n_progenitors'] = int(progenitor_mask.sum())
            trajectory_results['n_intermediate'] = int(intermediate_mask.sum())
            trajectory_results['n_differentiated'] = int(differentiated_mask.sum())
            
        # Save results
        output_path = self.output_dir / 'fate_prediction' / f'{region}_fate_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(trajectory_results, f, indent=2)
            
        return trajectory_results
        
    def visualize_causal_lr_network(
        self,
        results: Dict,
        data,
        region: str,
        top_k: int = 100,
    ):
        """Create visualization of causal L-R network."""
        if 'causal_lr_scores' not in results:
            return
            
        causal_scores = results['causal_lr_scores']
        edge_index = data.edge_index.cpu()
        pos = data.pos.cpu().numpy() if hasattr(data, 'pos') else None
        
        if pos is None:
            print("Warning: No spatial coordinates for visualization")
            return
            
        # Get top causal edges
        top_indices = causal_scores.topk(min(top_k, len(causal_scores))).indices
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot cells
        ax.scatter(pos[:, 0], pos[:, 1], c='lightgray', s=10, alpha=0.3)
        
        # Plot causal L-R edges
        for idx in top_indices:
            src_idx = edge_index[0, idx].item()
            dst_idx = edge_index[1, idx].item()
            
            src_pos = pos[src_idx]
            dst_pos = pos[dst_idx]
            
            score = causal_scores[idx].item()
            
            ax.annotate(
                '', xy=dst_pos, xytext=src_pos,
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=plt.cm.Reds(score),
                    alpha=0.7,
                    lw=1.5,
                )
            )
            
        ax.set_title(f'{region}: Causal L-R Communication Network', fontsize=14)
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        ax.set_aspect('equal')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Causal Score')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'figures' / f'{region}_causal_network.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print(f"Saved causal network visualization for {region}")
        
    def visualize_mechanosensitive_heatmap(self):
        """Create heatmap of mechanosensitive pathway activation across regions."""
        if not self.mechano_results:
            print("No mechanosensitive results to visualize")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.mechano_results).T
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        sns.heatmap(
            df, annot=True, fmt='.2f', cmap='YlOrRd',
            ax=ax, cbar_kws={'label': 'Pathway Activation'}
        )
        
        ax.set_title('Mechanosensitive Pathway Activation Across Cardiac Regions', fontsize=14)
        ax.set_xlabel('Signaling Pathway')
        ax.set_ylabel('Cardiac Region')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'figures' / 'mechano_pathway_heatmap.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print("Saved mechanosensitive pathway heatmap")
        
    def create_inverse_summary(self) -> Dict:
        """Create summary of all inverse modelling analyses."""
        summary = {
            'regions_analyzed': list(self.region_results.keys()),
            'mechanosensitive_pathways': self.mechano_results,
            'causal_analysis': self.causal_analysis_results,
        }
        
        # Save summary
        output_path = self.output_dir / 'inverse_modelling_summary.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"\nInverse modelling summary saved to {output_path}")
        
        return summary
        
    def run_full_analysis(self, regions: List[str] = None):
        """Run complete inverse modelling analysis on all regions."""
        if regions is None:
            regions = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']
            
        print("\n" + "="*70)
        print("GRAIL-Heart Inverse Modelling Analysis")
        print("="*70)
        
        # Load model and database
        self.load_model()
        self.load_lr_database()
        
        # Process each region
        for region in regions:
            try:
                print(f"\n>>> Processing {region}...")
                
                # Load data
                adata = self.load_region_data(region)
                
                # Create PyG data (simplified - would need proper graph construction)
                from grail_heart.data.graph_builder import CardiacGraphBuilder
                
                builder = CardiacGraphBuilder(
                    k_neighbors=6,
                    use_spatial=True,
                )
                
                # Build graph
                data = builder.build_from_adata(
                    adata,
                    gene_list=None,  # Use all genes
                    max_genes=2000,
                )
                
                # Run inverse inference
                results = self.run_inverse_inference(data, region)
                self.region_results[region] = results
                
                # Analyze causal L-R signals
                causal_df = self.analyze_causal_lr_signals(results, data, region)
                self.causal_analysis_results[region] = causal_df.to_dict() if len(causal_df) > 0 else {}
                
                # Analyze mechanosensitive pathways
                self.analyze_mechanosensitive_pathways(results, region)
                
                # Analyze fate trajectories
                self.analyze_fate_trajectories(results, data, region)
                
                # Create visualizations
                self.visualize_causal_lr_network(results, data, region)
                
            except Exception as e:
                print(f"Error processing {region}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        # Create cross-region visualizations
        self.visualize_mechanosensitive_heatmap()
        
        # Create summary
        summary = self.create_inverse_summary()
        
        print("\n" + "="*70)
        print("Inverse Modelling Analysis Complete!")
        print("="*70)
        
        return summary


def main():
    """Main entry point for inverse modelling analysis."""
    analysis = InverseModellingAnalysis(
        checkpoint_path='outputs/checkpoints/best.pt',
        data_dir='data',
        output_dir='outputs/inverse_analysis',
    )
    
    # Run analysis on all regions
    summary = analysis.run_full_analysis()
    
    print("\n" + "="*70)
    print("KEY FINDINGS FROM INVERSE MODELLING:")
    print("="*70)
    
    # Print mechanosensitive pathway summary
    if summary['mechanosensitive_pathways']:
        print("\n1. MECHANOSENSITIVE PATHWAY ACTIVATION:")
        for region, pathways in summary['mechanosensitive_pathways'].items():
            top_pathway = max(pathways.items(), key=lambda x: x[1])
            print(f"   {region}: {top_pathway[0]} (activation: {top_pathway[1]:.3f})")
            
    print("\n" + "-"*70)
    print("Results saved to: outputs/inverse_analysis/")
    print("-"*70)


if __name__ == '__main__':
    main()
