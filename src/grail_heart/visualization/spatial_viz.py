"""
Spatial Visualization Module for GRAIL-Heart

Creates publication-ready visualizations of:
- Cell-cell communication networks on spatial coordinates
- L-R interaction heatmaps
- Signaling pathway activity maps
- Network topology plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import warnings
warnings.filterwarnings('ignore')


# Define custom colormaps
COMMUNICATION_CMAP = LinearSegmentedColormap.from_list(
    'communication',
    ['#f7fbff', '#6baed6', '#2171b5', '#08306b'],
    N=256
)

PATHWAY_COLORS = {
    'VEGF': '#e41a1c',
    'PDGF': '#377eb8',
    'FGF': '#4daf4a',
    'EGF': '#984ea3',
    'TGFb': '#ff7f00',
    'BMP': '#ffff33',
    'WNT': '#a65628',
    'NOTCH': '#f781bf',
    'Chemokine': '#999999',
    'Interleukin': '#66c2a5',
    'TNF': '#fc8d62',
    'ECM': '#8da0cb',
    'Cardiac': '#e78ac3',
    'Semaphorin': '#a6d854',
    'Ephrin': '#ffd92f',
    'Neurotrophin': '#e5c494',
    'Hedgehog': '#b3b3b3',
    'Adhesion': '#8dd3c7',
    'Complement': '#bebada',
    'Interferon': '#fb8072',
    'CSF': '#80b1d3',
    'RAS': '#fdb462',
    'Angiogenesis': '#b3de69',
    'GDF': '#fccde5',
    'Activin': '#d9d9d9',
    'Netrin': '#bc80bd',
    'Slit': '#ccebc5',
    'Lectin': '#ffed6f',
    'TAM': '#1f78b4',
    'DAMP': '#33a02c',
    'Neuregulin': '#fb9a99',
}


class SpatialVisualizer:
    """
    Visualize spatial cell-cell communication networks.
    """
    
    def __init__(
        self,
        output_dir: str = 'outputs/figures',
        fig_format: str = 'png',
        dpi: int = 300,
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save figures
            fig_format: Output format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_format = fig_format
        self.dpi = dpi
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['figure.titlesize'] = 16
    
    def plot_spatial_communication(
        self,
        coords: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
        cell_colors: Optional[np.ndarray] = None,
        title: str = 'Spatial Communication Network',
        cell_size: float = 20,
        edge_alpha: float = 0.3,
        edge_width_scale: float = 1.0,
        top_edges: int = 5000,
        show_colorbar: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cell-cell communication network on spatial coordinates.
        
        Args:
            coords: Spatial coordinates [n_cells, 2]
            edge_index: Edge indices [2, n_edges]
            edge_weights: Optional weights for edges
            cell_colors: Optional color values for cells
            title: Plot title
            cell_size: Size of cell points
            edge_alpha: Transparency of edges
            edge_width_scale: Scale factor for edge widths
            top_edges: Number of top edges to plot (for clarity)
            show_colorbar: Whether to show colorbar
            figsize: Figure size
            save_name: Filename to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort edges by weight and select top
        if edge_weights is not None:
            sorted_idx = np.argsort(edge_weights)[::-1][:top_edges]
            edge_weights = edge_weights[sorted_idx]
            edge_index = edge_index[:, sorted_idx]
        else:
            if edge_index.shape[1] > top_edges:
                idx = np.random.choice(edge_index.shape[1], top_edges, replace=False)
                edge_index = edge_index[:, idx]
        
        # Create line segments for edges
        segments = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            segments.append([coords[src], coords[dst]])
        
        # Plot edges
        if len(segments) > 0:
            if edge_weights is not None:
                # Normalize weights for coloring
                norm = Normalize(vmin=0, vmax=np.percentile(edge_weights, 95))
                colors = cm.Blues(norm(edge_weights))
                widths = 0.5 + edge_width_scale * norm(edge_weights) * 2
                
                lc = LineCollection(
                    segments, 
                    colors=colors, 
                    linewidths=widths,
                    alpha=edge_alpha
                )
            else:
                lc = LineCollection(
                    segments,
                    colors='steelblue',
                    linewidths=0.5,
                    alpha=edge_alpha
                )
            ax.add_collection(lc)
        
        # Plot cells
        if cell_colors is not None:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=cell_colors,
                s=cell_size,
                cmap='YlOrRd',
                alpha=0.8,
                edgecolors='none',
                zorder=2
            )
            if show_colorbar:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Communication Score', fontsize=10)
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c='#2c3e50',
                s=cell_size,
                alpha=0.6,
                edgecolors='none',
                zorder=2
            )
        
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(
                self.output_dir / f'{save_name}.{self.fig_format}',
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white'
            )
        
        return fig
    
    def plot_lr_heatmap(
        self,
        lr_scores: pd.DataFrame,
        top_n: int = 30,
        figsize: Tuple[int, int] = (14, 10),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap of L-R interaction scores.
        
        Args:
            lr_scores: DataFrame with L-R scores (regions as columns)
            top_n: Number of top L-R pairs to show
            figsize: Figure size
            save_name: Filename to save
            
        Returns:
            matplotlib Figure
        """
        # Prepare data - get columns that look like region names
        score_cols = [c for c in lr_scores.columns if c not in 
                      ['ligand', 'receptor', 'pathway', 'function', 'lr_pair']]
        
        if not score_cols:
            # Single region case
            lr_scores = lr_scores.sort_values('mean_score', ascending=False).head(top_n)
            lr_scores['lr_pair'] = lr_scores['ligand'] + ' → ' + lr_scores['receptor']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(lr_scores)), lr_scores['mean_score'].values)
            
            # Color by pathway
            for i, (_, row) in enumerate(lr_scores.iterrows()):
                color = PATHWAY_COLORS.get(row['pathway'], '#666666')
                bars[i].set_color(color)
            
            ax.set_yticks(range(len(lr_scores)))
            ax.set_yticklabels(lr_scores['lr_pair'].values)
            ax.set_xlabel('Interaction Score')
            ax.set_title('Top L-R Interactions', fontweight='bold')
            ax.invert_yaxis()
            
            # Add legend
            unique_pathways = lr_scores['pathway'].unique()
            patches = [mpatches.Patch(color=PATHWAY_COLORS.get(p, '#666'), label=p) 
                       for p in unique_pathways]
            ax.legend(handles=patches, loc='lower right', fontsize=8)
            
        else:
            # Multi-region heatmap
            # Create lr_pair column
            lr_scores = lr_scores.copy()
            lr_scores['lr_pair'] = lr_scores['ligand'] + ' → ' + lr_scores['receptor']
            
            # Get top pairs by mean score across regions
            lr_scores['mean_all'] = lr_scores[score_cols].mean(axis=1)
            top_pairs = lr_scores.nlargest(top_n, 'mean_all')
            
            # Create matrix
            matrix = top_pairs[score_cols].values
            
            fig, ax = plt.subplots(figsize=figsize)
            
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(len(score_cols)))
            ax.set_xticklabels(score_cols, rotation=45, ha='right')
            ax.set_yticks(range(len(top_pairs)))
            ax.set_yticklabels(top_pairs['lr_pair'].values)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Interaction Score', fontsize=10)
            
            ax.set_xlabel('Region')
            ax.set_ylabel('L-R Pair')
            ax.set_title('L-R Interaction Scores Across Regions', fontweight='bold')
            
            # Add grid
            ax.set_xticks(np.arange(-0.5, len(score_cols), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(top_pairs), 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(
                self.output_dir / f'{save_name}.{self.fig_format}',
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white'
            )
        
        return fig
    
    def plot_pathway_activity(
        self,
        lr_scores: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot pathway activity summary.
        
        Args:
            lr_scores: DataFrame with L-R scores
            figsize: Figure size
            save_name: Filename to save
            
        Returns:
            matplotlib Figure
        """
        # Aggregate by pathway
        if 'pathway' not in lr_scores.columns:
            raise ValueError("DataFrame must have 'pathway' column")
        
        # Get score columns
        score_cols = [c for c in lr_scores.columns if c not in 
                      ['ligand', 'receptor', 'pathway', 'function', 'lr_pair',
                       'mean_score', 'max_score', 'total_score', 'n_interactions', 'pct_edges']]
        
        if score_cols:
            # Multi-region case
            pathway_scores = lr_scores.groupby('pathway')[score_cols].mean()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create grouped bar chart
            n_pathways = len(pathway_scores)
            n_regions = len(score_cols)
            x = np.arange(n_pathways)
            width = 0.8 / n_regions
            
            for i, col in enumerate(score_cols):
                offset = (i - n_regions/2 + 0.5) * width
                bars = ax.bar(x + offset, pathway_scores[col], width, 
                             label=col, alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(pathway_scores.index, rotation=45, ha='right')
            ax.set_ylabel('Mean Interaction Score')
            ax.set_title('Pathway Activity Across Regions', fontweight='bold')
            ax.legend(title='Region', bbox_to_anchor=(1.02, 1), loc='upper left')
            
        else:
            # Single region - use total or mean score
            score_col = 'total_score' if 'total_score' in lr_scores.columns else 'mean_score'
            pathway_scores = lr_scores.groupby('pathway')[score_col].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            colors = [PATHWAY_COLORS.get(p, '#666666') for p in pathway_scores.index]
            bars = ax.barh(range(len(pathway_scores)), pathway_scores.values, color=colors)
            
            ax.set_yticks(range(len(pathway_scores)))
            ax.set_yticklabels(pathway_scores.index)
            ax.set_xlabel('Total Pathway Activity')
            ax.set_title('Signaling Pathway Activity', fontweight='bold')
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(
                self.output_dir / f'{save_name}.{self.fig_format}',
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white'
            )
        
        return fig
    
    def plot_specific_lr_spatial(
        self,
        coords: np.ndarray,
        edge_index: np.ndarray,
        expression: np.ndarray,
        gene_names: List[str],
        ligand: str,
        receptor: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot spatial distribution of a specific L-R pair.
        
        Args:
            coords: Spatial coordinates [n_cells, 2]
            edge_index: Edge indices [2, n_edges]
            expression: Gene expression [n_cells, n_genes]
            gene_names: Gene names
            ligand: Ligand gene name
            receptor: Receptor gene name
            title: Optional title
            figsize: Figure size
            save_name: Filename to save
            
        Returns:
            matplotlib Figure
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        if ligand not in gene_to_idx or receptor not in gene_to_idx:
            raise ValueError(f"Gene not found: {ligand} or {receptor}")
        
        lig_idx = gene_to_idx[ligand]
        rec_idx = gene_to_idx[receptor]
        
        lig_expr = expression[:, lig_idx]
        rec_expr = expression[:, rec_idx]
        
        # Compute communication scores per cell
        # For each cell, sum incoming L-R scores
        comm_scores = np.zeros(len(coords))
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            score = lig_expr[src] * rec_expr[dst]
            comm_scores[dst] += score  # Receiver gets the score
        
        # Normalize
        if comm_scores.max() > 0:
            comm_scores = comm_scores / comm_scores.max()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot ligand expression
        sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=lig_expr,
                              s=10, cmap='Reds', alpha=0.8)
        axes[0].set_title(f'{ligand} (Ligand)', fontweight='bold')
        axes[0].set_xlabel('Spatial X')
        axes[0].set_ylabel('Spatial Y')
        axes[0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0], shrink=0.8, label='Expression')
        
        # Plot receptor expression
        sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=rec_expr,
                              s=10, cmap='Blues', alpha=0.8)
        axes[1].set_title(f'{receptor} (Receptor)', fontweight='bold')
        axes[1].set_xlabel('Spatial X')
        axes[1].set_ylabel('Spatial Y')
        axes[1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[1], shrink=0.8, label='Expression')
        
        # Plot communication score
        sc3 = axes[2].scatter(coords[:, 0], coords[:, 1], c=comm_scores,
                              s=10, cmap='Purples', alpha=0.8)
        axes[2].set_title(f'{ligand}→{receptor} Signal', fontweight='bold')
        axes[2].set_xlabel('Spatial X')
        axes[2].set_ylabel('Spatial Y')
        axes[2].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[2], shrink=0.8, label='Communication')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(
                self.output_dir / f'{save_name}.{self.fig_format}',
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white'
            )
        
        return fig
    
    def plot_region_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'mean_score',
        top_n: int = 20,
        figsize: Tuple[int, int] = (14, 10),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create multi-panel comparison across regions.
        
        Args:
            comparison_df: DataFrame with L-R scores per region (wide format)
            metric: Score metric column name
            top_n: Number of top pairs to show
            figsize: Figure size
            save_name: Filename to save
            
        Returns:
            matplotlib Figure
        """
        # Get region columns
        region_cols = [c for c in comparison_df.columns if c not in 
                       ['ligand', 'receptor', 'pathway', 'function', 'lr_pair']]
        
        n_regions = len(region_cols)
        
        if n_regions == 0:
            raise ValueError("No region columns found in DataFrame")
        
        # Create figure
        n_cols = min(3, n_regions)
        n_rows = (n_regions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        
        for idx, region in enumerate(region_cols):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            # Get top pairs for this region
            region_data = comparison_df[['ligand', 'receptor', 'pathway', region]].copy()
            region_data = region_data.nlargest(top_n, region)
            region_data['lr_pair'] = region_data['ligand'] + '→' + region_data['receptor']
            
            # Plot horizontal bars
            colors = [PATHWAY_COLORS.get(p, '#666') for p in region_data['pathway']]
            ax.barh(range(len(region_data)), region_data[region].values, color=colors)
            
            ax.set_yticks(range(len(region_data)))
            ax.set_yticklabels(region_data['lr_pair'], fontsize=8)
            ax.set_xlabel('Score')
            ax.set_title(f'{region}', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        # Hide empty subplots
        for idx in range(n_regions, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        # Add legend
        unique_pathways = comparison_df['pathway'].unique()
        patches = [mpatches.Patch(color=PATHWAY_COLORS.get(p, '#666'), label=p) 
                   for p in unique_pathways[:15]]  # Limit legend items
        fig.legend(handles=patches, loc='center right', 
                   bbox_to_anchor=(1.15, 0.5), fontsize=8)
        
        fig.suptitle('L-R Interactions by Region', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            fig.savefig(
                self.output_dir / f'{save_name}.{self.fig_format}',
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white'
            )
        
        return fig
    
    def plot_network_summary(
        self,
        n_cells_per_region: Dict[str, int],
        n_interactions_per_region: Dict[str, int],
        top_pathways_per_region: Dict[str, List[str]],
        figsize: Tuple[int, int] = (14, 6),
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create summary dashboard of network statistics.
        
        Args:
            n_cells_per_region: Number of cells per region
            n_interactions_per_region: Number of interactions per region
            top_pathways_per_region: Top active pathways per region
            figsize: Figure size
            save_name: Filename to save
            
        Returns:
            matplotlib Figure
        """
        regions = list(n_cells_per_region.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Panel 1: Number of cells
        cells = [n_cells_per_region[r] for r in regions]
        bars1 = axes[0].bar(regions, cells, color='steelblue', alpha=0.8)
        axes[0].set_ylabel('Number of Cells')
        axes[0].set_title('Cells per Region', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, cells):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val:,}', ha='center', va='bottom', fontsize=9)
        
        # Panel 2: Number of interactions
        interactions = [n_interactions_per_region.get(r, 0) for r in regions]
        bars2 = axes[1].bar(regions, interactions, color='coral', alpha=0.8)
        axes[1].set_ylabel('Number of L-R Interactions')
        axes[1].set_title('Detected Interactions', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, interactions):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', fontsize=9)
        
        # Panel 3: Top pathways (text summary)
        axes[2].axis('off')
        text_lines = ['Top Active Pathways:\n']
        for region in regions:
            pathways = top_pathways_per_region.get(region, [])[:3]
            text_lines.append(f'{region}: {", ".join(pathways) if pathways else "N/A"}')
        
        axes[2].text(0.1, 0.9, '\n'.join(text_lines),
                    transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].set_title('Pathway Summary', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(
                self.output_dir / f'{save_name}.{self.fig_format}',
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white'
            )
        
        return fig


def main():
    """Test the visualizer with sample data."""
    import numpy as np
    
    # Create sample data
    n_cells = 500
    n_edges = 2000
    
    coords = np.random.randn(n_cells, 2) * 100
    edge_index = np.random.randint(0, n_cells, size=(2, n_edges))
    edge_weights = np.random.exponential(1, n_edges)
    cell_colors = np.random.rand(n_cells)
    
    # Create visualizer
    viz = SpatialVisualizer(output_dir='outputs/figures')
    
    # Test spatial communication plot
    fig = viz.plot_spatial_communication(
        coords=coords,
        edge_index=edge_index,
        edge_weights=edge_weights,
        cell_colors=cell_colors,
        title='Test Communication Network',
        save_name='test_network'
    )
    plt.show()
    
    print("Visualization test complete!")


if __name__ == '__main__':
    main()
