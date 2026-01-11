"""
GRAIL-Heart Results Analysis and Visualization

Analyzes ablation study and benchmark results, generating publication-ready
figures and tables for the methods paper.

Usage:
    python analyze_results.py --ablation outputs/ablation/ablation_results_*.csv
    python analyze_results.py --benchmark outputs/benchmarks/benchmark_results_*.csv
    python analyze_results.py --all
"""

import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


class ResultsAnalyzer:
    """
    Analyzes and visualizes ablation and benchmark results.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        
    def load_ablation_results(self, pattern: str) -> pd.DataFrame:
        """Load ablation results from CSV files."""
        files = glob.glob(pattern)
        if not files:
            print(f"No files found matching: {pattern}")
            return pd.DataFrame()
            
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} ablation experiment results")
        return df
    
    def load_benchmark_results(self, pattern: str) -> pd.DataFrame:
        """Load benchmark results from CSV files."""
        files = glob.glob(pattern)
        if not files:
            print(f"No files found matching: {pattern}")
            return pd.DataFrame()
            
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} benchmark results")
        return df
    
    # =========================================================================
    # ABLATION ANALYSIS
    # =========================================================================
    
    def analyze_ablation_gat_layers(self, df: pd.DataFrame) -> None:
        """Analyze impact of GAT layer depth."""
        subset = df[df['ablation_type'] == 'gat_layers'].copy()
        if len(subset) == 0:
            return
            
        # Extract layer count
        subset['n_layers'] = subset['ablation_setting'].str.extract(r'(\d+)').astype(int)
        subset = subset.sort_values('n_layers')
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # L-R AUROC
        ax = axes[0]
        ax.plot(subset['n_layers'], subset['lr_auroc'], 'o-', color='#2ecc71', linewidth=2, markersize=8)
        ax.set_xlabel('Number of GAT Layers')
        ax.set_ylabel('L-R AUROC')
        ax.set_title('L-R Prediction Performance')
        ax.set_xticks(subset['n_layers'])
        
        # Reconstruction R²
        ax = axes[1]
        ax.plot(subset['n_layers'], subset['recon_r2'], 's-', color='#3498db', linewidth=2, markersize=8)
        ax.set_xlabel('Number of GAT Layers')
        ax.set_ylabel('Reconstruction R²')
        ax.set_title('Gene Expression Reconstruction')
        ax.set_xticks(subset['n_layers'])
        
        # Parameter count
        ax = axes[2]
        ax.bar(subset['n_layers'], subset['n_params'] / 1e6, color='#9b59b6', alpha=0.7)
        ax.set_xlabel('Number of GAT Layers')
        ax.set_ylabel('Parameters (M)')
        ax.set_title('Model Size')
        ax.set_xticks(subset['n_layers'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'ablation_gat_layers.png')
        plt.savefig(self.output_dir / 'figures' / 'ablation_gat_layers.pdf')
        plt.close()
        print("  Saved: ablation_gat_layers.png/pdf")
    
    def analyze_ablation_attention_heads(self, df: pd.DataFrame) -> None:
        """Analyze impact of attention heads."""
        subset = df[df['ablation_type'] == 'attention_heads'].copy()
        if len(subset) == 0:
            return
            
        subset['n_heads'] = subset['ablation_setting'].str.extract(r'(\d+)').astype(int)
        subset = subset.sort_values('n_heads')
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(subset))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, subset['lr_auroc'], width, label='L-R AUROC', color='#2ecc71')
        bars2 = ax.bar(x + width/2, subset['recon_r2'], width, label='Recon R²', color='#3498db')
        
        ax.set_xlabel('Number of Attention Heads')
        ax.set_ylabel('Score')
        ax.set_title('Effect of Attention Heads on Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['n_heads'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'ablation_attention_heads.png')
        plt.savefig(self.output_dir / 'figures' / 'ablation_attention_heads.pdf')
        plt.close()
        print("  Saved: ablation_attention_heads.png/pdf")
    
    def analyze_ablation_tasks(self, df: pd.DataFrame) -> None:
        """Analyze multi-task learning contribution."""
        subset = df[df['ablation_type'] == 'tasks'].copy()
        if len(subset) == 0:
            return
            
        # Order tasks meaningfully
        task_order = ['lr_only', 'recon_only', 'classify_only', 'lr_recon', 'lr_classify', 'all_tasks']
        subset['order'] = subset['ablation_setting'].map({t: i for i, t in enumerate(task_order)})
        subset = subset.sort_values('order')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(len(subset))
        width = 0.25
        
        # Handle missing columns gracefully
        metrics = []
        labels = []
        colors = []
        
        if 'lr_auroc' in subset.columns:
            metrics.append(subset['lr_auroc'].fillna(0))
            labels.append('L-R AUROC')
            colors.append('#2ecc71')
        if 'recon_r2' in subset.columns:
            metrics.append(subset['recon_r2'].fillna(0))
            labels.append('Recon R²')
            colors.append('#3498db')
        if 'accuracy' in subset.columns:
            metrics.append(subset['accuracy'].fillna(0))
            labels.append('Accuracy')
            colors.append('#e74c3c')
            
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            offset = (i - len(metrics)/2 + 0.5) * width
            ax.bar(x + offset, metric, width, label=label, color=color, alpha=0.8)
        
        ax.set_xlabel('Task Configuration')
        ax.set_ylabel('Score')
        ax.set_title('Multi-Task Learning Ablation')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in subset['ablation_setting']], rotation=0)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'ablation_multitask.png')
        plt.savefig(self.output_dir / 'figures' / 'ablation_multitask.pdf')
        plt.close()
        print("  Saved: ablation_multitask.png/pdf")
    
    def analyze_ablation_edge_types(self, df: pd.DataFrame) -> None:
        """Analyze impact of edge type awareness."""
        subset = df[df['ablation_type'] == 'edge_types'].copy()
        if len(subset) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(subset))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, subset['lr_auroc'], width, label='L-R AUROC', color='#2ecc71')
        bars2 = ax.bar(x + width/2, subset['recon_r2'], width, label='Recon R²', color='#3498db')
        
        ax.set_xlabel('Edge Type Configuration')
        ax.set_ylabel('Score')
        ax.set_title('Edge-Type Awareness Ablation')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in subset['ablation_setting']])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'ablation_edge_types.png')
        plt.savefig(self.output_dir / 'figures' / 'ablation_edge_types.pdf')
        plt.close()
        print("  Saved: ablation_edge_types.png/pdf")
    
    def create_ablation_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary table for ablation results."""
        summary_rows = []
        
        for ablation_type in df['ablation_type'].unique():
            subset = df[df['ablation_type'] == ablation_type]
            
            # Find baseline (usually the default configuration)
            baseline_names = ['3_layers', '8_heads', 'all_tasks', 'dual_edges', 'dim_256', 'mlp_encoder', 'knn_6', 'residual_decoder']
            baseline = subset[subset['ablation_setting'].isin(baseline_names)]
            if len(baseline) == 0:
                baseline = subset.iloc[[len(subset)//2]]  # Middle row as fallback
            
            # Find best
            if 'lr_auroc' in subset.columns:
                best_idx = subset['lr_auroc'].idxmax()
                best = subset.loc[best_idx]
                
                summary_rows.append({
                    'Ablation Type': ablation_type.replace('_', ' ').title(),
                    'Baseline': baseline.iloc[0]['ablation_setting'],
                    'Best Config': best['ablation_setting'],
                    'Baseline AUROC': baseline.iloc[0].get('lr_auroc', 0),
                    'Best AUROC': best.get('lr_auroc', 0),
                    'Δ AUROC': best.get('lr_auroc', 0) - baseline.iloc[0].get('lr_auroc', 0),
                    'Baseline R²': baseline.iloc[0].get('recon_r2', 0),
                    'Best R²': best.get('recon_r2', 0),
                })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save table
        summary_df.to_csv(self.output_dir / 'tables' / 'ablation_summary.csv', index=False)
        
        # Save LaTeX version
        latex_str = summary_df.to_latex(index=False, float_format='%.3f', escape=False)
        with open(self.output_dir / 'tables' / 'ablation_summary.tex', 'w', encoding='utf-8') as f:
            f.write(latex_str)
            
        print("  Saved: ablation_summary.csv/tex")
        return summary_df
    
    # =========================================================================
    # BENCHMARK ANALYSIS
    # =========================================================================
    
    def analyze_benchmark_comparison(self, df: pd.DataFrame) -> None:
        """Create benchmark comparison visualizations."""
        if len(df) == 0:
            return
            
        # Sort by performance
        df = df.sort_values('lr_auroc', ascending=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        
        # L-R AUROC comparison
        ax = axes[0]
        bars = ax.barh(df['method'], df['lr_auroc'], color=colors)
        ax.set_xlabel('AUROC')
        ax.set_title('L-R Interaction Prediction')
        ax.set_xlim(0, 1)
        
        # Highlight GRAIL-Heart
        for i, (bar, method) in enumerate(zip(bars, df['method'])):
            if 'grail' in method.lower():
                bar.set_color('#e74c3c')
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        # Reconstruction R² comparison
        ax = axes[1]
        if 'recon_r2' in df.columns:
            bars = ax.barh(df['method'], df['recon_r2'].fillna(0), color=colors)
            ax.set_xlabel('R²')
            ax.set_title('Gene Expression Reconstruction')
            ax.set_xlim(0, 1)
            
            for i, (bar, method) in enumerate(zip(bars, df['method'])):
                if 'grail' in method.lower():
                    bar.set_color('#e74c3c')
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)
        
        # Inference time comparison
        ax = axes[2]
        if 'inference_time_mean' in df.columns:
            bars = ax.barh(df['method'], df['inference_time_mean'], color=colors)
            ax.set_xlabel('Time (s)')
            ax.set_title('Inference Time')
            
            for i, (bar, method) in enumerate(zip(bars, df['method'])):
                if 'grail' in method.lower():
                    bar.set_color('#e74c3c')
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'benchmark_comparison.png')
        plt.savefig(self.output_dir / 'figures' / 'benchmark_comparison.pdf')
        plt.close()
        print("  Saved: benchmark_comparison.png/pdf")
    
    def create_benchmark_radar_chart(self, df: pd.DataFrame) -> None:
        """Create radar chart comparing methods across metrics."""
        if len(df) == 0:
            return
            
        metrics = ['lr_auroc', 'lr_auprc', 'recon_r2', 'accuracy']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) < 3:
            return
            
        # Prepare data
        methods = df['method'].tolist()
        values = df[available_metrics].fillna(0).values
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            vals = values[i].tolist()
            vals += vals[:1]
            
            linewidth = 3 if 'grail' in method.lower() else 1.5
            ax.plot(angles, vals, 'o-', linewidth=linewidth, label=method, color=color)
            ax.fill(angles, vals, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n').replace('recon ', 'Recon\n').title() for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Multi-Metric Comparison', y=1.08)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'benchmark_radar.png')
        plt.savefig(self.output_dir / 'figures' / 'benchmark_radar.pdf')
        plt.close()
        print("  Saved: benchmark_radar.png/pdf")
    
    def create_benchmark_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create publication-ready benchmark table."""
        if len(df) == 0:
            return df
            
        # Select and rename columns
        cols = {
            'method': 'Method',
            'lr_auroc': 'L-R AUROC',
            'lr_auprc': 'L-R AUPRC',
            'recon_r2': 'Recon R²',
            'accuracy': 'Accuracy',
            'n_params': 'Parameters',
            'inference_time_mean': 'Time (s)',
        }
        
        available_cols = {k: v for k, v in cols.items() if k in df.columns}
        table_df = df[list(available_cols.keys())].copy()
        table_df.columns = list(available_cols.values())
        
        # Format parameters
        if 'Parameters' in table_df.columns:
            table_df['Parameters'] = table_df['Parameters'].apply(
                lambda x: f"{x/1e6:.2f}M" if x > 0 else "N/A"
            )
        
        # Sort by AUROC
        table_df = table_df.sort_values('L-R AUROC', ascending=False)
        
        # Save
        table_df.to_csv(self.output_dir / 'tables' / 'benchmark_comparison.csv', index=False)
        
        # LaTeX version
        latex_str = table_df.to_latex(index=False, float_format='%.3f', escape=False)
        with open(self.output_dir / 'tables' / 'benchmark_comparison.tex', 'w', encoding='utf-8') as f:
            f.write(latex_str)
            
        print("  Saved: benchmark_comparison.csv/tex")
        return table_df
    
    def load_cv_results(self, cv_dir: str) -> pd.DataFrame:
        """Load cross-validation results from a CV output directory."""
        import yaml
        cv_path = Path(cv_dir)
        
        # Try to load cv_results.yaml or fold_metrics_partial.yaml
        results_file = cv_path / 'cv_results.yaml'
        if not results_file.exists():
            results_file = cv_path / 'fold_metrics_partial.yaml'
        
        if not results_file.exists():
            print(f"No CV results found in {cv_dir}")
            return pd.DataFrame()
        
        with open(results_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle different formats
        if isinstance(data, dict) and 'per_fold' in data:
            fold_metrics = data['per_fold']
        elif isinstance(data, list):
            fold_metrics = data
        else:
            print(f"Unknown CV results format")
            return pd.DataFrame()
        
        df = pd.DataFrame(fold_metrics)
        print(f"Loaded {len(df)} CV fold results")
        return df
    
    def analyze_cv_results(self, df: pd.DataFrame) -> None:
        """Analyze cross-validation results with per-region breakdown."""
        if len(df) == 0:
            return
        
        print("\nGenerating CV analysis figures...")
        
        # 1. Per-region performance box plots
        self._plot_cv_region_performance(df)
        
        # 2. CV summary statistics
        self._create_cv_summary_table(df)
        
        # 3. Region radar chart
        self._create_cv_radar_chart(df)
    
    def _plot_cv_region_performance(self, df: pd.DataFrame) -> None:
        """Create per-region performance comparison."""
        metrics = ['val_auroc', 'val_auprc', 'val_r2', 'val_accuracy', 'val_f1']
        metric_labels = ['L-R AUROC', 'L-R AUPRC', 'Recon R²', 'Cell Type Acc.', 'F1 Score']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
        
        regions = df['region'].tolist()
        colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
        
        for ax, metric, label in zip(axes, metrics, metric_labels):
            if metric in df.columns:
                values = df[metric].values
                bars = ax.bar(regions, values, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_ylabel(label)
                ax.set_xlabel('Cardiac Region')
                ax.set_ylim(0, 1.05)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)
                
                # Add mean line
                mean_val = np.mean(values)
                ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(len(regions)-0.5, mean_val + 0.02, f'μ={mean_val:.2f}', 
                       color='red', fontsize=8, ha='right')
        
        plt.suptitle('GRAIL-Heart Cross-Validation Performance by Cardiac Region', fontsize=14, y=1.02)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            fig.savefig(self.output_dir / 'figures' / f'cv_region_performance.{ext}')
        plt.close()
        print("  Saved: cv_region_performance.png/pdf")
    
    def _create_cv_summary_table(self, df: pd.DataFrame) -> None:
        """Create CV summary statistics table."""
        metrics = ['val_auroc', 'val_auprc', 'val_r2', 'val_pearson_mean', 
                   'val_accuracy', 'val_f1', 'val_precision', 'val_recall']
        metric_names = ['L-R AUROC', 'L-R AUPRC', 'Recon R²', 'Pearson Corr.',
                       'Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        summary_rows = []
        for metric, name in zip(metrics, metric_names):
            if metric in df.columns:
                values = df[metric].dropna()
                summary_rows.append({
                    'Metric': name,
                    'Mean': f'{values.mean():.4f}',
                    'Std': f'{values.std():.4f}',
                    'Min': f'{values.min():.4f}',
                    'Max': f'{values.max():.4f}',
                    'Best Region': df.loc[values.idxmax(), 'region'] if len(values) > 0 else 'N/A'
                })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.output_dir / 'tables' / 'cv_summary.csv', index=False)
        
        latex_str = summary_df.to_latex(index=False, escape=False)
        with open(self.output_dir / 'tables' / 'cv_summary.tex', 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        print("  Saved: cv_summary.csv/tex")
        
        # Also create per-region table
        region_cols = ['region', 'val_auroc', 'val_auprc', 'val_r2', 'val_accuracy']
        if all(c in df.columns for c in region_cols):
            region_df = df[region_cols].copy()
            region_df.columns = ['Region', 'L-R AUROC', 'L-R AUPRC', 'Recon R²', 'Accuracy']
            region_df.to_csv(self.output_dir / 'tables' / 'cv_per_region.csv', index=False)
            print("  Saved: cv_per_region.csv")
    
    def _create_cv_radar_chart(self, df: pd.DataFrame) -> None:
        """Create radar chart comparing regions."""
        metrics = ['val_auroc', 'val_auprc', 'val_r2', 'val_accuracy', 'val_f1']
        metric_labels = ['L-R\nAUROC', 'L-R\nAUPRC', 'Recon\nR²', 'Cell Type\nAcc.', 'F1\nScore']
        
        # Filter to available metrics
        available = [m for m in metrics if m in df.columns]
        labels = [metric_labels[metrics.index(m)] for m in available]
        
        if len(available) < 3:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2*np.pi, len(available), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(df)))
        
        for idx, (_, row) in enumerate(df.iterrows()):
            values = [row[m] for m in available]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['region'], color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Cross-Validation Performance by Cardiac Region', y=1.08)
        
        for ext in ['png', 'pdf']:
            fig.savefig(self.output_dir / 'figures' / f'cv_radar.{ext}', bbox_inches='tight')
        plt.close()
        print("  Saved: cv_radar.png/pdf")
    
    def analyze_ablation(self, df: pd.DataFrame) -> None:
        """Run all ablation analyses."""
        print("\nGenerating ablation analysis figures...")
        
        self.analyze_ablation_gat_layers(df)
        self.analyze_ablation_attention_heads(df)
        self.analyze_ablation_tasks(df)
        self.analyze_ablation_edge_types(df)
        
        print("\nGenerating ablation summary table...")
        self.create_ablation_summary_table(df)
    
    def analyze_benchmarks(self, df: pd.DataFrame) -> None:
        """Run all benchmark analyses."""
        print("\nGenerating benchmark comparison figures...")
        
        self.analyze_benchmark_comparison(df)
        self.create_benchmark_radar_chart(df)
        
        print("\nGenerating benchmark table...")
        self.create_benchmark_table(df)


def main():
    parser = argparse.ArgumentParser(description='Analyze GRAIL-Heart results')
    parser.add_argument('--ablation', type=str, default=None,
                       help='Path pattern for ablation results')
    parser.add_argument('--benchmark', type=str, default=None,
                       help='Path pattern for benchmark results')
    parser.add_argument('--cv_dir', type=str, default=None,
                       help='Path to cross-validation output directory')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all available results')
    parser.add_argument('--output_dir', type=str, default='outputs/analysis',
                       help='Output directory for figures and tables')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    analyzer = ResultsAnalyzer(output_dir)
    
    if args.all:
        args.ablation = 'outputs/ablation/combined_results.csv'
        args.benchmark = 'outputs/benchmark_final/benchmark_results_*.csv'
        # Auto-detect CV directory
        cv_dirs = list(Path('outputs').glob('cv_*'))
        if cv_dirs:
            args.cv_dir = str(sorted(cv_dirs)[-1])  # Use most recent
    
    if args.ablation:
        print(f"\nLoading ablation results: {args.ablation}")
        ablation_df = analyzer.load_ablation_results(args.ablation)
        if len(ablation_df) > 0:
            analyzer.analyze_ablation(ablation_df)
            
    if args.benchmark:
        print(f"\nLoading benchmark results: {args.benchmark}")
        benchmark_df = analyzer.load_benchmark_results(args.benchmark)
        if len(benchmark_df) > 0:
            analyzer.analyze_benchmarks(benchmark_df)
    
    if args.cv_dir:
        print(f"\nLoading CV results: {args.cv_dir}")
        cv_df = analyzer.load_cv_results(args.cv_dir)
        if len(cv_df) > 0:
            analyzer.analyze_cv_results(cv_df)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
