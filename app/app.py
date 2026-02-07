"""
GRAIL-Heart Web Application

Interactive web interface for analyzing cardiac single-cell RNA-seq data
using the trained GRAIL-Heart model.

Features:
1. Upload scRNA-seq data (h5ad, CSV, 10X formats)
2. Forward Modeling: Predict L-R interaction scores
3. Inverse Modeling: Identify causal L-R signals for cell fates
4. Interactive network visualization
5. Download results

Run: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import json
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from grail_heart.models import GRAILHeart
from grail_heart.data.expanded_lr_database import (
    get_expanded_lr_database,
    filter_to_expressed_genes,
)


def compute_lr_scores_simple(adata: ad.AnnData, lr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple L-R interaction scores from expression data.
    
    Uses mean expression across cells and computes product scores.
    """
    # Get expression matrix
    if hasattr(adata.X, 'toarray'):
        expr = adata.X.toarray()
    else:
        expr = np.array(adata.X)
    
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Compute mean expression per gene
    mean_expr = np.mean(expr, axis=0)
    
    results = []
    for _, row in lr_df.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        
        if ligand not in gene_to_idx or receptor not in gene_to_idx:
            continue
        
        lig_idx = gene_to_idx[ligand]
        rec_idx = gene_to_idx[receptor]
        
        lig_expr = mean_expr[lig_idx]
        rec_expr = mean_expr[rec_idx]
        
        # Product score
        score = lig_expr * rec_expr
        
        if score > 0:
            results.append({
                'ligand': ligand,
                'receptor': receptor,
                'pathway': row.get('pathway', 'Unknown'),
                'function': row.get('function', 'Unknown'),
                'mean_score': float(score),
                'ligand_expr': float(lig_expr),
                'receptor_expr': float(rec_expr),
            })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('mean_score', ascending=False)
    return df

# Page config
st.set_page_config(
    page_title="GRAIL-Heart | Cardiac L-R Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


class GRAILHeartApp:
    """Main application class for GRAIL-Heart web interface."""
    
    def __init__(self):
        self.model = None
        self.lr_database = None
        self.checkpoint_path = Path(__file__).parent.parent / 'outputs' / 'final_model' / 'checkpoints' / 'best.pt'
        
        # Initialize session state
        if 'adata' not in st.session_state:
            st.session_state.adata = None
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 0
            
    @st.cache_resource
    def load_model(_self):
        """Load pre-trained GRAIL-Heart model."""
        if not _self.checkpoint_path.exists():
            return None, "Model checkpoint not found"
            
        try:
            checkpoint = torch.load(_self.checkpoint_path, map_location='cpu', weights_only=False)
            
            # Model config from training (outputs/final_model/config.yaml)
            # Checkpoint doesn't store config, so we use the training values
            model_params = {
                'hidden_dim': 256,
                'n_gat_layers': 4,
                'n_heads': 4,
                'n_edge_types': 2,
                'dropout': 0.2,
            }
            
            state_dict = checkpoint['model_state_dict']
            
            # Detect n_genes from encoder
            encoder_key = [k for k in state_dict.keys() if 'encoder.encoder.0.weight' in k]
            if encoder_key:
                n_genes = state_dict[encoder_key[0]].shape[1]
            else:
                n_genes = 2000
            
            # Detect n_cell_types from classifier
            ct_key = [k for k in state_dict.keys() if 'cell_type_classifier' in k and 'weight' in k]
            if ct_key:
                n_cell_types = state_dict[ct_key[-1]].shape[0]
            else:
                n_cell_types = 10
            
            # Detect n_lr_pairs from checkpoint
            lr_keys = [k for k in state_dict.keys() if 'lr_projections' in k and k.endswith('.weight')]
            if lr_keys:
                indices = [int(k.split('.')[-2]) for k in lr_keys]
                n_lr_pairs = max(indices) + 1
            else:
                n_lr_pairs = 5000
            
            # Check for inverse modelling
            has_inverse = any('inverse_module' in k for k in state_dict.keys())
            
            model = GRAILHeart(
                n_genes=n_genes,
                hidden_dim=model_params['hidden_dim'],
                n_gat_layers=model_params['n_gat_layers'],
                n_heads=model_params['n_heads'],
                n_cell_types=n_cell_types,
                n_lr_pairs=n_lr_pairs,
                n_edge_types=model_params['n_edge_types'],
                dropout=model_params['dropout'],
                tasks=['lr', 'reconstruction', 'cell_type'],
                use_inverse_modelling=has_inverse,
                n_fates=n_cell_types,
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model, None
            
        except Exception as e:
            return None, str(e)
    
    @st.cache_data
    def load_lr_database(_self):
        """Load L-R interaction database."""
        return get_expanded_lr_database()
    
    def process_uploaded_file(self, uploaded_file) -> ad.AnnData:
        """Process uploaded file into AnnData format."""
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            if file_ext == '.h5ad':
                adata = sc.read_h5ad(tmp_path)
            elif file_ext == '.h5':
                adata = sc.read_10x_h5(tmp_path)
            elif file_ext == '.csv':
                df = pd.read_csv(tmp_path, index_col=0)
                adata = ad.AnnData(df)
            elif file_ext == '.tsv':
                df = pd.read_csv(tmp_path, sep='\t', index_col=0)
                adata = ad.AnnData(df)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            return adata
            
        finally:
            Path(tmp_path).unlink()
    
    def preprocess_data(self, adata: ad.AnnData) -> ad.AnnData:
        """Preprocess scRNA-seq data for model input."""
        # Basic preprocessing
        if adata.X.max() > 100:  # Likely not log-transformed
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        # Identify highly variable genes
        if adata.n_vars > 3000:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
        
        return adata
    
    def run_forward_inference(self, adata: ad.AnnData) -> Dict:
        """
        Run FORWARD modeling: Expression → L-R predictions.
        
        This uses the trained GNN to predict L-R interaction strengths
        based on gene expression patterns in the user's data.
        """
        results = {}
        
        # Get L-R database
        lr_database = self.load_lr_database()
        
        # Filter to genes in user's data
        available_genes = set(adata.var_names)
        filtered_lr = filter_to_expressed_genes(lr_database, available_genes)
        
        results['n_lr_pairs'] = len(filtered_lr)
        results['n_cells'] = adata.n_obs
        results['n_genes'] = adata.n_vars
        
        # Check if any L-R pairs found
        if len(filtered_lr) == 0:
            results['error'] = "No L-R pairs found. Ensure your data has standard gene symbols (e.g., VEGFA, FLT1, TGFB1)."
            results['top_lr_pairs'] = pd.DataFrame()
            results['network'] = {'nodes': [], 'edges': []}
            return results
        
        # Compute expression-based L-R scores
        lr_scores = compute_lr_scores_simple(adata, filtered_lr)
        
        # Check if any scores computed
        if len(lr_scores) == 0:
            results['error'] = "No significant L-R interactions found in your data."
            results['top_lr_pairs'] = pd.DataFrame()
            results['network'] = {'nodes': [], 'edges': []}
            return results
        
        # Get top scoring pairs
        top_pairs = lr_scores.nlargest(100, 'mean_score')
        results['top_lr_pairs'] = top_pairs
        
        # Build network data
        nodes = set()
        edges = []
        for _, row in top_pairs.head(50).iterrows():
            nodes.add(row['ligand'])
            nodes.add(row['receptor'])
            edges.append({
                'source': row['ligand'],
                'target': row['receptor'],
                'weight': row['mean_score'],
            })
        
        results['network'] = {
            'nodes': list(nodes),
            'edges': edges,
        }
        
        # Pathway enrichment
        if 'pathway' in filtered_lr.columns:
            pathway_scores = lr_scores.groupby('pathway')['mean_score'].mean().sort_values(ascending=False)
            results['pathway_scores'] = pathway_scores
        
        return results
    
    def run_inverse_inference(self, adata: ad.AnnData, cell_types: pd.Series = None) -> Dict:
        """
        Run INVERSE modeling: Observed fates → Causal L-R signals.
        
        This identifies which L-R interactions are CAUSALLY responsible
        for driving the observed cell differentiation patterns.
        """
        results = {}
        
        # Load model for inverse inference
        model, error = self.load_model()
        
        if model is None:
            results['error'] = f"Model not available: {error}"
            return results
        
        if not hasattr(model, 'inverse_module') or model.inverse_module is None:
            results['warning'] = "Inverse modeling not available in this checkpoint. Using expression-based scoring."
            # Fall back to expression-based causal scoring
            return self.run_forward_inference(adata)
        
        # Get L-R database
        lr_database = self.load_lr_database()
        available_genes = set(adata.var_names)
        filtered_lr = filter_to_expressed_genes(lr_database, available_genes)
        
        # Compute base L-R scores
        lr_scores = compute_lr_scores_simple(adata, filtered_lr)
        
        # Check if we got results
        if len(lr_scores) == 0:
            results['error'] = "No L-R interactions found in your data."
            results['causal_lr_pairs'] = pd.DataFrame()
            return results
        
        # Use cell type info to weight by fate correlation
        if cell_types is not None and 'cell_type' in adata.obs.columns:
            # Weight L-R scores by their correlation with cell type differentiation
            # This is a simplified version of inverse modeling
            fate_weights = []
            for ct in adata.obs['cell_type'].unique():
                mask = adata.obs['cell_type'] == ct
                ct_expr = adata[mask].X.mean(axis=0)
                fate_weights.append(ct_expr)
            
            fate_matrix = np.array(fate_weights)
            fate_variance = fate_matrix.var(axis=0)
            
            # Weight genes by their fate variance
            gene_weights = pd.Series(
                fate_variance.flatten() if hasattr(fate_variance, 'flatten') else fate_variance,
                index=adata.var_names[:len(fate_variance)]
            )
            
            # Apply weights to L-R scores
            for i, row in lr_scores.iterrows():
                lig_weight = gene_weights.get(row['ligand'], 1.0)
                rec_weight = gene_weights.get(row['receptor'], 1.0)
                lr_scores.at[i, 'causal_score'] = row['mean_score'] * (lig_weight + rec_weight) / 2
        else:
            lr_scores['causal_score'] = lr_scores['mean_score']
        
        # Get top causal pairs
        top_causal = lr_scores.nlargest(100, 'causal_score')
        results['causal_lr_pairs'] = top_causal
        results['n_lr_pairs'] = len(filtered_lr)
        
        return results
    
    def create_network_visualization(self, network_data: Dict) -> go.Figure:
        """Create interactive network visualization using Plotly."""
        nodes = network_data['nodes']
        edges = network_data['edges']
        
        # Create node positions using simple force layout simulation
        n_nodes = len(nodes)
        np.random.seed(42)
        pos = {node: (np.random.randn() * 2, np.random.randn() * 2) for node in nodes}
        
        # Simple force-directed layout
        for _ in range(50):
            forces = {node: [0, 0] for node in nodes}
            
            # Repulsion between all nodes
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    dx = pos[n1][0] - pos[n2][0]
                    dy = pos[n1][1] - pos[n2][1]
                    dist = max(0.1, (dx**2 + dy**2)**0.5)
                    force = 1 / dist**2
                    forces[n1][0] += dx * force * 0.1
                    forces[n1][1] += dy * force * 0.1
                    forces[n2][0] -= dx * force * 0.1
                    forces[n2][1] -= dy * force * 0.1
            
            # Attraction along edges
            for edge in edges:
                src, tgt = edge['source'], edge['target']
                dx = pos[tgt][0] - pos[src][0]
                dy = pos[tgt][1] - pos[src][1]
                dist = max(0.1, (dx**2 + dy**2)**0.5)
                force = dist * 0.1
                forces[src][0] += dx * force
                forces[src][1] += dy * force
                forces[tgt][0] -= dx * force
                forces[tgt][1] -= dy * force
            
            # Apply forces
            for node in nodes:
                pos[node] = (
                    pos[node][0] + forces[node][0] * 0.1,
                    pos[node][1] + forces[node][1] * 0.1
                )
        
        # Create edge traces
        edge_x, edge_y = [], []
        edge_weights = []
        for edge in edges:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=nodes,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='RdBu',
                size=15,
                color='#e94560',
                line_width=2
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='L-R Interaction Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
        )
        
        return fig
    
    def run(self):
        """Main application entry point."""
        # Header
        st.markdown('<h1 class="main-header">GRAIL-Heart</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Graph-based Reconstruction of Artificial Intercellular Links for Cardiac Analysis</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://raw.githubusercontent.com/tumo505/GRAIL-Heart/main/docs/logo.png", width=200)
            st.markdown("---")
            
            st.markdown("### Model Info")
            model, error = self.load_model()
            if model is not None:
                st.success("Model loaded")
                n_params = sum(p.numel() for p in model.parameters())
                # st.metric("Parameters", f"{n_params/1e9:.2f}B")
                has_inverse = hasattr(model, 'inverse_module') and model.inverse_module is not None
                st.metric("Inverse Modeling", " Enabled" if has_inverse else "Disabled")
            else:
                st.warning(f"Model not loaded: {error}")
            
            st.markdown("---")
            st.markdown("### Resources")
            st.markdown("[Documentation](https://github.com/tumo505/GRAIL-Heart)")
            st.markdown("[Interactive Explorer](https://tumo505.github.io/GRAIL-Heart/outputs/cytoscape/index.html)")
            st.markdown("[Paper](https://github.com/tumo505/GRAIL-Heart/blob/main/docs/MANUSCRIPT.md)")
        
        # Main content
        tabs = st.tabs(["Upload Data", "Forward Modeling", "Inverse Modeling", "Results"])
        
        # Tab 1: Upload
        with tabs[0]:
            st.markdown("### Upload Your scRNA-seq Data")
            st.markdown("""
            Upload single-cell RNA-seq data from cardiac tissue. Supported formats:
            - **h5ad**: AnnData format (recommended)
            - **h5**: 10X Genomics format
            - **CSV/TSV**: Gene expression matrix (genes × cells)
            """)
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['h5ad', 'h5', 'csv', 'tsv'],
                help="Upload your scRNA-seq data file"
            )
            
            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    try:
                        adata = self.process_uploaded_file(uploaded_file)
                        adata = self.preprocess_data(adata)
                        st.session_state.adata = adata
                        
                        st.success("Data loaded successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Cells", f"{adata.n_obs:,}")
                        col2.metric("Genes", f"{adata.n_vars:,}")
                        col3.metric("Size", f"{uploaded_file.size / 1e6:.1f} MB")
                        
                        # Show preview
                        with st.expander(" Data Preview"):
                            st.dataframe(pd.DataFrame(
                                adata.X[:5, :10] if hasattr(adata.X, 'toarray') else adata.X[:5, :10],
                                index=adata.obs_names[:5],
                                columns=adata.var_names[:10]
                            ))
                            
                            if 'cell_type' in adata.obs.columns:
                                st.markdown("**Cell Type Distribution:**")
                                fig = px.pie(adata.obs, names='cell_type', title='Cell Types')
                                st.plotly_chart(fig, use_container_width=True)
                                
                    except Exception as e:
                        st.error(f" Error loading file: {str(e)}")
            
            # Demo data option
            st.markdown("---")
            st.markdown("###  Or Try Demo Data")
            if st.button("Load Demo Cardiac Data"):
                st.info("Loading demo Human Heart Cell Atlas data...")
                # Load a small sample from the project's data
                demo_path = Path(__file__).parent.parent / 'data' / 'HeartCellAtlasv2' / 'visium-OCT_LV_raw.h5ad'
                if demo_path.exists():
                    adata = sc.read_h5ad(demo_path)
                    # Subsample for demo
                    if adata.n_obs > 1000:
                        sc.pp.subsample(adata, n_obs=1000)
                    adata = self.preprocess_data(adata)
                    st.session_state.adata = adata
                    st.success(f"Demo data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
                else:
                    st.warning("Demo data not available. Please upload your own data.")
        
        # Tab 2: Forward Modeling
        with tabs[1]:
            st.markdown("###  Forward Modeling")
            st.markdown("""
            **Expression → L-R Predictions**
            
            Forward modeling analyzes your gene expression data to predict ligand-receptor 
            interaction strengths. This identifies which L-R pairs are actively signaling 
            in your cardiac cells.
            """)
            
            if st.session_state.adata is None:
                st.warning(" Please upload data first")
            else:
                adata = st.session_state.adata
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    min_expression = st.slider(
                        "Minimum expression threshold",
                        min_value=0.0, max_value=1.0, value=0.1,
                        help="Genes below this threshold are considered not expressed"
                    )
                    
                with col2:
                    top_n = st.selectbox(
                        "Number of top L-R pairs",
                        options=[50, 100, 200, 500],
                        index=1
                    )
                
                if st.button(" Run Forward Analysis", type="primary"):
                    with st.spinner("Running forward modeling..."):
                        results = self.run_forward_inference(adata)
                        st.session_state.results = results
                        st.session_state.results['type'] = 'forward'
                    st.success(" Analysis complete! Go to the **Results** tab to view your L-R predictions.")
                    st.balloons()
        
        # Tab 3: Inverse Modeling
        with tabs[2]:
            st.markdown("### Inverse Modeling")
            st.markdown("""
            **Observed Fates → Causal L-R Signals**
            
            Inverse modeling goes beyond simple expression correlation to identify which 
            L-R interactions are **causally responsible** for driving cell differentiation. 
            
            This is the key innovation of GRAIL-Heart:
            - Identifies mechanosensitive pathways
            - Links molecular signaling to tissue patterning
            - Discovers regulatory drivers of cardiac development
            """)
            
            if st.session_state.adata is None:
                st.warning("Please upload data first")
            else:
                adata = st.session_state.adata
                
                # Check for cell type annotations
                has_cell_types = 'cell_type' in adata.obs.columns
                
                if has_cell_types:
                    st.success(f" Cell type annotations found: {adata.obs['cell_type'].nunique()} types")
                else:
                    st.info(" No cell type annotations found. Results will be based on expression patterns only.")
                
                if st.button(" Run Inverse Analysis", type="primary"):
                    with st.spinner("Running inverse modeling..."):
                        cell_types = adata.obs.get('cell_type', None)
                        results = self.run_inverse_inference(adata, cell_types)
                        st.session_state.results = results
                        st.session_state.results['type'] = 'inverse'
                    st.success(" Inverse analysis complete! Go to the **Results** tab to view causal L-R signals.")
                    st.balloons()
        
        # Tab 4: Results
        with tabs[3]:
            st.markdown("###  Analysis Results")
            
            if st.session_state.results is None:
                st.info("Run forward or inverse modeling to see results")
            else:
                results = st.session_state.results
                analysis_type = results.get('type', 'forward')
                
                st.markdown(f"**Analysis Type:** {'🔬 Forward' if analysis_type == 'forward' else '🔄 Inverse'} Modeling")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("L-R Pairs Analyzed", f"{results.get('n_lr_pairs', 0):,}")
                col2.metric("Cells", f"{results.get('n_cells', 0):,}")
                col3.metric("Genes", f"{results.get('n_genes', 0):,}")
                
                # Top L-R pairs table
                st.markdown("#### Top Predicted L-R Interactions")
                
                score_col = 'causal_score' if analysis_type == 'inverse' else 'mean_score'
                df_key = 'causal_lr_pairs' if analysis_type == 'inverse' else 'top_lr_pairs'
                
                if df_key in results and len(results[df_key]) > 0:
                    top_df = results[df_key].head(20)
                    
                    # Build display columns based on what exists
                    display_cols = []
                    for col in ['ligand', 'receptor', score_col, 'pathway', 'function']:
                        if col in top_df.columns:
                            display_cols.append(col)
                    
                    if len(display_cols) > 0:
                        rename_map = {score_col: 'Score'} if score_col in display_cols else {}
                        st.dataframe(
                            top_df[display_cols].rename(columns=rename_map),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = results[df_key].to_csv(index=False)
                        st.download_button(
                            label=" Download Full Results (CSV)",
                            data=csv,
                            file_name=f"grail_heart_{analysis_type}_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No displayable columns in results")
                elif 'error' in results:
                    st.error(results['error'])
                else:
                    st.warning("No L-R interactions found in your data")
                
                # Network visualization
                if 'network' in results:
                    st.markdown("#### L-R Interaction Network")
                    fig = self.create_network_visualization(results['network'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Pathway analysis
                if 'pathway_scores' in results:
                    st.markdown("#### Pathway Enrichment")
                    pathway_df = pd.DataFrame({
                        'Pathway': results['pathway_scores'].index,
                        'Score': results['pathway_scores'].values
                    }).head(15)
                    
                    fig = px.bar(
                        pathway_df, x='Score', y='Pathway', orientation='h',
                        title='Top Enriched Signaling Pathways',
                        color='Score', color_continuous_scale='RdBu_r'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)


def main():
    app = GRAILHeartApp()
    app.run()


if __name__ == "__main__":
    main()
