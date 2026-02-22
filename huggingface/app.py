"""
GRAIL-Heart — Hugging Face Spaces Edition

Interactive web interface for analysing cardiac cell-cell communication
using the trained GRAIL-Heart GNN model with inverse modelling.

Adapted from the full app for deployment on HF Spaces with:
- Relative paths (no PROJECT_ROOT dependency)
- Bundled demo data (500-cell LV subsample)
- CPU-only inference
- Stripped checkpoint (~50 MB, no optimizer state)

Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import sys
import warnings
from typing import Dict, List, Optional, Any

warnings.filterwarnings("ignore")

# ── Resolve paths (HF Spaces: app.py is at repo root) ─────────────
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

from grail_heart.models import GRAILHeart
from grail_heart.data.cellchat_database import (
    get_omnipath_lr_database,
    get_mechanosensitive_gene_sets,
    build_pathway_gene_mask,
    annotate_cardiac_pathways,
    CARDIAC_PATHWAYS,
)

# ── Paths for bundled data ─────────────────────────────────────────
CHECKPOINT_PATH = APP_DIR / "model" / "best.pt"
PRECOMPUTED_DIR = APP_DIR / "demo_data" / "tables"
DEMO_H5AD = APP_DIR / "demo_data" / "demo_lv_500cells.h5ad"
LR_CACHE = APP_DIR / "demo_data" / "lr_database_cache.csv"

REGIONS = ["AX", "LA", "LV", "RA", "RV", "SP"]
REGION_NAMES = {
    "AX": "Apex",
    "LA": "Left Atrium",
    "LV": "Left Ventricle",
    "RA": "Right Atrium",
    "RV": "Right Ventricle",
    "SP": "Septum",
}

# ── Bioicons (CC-0 / CC-BY scientific icons from bioicons.com) ─────
_BIO = "https://bioicons.com/icons"
BIOICONS = {
    "heart": f"{_BIO}/cc-by-3.0/Human_physiology/Servier/heart.svg",
    "umap": f"{_BIO}/cc-0/Scientific_graphs/James-Lloyd/SingleCell_Clustering_DataReduction_UMAP.svg",
    "dna": f"{_BIO}/cc-0/Nucleic_acids/Kumar/DNA.svg",
    "microscope": f"{_BIO}/cc-by-3.0/Lab_apparatus/Servier/microscope.svg",
    "receptor": f"{_BIO}/cc-by-3.0/Receptors_channels/Servier/receptor-membrane-ligand.svg",
    "heatmap": f"{_BIO}/cc-by-4.0/Scientific_graphs/ChenxinLi/heatmap.svg",
    "cardiomyocyte": f"{_BIO}/cc-0/Cell_types/Marnie-Maddock/simple_cardiomyocyte.svg",
    "network": f"{_BIO}/cc-by-4.0/Chemo-_and_Bioinformatics/ChenxinLi/Network.svg",
}


def bioicon(name: str, size: int = 28) -> str:
    """Return an inline HTML <img> tag for a bioicon from bioicons.com."""
    url = BIOICONS.get(name, "")
    if url:
        return (
            f'<img src="{url}" width="{size}" height="{size}" '
            f'style="vertical-align:middle; margin-right:6px;">'
        )
    return ""


# ═══════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading GRAIL-Heart model …")
def load_model(ckpt_path: str = str(CHECKPOINT_PATH)):
    """Load pre-trained model, inferring architecture from state dict."""
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        return None, f"Checkpoint not found: {ckpt}"

    try:
        checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]

        # ── Infer dims from state_dict ─────────────────────────────
        enc_key = [k for k in state if "gene_encoder.encoder.0.weight" in k]
        n_genes = state[enc_key[0]].shape[1] if enc_key else 2000

        ct_keys = [k for k in state if "cell_type_predictor" in k and k.endswith(".weight")]
        n_cell_types = state[ct_keys[-1]].shape[0] if ct_keys else 10

        gat_layer_ids = [
            int(k.split(".")[2])
            for k in state
            if k.startswith("gat.layers.") and k.split(".")[2].isdigit()
        ]
        gat_layers = max(gat_layer_ids) + 1 if gat_layer_ids else 4

        head_key = [k for k in state if "gat.layers.0.gat.att_src" in k]
        n_heads = state[head_key[0]].shape[1] if head_key else 4

        hid_key = [k for k in state if "gat.layers.0.gat.lin_src.weight" in k]
        hidden_dim = state[hid_key[0]].shape[0] if hid_key else 256

        has_inverse = any("inverse_module" in k for k in state)

        pw_key = [k for k in state if "lr_target_decoder.pathway_gene_matrix" in k]
        n_pathways = state[pw_key[0]].shape[0] if pw_key else 0
        mech_key = [k for k in state if "mechano_module.pathway_gene_mask" in k]
        n_mechano = state[mech_key[0]].shape[0] if mech_key else 0

        kwargs: Dict[str, Any] = {}
        if n_pathways > 0:
            kwargs["pathway_gene_mask"] = torch.zeros(n_pathways, n_genes)
        if n_mechano > 0:
            kwargs["mechano_gene_mask"] = torch.zeros(n_mechano, n_genes)
            kwargs["mechano_pathway_names"] = [f"mechano_{i}" for i in range(n_mechano)]

        model = GRAILHeart(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            n_gat_layers=gat_layers,
            n_heads=n_heads,
            n_cell_types=n_cell_types,
            n_lr_pairs=None,
            n_edge_types=2,
            dropout=0.2,
            tasks=["lr", "reconstruction", "cell_type"],
            use_inverse_modelling=has_inverse,
            n_fates=n_cell_types,
            n_pathways=n_pathways if n_pathways else None,
            n_mechano_pathways=n_mechano if n_mechano else None,
            **kwargs,
        )
        model.load_state_dict(state, strict=False)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        return model, {
            "n_params": n_params,
            "n_genes": n_genes,
            "hidden_dim": hidden_dim,
            "n_gat_layers": gat_layers,
            "n_heads": n_heads,
            "n_cell_types": n_cell_types,
            "has_inverse": has_inverse,
            "n_pathways": n_pathways,
            "n_mechano": n_mechano,
        }
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════
#  L-R database (use bundled cache if available)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading OmniPath L-R database …")
def load_lr_database() -> pd.DataFrame:
    """Load and annotate the OmniPath L-R database.

    Tries to load from a bundled CSV cache first (avoids network call on HF).
    Falls back to live OmniPath query.
    """
    if LR_CACHE.exists():
        lr = pd.read_csv(LR_CACHE)
        if "ligand" in lr.columns and "receptor" in lr.columns:
            return lr
    # Fallback: live query
    lr = get_omnipath_lr_database()
    lr = annotate_cardiac_pathways(lr)
    return lr


# ═══════════════════════════════════════════════════════════════════
#  Pre-computed results
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_precomputed_scores(region: str) -> Optional[pd.DataFrame]:
    """Load pre-computed L-R scores for a Heart Cell Atlas region."""
    path = PRECOMPUTED_DIR / f"{region}_lr_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_cross_region_comparison() -> Optional[pd.DataFrame]:
    path = PRECOMPUTED_DIR / "cross_region_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════
#  Upload helpers
# ═══════════════════════════════════════════════════════════════════

def process_uploaded_file(uploaded_file) -> ad.AnnData:
    """Read an uploaded file into an AnnData object."""
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if ext == ".h5ad":
            adata = sc.read_h5ad(tmp_path)
        elif ext == ".h5":
            adata = sc.read_10x_h5(tmp_path)
        elif ext == ".csv":
            df = pd.read_csv(tmp_path, index_col=0)
            adata = ad.AnnData(df)
        elif ext == ".tsv":
            df = pd.read_csv(tmp_path, sep="\t", index_col=0)
            adata = ad.AnnData(df)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        return adata
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _resolve_gene_symbols(adata: ad.AnnData) -> ad.AnnData:
    """Convert var_names from Ensembl IDs to HGNC symbols when possible."""
    sample = list(adata.var_names[:200])
    ens_frac = sum(1 for g in sample if str(g).startswith("ENS")) / max(len(sample), 1)

    if ens_frac < 0.5:
        return adata

    symbol_col = None
    for candidate in ["SYMBOL", "gene_symbol", "gene_symbols", "gene_name",
                       "gene_short_name", "symbol", "name", "feature_name"]:
        if candidate in adata.var.columns:
            symbol_col = candidate
            break

    if symbol_col is None:
        return adata

    symbols = adata.var[symbol_col].astype(str).values
    valid = np.array([
        s != "" and s != "nan" and s != "None" and not s.startswith("ENS")
        for s in symbols
    ])

    if valid.sum() < 500:
        return adata

    adata = adata[:, valid].copy()
    adata.var_names = adata.var[symbol_col].values.astype(str)
    adata.var_names_make_unique()
    return adata


def preprocess_adata(adata: ad.AnnData, n_top_genes: int = 2000) -> ad.AnnData:
    """Standard preprocessing: filter -> normalize -> log1p -> HVG -> scale."""
    adata = adata.copy()
    adata = _resolve_gene_symbols(adata)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)

    if hasattr(adata.X, 'max'):
        xmax = adata.X.max()
        if hasattr(xmax, 'item'):
            xmax = xmax.item()
    else:
        xmax = np.max(adata.X)
    if xmax > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    adata.layers["log_norm"] = adata.X.copy()

    if adata.n_vars > n_top_genes + 500:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)

    sc.pp.scale(adata, max_value=10)
    return adata


# ═══════════════════════════════════════════════════════════════════
#  Inference on custom data
# ═══════════════════════════════════════════════════════════════════

def build_spatial_graph(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Build symmetric kNN graph. Returns [2, n_edges]."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(coords)
    _, idx = nn.kneighbors(coords)

    edges_set = set()
    src_list, dst_list = [], []
    for i in range(len(coords)):
        for j in idx[i]:
            if i != j and (i, j) not in edges_set:
                edges_set.add((i, j))
                edges_set.add((j, i))
                src_list.extend([i, j])
                dst_list.extend([j, i])

    return np.array([src_list, dst_list], dtype=np.int64)


def compute_expression_scores(
    adata: ad.AnnData,
    lr_db: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean expression product scores for each L-R pair."""
    if "log_norm" in adata.layers:
        X = adata.layers["log_norm"]
    else:
        X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    gene_idx = {g: i for i, g in enumerate(adata.var_names)}
    mean_expr = X.mean(axis=0)

    rows = []
    for _, row in lr_db.iterrows():
        lig, rec = row["ligand"], row["receptor"]
        if lig not in gene_idx or rec not in gene_idx:
            continue
        li, ri = gene_idx[lig], gene_idx[rec]
        score = float(mean_expr[li] * mean_expr[ri])
        if score > 0:
            rows.append({
                "ligand": lig,
                "receptor": rec,
                "pathway": row.get("pathway", ""),
                "function": row.get("function", ""),
                "mean_score": score,
                "ligand_expr": float(mean_expr[li]),
                "receptor_expr": float(mean_expr[ri]),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df.sort_values("mean_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def run_model_inference(
    model,
    adata: ad.AnnData,
    edge_index: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run the GRAIL-Heart GNN and return per-edge scores."""
    from torch_geometric.data import Data

    device = next(model.parameters()).device

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    x = torch.tensor(np.array(X, dtype=np.float32), device=device)

    coords = _get_spatial_coords(adata)
    cmin, cmax = coords.min(0), coords.max(0)
    rng = np.maximum(cmax - cmin, 1e-6)
    coords_norm = (coords - cmin) / rng

    ei = torch.tensor(edge_index, dtype=torch.long, device=device)
    pos = torch.tensor(coords_norm, dtype=torch.float32, device=device)
    et = torch.zeros(ei.shape[1], dtype=torch.long, device=device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    if "cell_type" in adata.obs.columns:
        ct_cats = adata.obs["cell_type"].astype("category")
        y = torch.tensor(ct_cats.cat.codes.values, dtype=torch.long, device=device)
    else:
        y = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    data = Data(x=x, edge_index=ei, pos=pos, edge_type=et, batch=batch, y=y)

    with torch.no_grad():
        outputs = model(data)

    results: Dict[str, np.ndarray] = {}

    if "lr_scores" in outputs and outputs["lr_scores"] is not None:
        scores = outputs["lr_scores"]
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        results["forward_lr_scores"] = torch.sigmoid(scores).cpu().numpy()

    if "causal_scores" in outputs and outputs["causal_scores"] is not None:
        cs = outputs["causal_scores"]
        if cs.dim() > 1:
            cs = cs.mean(-1)
        results["causal_scores"] = cs.cpu().numpy()

    if "fate_logits" in outputs and outputs["fate_logits"] is not None:
        results["fate_probs"] = (
            torch.softmax(outputs["fate_logits"], dim=-1).cpu().numpy()
        )

    if "pathway_activation" in outputs and outputs["pathway_activation"] is not None:
        results["pathway_activation"] = outputs["pathway_activation"].cpu().numpy()
    if (
        "mechano_pathway_activation" in outputs
        and outputs["mechano_pathway_activation"] is not None
    ):
        results["mechano_activation"] = (
            outputs["mechano_pathway_activation"].cpu().numpy()
        )

    return results


def _get_spatial_coords(adata: ad.AnnData) -> np.ndarray:
    """Extract spatial coordinates from AnnData."""
    for key in ["spatial", "X_spatial"]:
        if key in adata.obsm:
            return np.array(adata.obsm[key], dtype=np.float32)
    return np.zeros((adata.n_obs, 2), dtype=np.float32)


def combine_scores(
    expr_scores: pd.DataFrame,
    model_out: Dict[str, np.ndarray],
    adata: ad.AnnData,
    edge_index: np.ndarray,
) -> pd.DataFrame:
    """Map per-edge causal scores -> per-L-R-pair causal scores.

    Uses expression-gated 90th-percentile aggregation.
    """
    df = expr_scores.copy()
    causal = model_out.get("causal_scores")
    if causal is None or len(df) == 0:
        df["causal_score"] = df["mean_score"]
        return df

    if "log_norm" in adata.layers:
        X = adata.layers["log_norm"]
    else:
        X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)
    gene_idx = {g: i for i, g in enumerate(adata.var_names)}

    src = edge_index[0]
    dst = edge_index[1]
    THRESHOLD = 0.5

    pair_causal = []
    for _, row in df.iterrows():
        lig, rec = row["ligand"], row["receptor"]
        li = gene_idx.get(lig)
        ri = gene_idx.get(rec)
        if li is None or ri is None:
            pair_causal.append(0.0)
            continue

        lig_vals = X[src, li]
        rec_vals = X[dst, ri]
        mask = (lig_vals > THRESHOLD) & (rec_vals > THRESHOLD)

        if mask.sum() < 5:
            pair_causal.append(0.0)
            continue

        active_causal = causal[mask]
        p90 = float(np.percentile(active_causal, 90))
        expr_weight = float(np.sqrt(lig_vals[mask].mean() * rec_vals[mask].mean()))
        pair_causal.append(p90 * expr_weight)

    df["causal_score"] = pair_causal
    mx = df["causal_score"].max()
    if mx > 0:
        df["causal_score"] /= mx

    df.sort_values("causal_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════

def make_lr_bar_chart(df: pd.DataFrame, score_col: str, title: str, n: int = 20):
    """Horizontal bar chart of top L-R pairs."""
    top = df.nlargest(n, score_col).copy()
    top["pair"] = top["ligand"] + " → " + top["receptor"]
    fig = px.bar(
        top, x=score_col, y="pair", orientation="h",
        color=score_col, color_continuous_scale="Reds",
        title=title,
        labels={score_col: "Score", "pair": "L-R Pair"},
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        height=max(400, n * 24),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_pathway_chart(df: pd.DataFrame, score_col: str = "mean_score"):
    """Bar chart grouped by pathway."""
    if "pathway" not in df.columns or len(df) == 0:
        return None
    pw = df.groupby("pathway")[score_col].mean().sort_values(ascending=False).head(15)
    if len(pw) == 0:
        return None
    fig = px.bar(
        x=pw.values, y=pw.index, orientation="h",
        color=pw.values, color_continuous_scale="Viridis",
        title="Mean Score by Pathway",
        labels={"x": score_col, "y": "Pathway"},
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_network_graph(df: pd.DataFrame, score_col: str, n: int = 40):
    """Force-directed network of top L-R pairs."""
    top = df.nlargest(n, score_col)
    nodes = list(set(top["ligand"]).union(set(top["receptor"])))
    if len(nodes) == 0:
        return go.Figure()

    np.random.seed(42)
    pos = {nd: [np.random.randn(), np.random.randn()] for nd in nodes}

    for _ in range(80):
        forces = {nd: [0.0, 0.0] for nd in nodes}
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                dx = pos[n1][0] - pos[n2][0]
                dy = pos[n1][1] - pos[n2][1]
                d = max(0.05, (dx**2 + dy**2) ** 0.5)
                f = 1.0 / d**2
                forces[n1][0] += dx * f * 0.05
                forces[n1][1] += dy * f * 0.05
                forces[n2][0] -= dx * f * 0.05
                forces[n2][1] -= dy * f * 0.05
        for _, row in top.iterrows():
            s, t = row["ligand"], row["receptor"]
            dx = pos[t][0] - pos[s][0]
            dy = pos[t][1] - pos[s][1]
            d = max(0.05, (dx**2 + dy**2) ** 0.5)
            f = d * 0.08
            forces[s][0] += dx * f
            forces[s][1] += dy * f
            forces[t][0] -= dx * f
            forces[t][1] -= dy * f
        for nd in nodes:
            pos[nd][0] += forces[nd][0] * 0.08
            pos[nd][1] += forces[nd][1] * 0.08

    ex, ey = [], []
    for _, row in top.iterrows():
        s, t = row["ligand"], row["receptor"]
        ex.extend([pos[s][0], pos[t][0], None])
        ey.extend([pos[s][1], pos[t][1], None])

    edge_trace = go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.8, color="#bbb"), hoverinfo="none",
    )

    ligands = set(top["ligand"])
    receptors = set(top["receptor"])
    colors = []
    for nd in nodes:
        if nd in ligands and nd in receptors:
            colors.append("#9b59b6")
        elif nd in ligands:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    node_trace = go.Scatter(
        x=[pos[nd][0] for nd in nodes],
        y=[pos[nd][1] for nd in nodes],
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=12, color=colors, line=dict(width=1, color="white")),
        hoverinfo="text",
    )

    fig = go.Figure(
        [edge_trace, node_trace],
        layout=go.Layout(
            title="L-R Interaction Network",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=35, b=0),
            height=550,
        ),
    )
    return fig


def make_cross_region_heatmap(df: pd.DataFrame, n: int = 25):
    """Heatmap of top L-R pairs across regions."""
    top = df.nlargest(n, "mean_score")
    top["pair"] = top["ligand"] + " → " + top["receptor"]
    region_cols = [c for c in REGIONS if c in top.columns]
    matrix = top.set_index("pair")[region_cols]
    fig = px.imshow(
        matrix,
        color_continuous_scale="RdBu_r",
        title="Top L-R Interactions Across Cardiac Regions",
        labels=dict(x="Region", y="L-R Pair", color="Score"),
        aspect="auto",
    )
    fig.update_layout(height=max(400, n * 22), margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GRAIL-Heart | Cardiac L-R Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#e94560;
               text-align:center; margin-bottom:0.2rem; }
.sub-header  { font-size:1rem; color:#888; text-align:center;
               margin-bottom:1.5rem; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 0.8rem 1rem; border-radius: 10px; color: white;
}
div[data-testid="stMetric"] label { color: #aaa !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════

if "adata" not in st.session_state:
    st.session_state.adata = None
if "upload_results" not in st.session_state:
    st.session_state.upload_results = None
if "upload_causal" not in st.session_state:
    st.session_state.upload_causal = None

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"## {bioicon('heart')} GRAIL-Heart", unsafe_allow_html=True)
    st.caption("Graph-based Reconstruction of Artificial Intercellular Links")
    st.markdown("---")

    model_result = load_model()
    if isinstance(model_result[1], dict):
        model, info = model_result
        st.success("Model loaded")
        st.metric("Parameters", f"{info['n_params']/1e6:.1f} M")
        st.metric(
            "Architecture",
            f"{info['n_gat_layers']}L / {info['n_heads']}H / {info['hidden_dim']}d",
        )
        st.metric(
            "Inverse Modelling",
            "Enabled" if info["has_inverse"] else "Disabled",
        )
        if info["n_pathways"]:
            st.metric("Pathways", f"{info['n_pathways']} ({info['n_mechano']} mechano)")
    else:
        model = None
        st.warning(f"Model: {model_result[1]}")

    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("📖 [GitHub](https://github.com/Tumo505/GRAIL-Heart)")
    st.markdown(
        "🕸️ [Network Explorer](https://tumo505.github.io/GRAIL-Heart/outputs/cytoscape/index.html)"
    )
    st.markdown(
        "📄 [Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6137179)"
    )

# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">GRAIL-Heart</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Causal Ligand-Receptor Analysis for Cardiac Spatial Transcriptomics"
    "</p>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════

tab_overview, tab_atlas, tab_upload, tab_forward, tab_inverse, tab_results = st.tabs(
    [
        "🫀 Overview",
        "🗺️ Atlas Explorer",
        "📤 Upload & Analyse",
        "🔬 Forward Model",
        "🔄 Inverse / Causal",
        "📊 Results",
    ]
)

# ── TAB 0: Overview ───────────────────────────────────────────────
with tab_overview:
    st.markdown("## What is GRAIL-Heart?")
    st.markdown(
        """
GRAIL-Heart is a **Graph Neural Network** framework for analysing cell-cell
communication in cardiac spatial transcriptomics data. It combines:

| Component | Description |
|-----------|-------------|
| **Forward modelling** | Predict which ligand-receptor pairs are actively signalling |
| **Inverse modelling** | Identify which L-R interactions *causally drive* cell fate decisions |
| **Counterfactual analysis** | Remove individual interactions and measure fate change |
| **Pathway grounding** | Link findings to MSigDB Hallmark + curated cardiac gene sets |

### How to use this app

1. **Atlas Explorer** — Browse pre-computed results from 6 Heart Cell Atlas v2 regions
2. **Upload & Analyse** — Upload your own data or load the demo dataset
3. **Forward Model** — Compute expression-based L-R interaction scores
4. **Inverse / Causal** — Run the full GNN inverse inference pipeline
5. **Results** — View, compare, and download all results
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cardiac Regions", "6")
    c2.metric("Total Cells", "42,654")
    c3.metric("L-R Pairs (OmniPath)", "22,234")
    c4.metric("Pathways Grounded", "35")

# ── TAB 1: Atlas Explorer ─────────────────────────────────────────
with tab_atlas:
    st.markdown(
        f"## {bioicon('umap')} Heart Cell Atlas v2 — Pre-computed Results",
        unsafe_allow_html=True,
    )

    cross_df = load_cross_region_comparison()
    if cross_df is not None:
        st.markdown("### Cross-Region Comparison")
        fig = make_cross_region_heatmap(cross_df, n=25)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Top 30 L-R interactions across all regions", expanded=False):
            st.dataframe(cross_df.nlargest(30, "mean_score"), use_container_width=True)

    st.markdown("---")

    st.markdown("### Per-Region Analysis")
    sel_region = st.selectbox(
        "Select cardiac region",
        options=REGIONS,
        format_func=lambda r: f"{r} — {REGION_NAMES[r]}",
    )

    region_df = load_precomputed_scores(sel_region)
    if region_df is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("L-R Pairs Detected", f"{len(region_df):,}")
        has_causal_col = "causal_score" in region_df.columns
        if has_causal_col and len(region_df) > 0:
            best_row = region_df.loc[region_df["causal_score"].idxmax()]
            m2.metric("Top Causal Pair", f"{best_row['ligand']}→{best_row['receptor']}")
            m3.metric("Max Causal Score", f"{best_row['causal_score']:.3f}")
        else:
            m2.metric(
                "Top Pair",
                f"{region_df.iloc[0]['ligand']}→{region_df.iloc[0]['receptor']}",
            )
            m3.metric("Max Score", f"{region_df.iloc[0]['mean_score']:.3f}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = make_lr_bar_chart(
                region_df, "mean_score", f"{sel_region} — Top by Expression", n=15
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if has_causal_col:
                causal_df = region_df[region_df["causal_score"] > 0].copy()
                if len(causal_df) > 0:
                    fig = make_lr_bar_chart(
                        causal_df, "causal_score",
                        f"{sel_region} — Top by Causal Score", n=15,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        score_col = (
            "causal_score"
            if has_causal_col and region_df["causal_score"].sum() > 0
            else "mean_score"
        )
        fig = make_network_graph(region_df, score_col, n=35)
        st.plotly_chart(fig, use_container_width=True)

        pw_fig = make_pathway_chart(region_df, score_col)
        if pw_fig:
            st.plotly_chart(pw_fig, use_container_width=True)

        csv = region_df.to_csv(index=False)
        st.download_button(
            f"Download {sel_region} scores (CSV)",
            data=csv,
            file_name=f"grail_heart_{sel_region}_scores.csv",
            mime="text/csv",
        )
    else:
        st.info(f"No pre-computed results for {sel_region}.")

# ── TAB 2: Upload & Analyse ───────────────────────────────────────
with tab_upload:
    st.markdown(f"## {bioicon('dna')} Upload Your Dataset", unsafe_allow_html=True)
    st.markdown(
        """
Upload spatial transcriptomics data in any of these formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| AnnData | `.h5ad` | **Recommended** — preserves cell type & spatial info |
| 10X Genomics | `.h5` | Raw feature-barcode matrix |
| CSV / TSV | `.csv`, `.tsv` | Gene expression matrix (genes x cells) |

**Requirements:**
- Gene symbols should be **standard HGNC** names (e.g. VEGFA, TGFB1, MMP2)
- For spatial analysis, the `.h5ad` should contain `adata.obsm['spatial']`
- For inverse modelling, cell type labels in `adata.obs['cell_type']` improve results
"""
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["h5ad", "h5", "csv", "tsv"],
        help="Upload scRNA-seq / spatial transcriptomics data",
    )

    if uploaded is not None:
        with st.spinner("Reading file …"):
            try:
                adata_raw = process_uploaded_file(uploaded)
                st.success(f"File loaded: **{uploaded.name}**")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                adata_raw = None

        if adata_raw is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Cells", f"{adata_raw.n_obs:,}")
            c2.metric("Genes", f"{adata_raw.n_vars:,}")
            c3.metric("File Size", f"{uploaded.size / 1e6:.1f} MB")

            has_spatial = "spatial" in adata_raw.obsm or "X_spatial" in adata_raw.obsm
            has_ct = "cell_type" in adata_raw.obs.columns

            s1, s2 = st.columns(2)
            s1.info(
                "Spatial coords: "
                + ("Found" if has_spatial else "Not found — will skip spatial graph")
            )
            s2.info(
                "Cell types: "
                + (
                    f"{adata_raw.obs['cell_type'].nunique()} types"
                    if has_ct
                    else "Not annotated"
                )
            )

            if has_ct:
                with st.expander("Cell type distribution"):
                    ct_counts = adata_raw.obs["cell_type"].value_counts()
                    fig = px.pie(
                        values=ct_counts.values,
                        names=ct_counts.index,
                        title="Cell Types",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Preprocessing")
            col1, col2 = st.columns(2)
            with col1:
                n_hvg = st.slider(
                    "Number of highly variable genes", 500, 5000, 2000, 100
                )
            with col2:
                k_neighbors = st.slider("k-nearest neighbours (graph)", 4, 15, 6)

            if st.button("Preprocess Data", type="primary"):
                with st.spinner("Preprocessing (QC -> normalise -> HVG -> scale) …"):
                    try:
                        adata_proc = preprocess_adata(adata_raw, n_top_genes=n_hvg)
                        adata_proc.uns["k_neighbors"] = k_neighbors
                        st.session_state.adata = adata_proc
                        st.success(
                            f"Preprocessed: {adata_proc.n_obs:,} cells x "
                            f"{adata_proc.n_vars:,} genes"
                        )
                    except Exception as e:
                        st.error(f"Preprocessing error: {e}")

    # ── Demo data section ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"### {bioicon('cardiomyocyte', 24)} Or Load Demo Data",
        unsafe_allow_html=True,
    )

    if DEMO_H5AD.exists():
        st.markdown(
            "Load a **500-cell subsample** from the Left Ventricle (LV) of the "
            "Heart Cell Atlas v2. Pre-processed and ready for forward or inverse analysis."
        )
        if st.button("🫀 Load Demo Data (LV, 500 cells)", type="primary"):
            with st.spinner("Loading demo dataset …"):
                adata = sc.read_h5ad(str(DEMO_H5AD))
                # Demo is already preprocessed; store directly
                adata.uns["k_neighbors"] = 6
                st.session_state.adata = adata
                st.success(
                    f"Demo loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes"
                )
    else:
        st.caption(
            "Demo data not found. Upload your own .h5ad file to get started."
        )


# ── TAB 3: Forward Modelling ──────────────────────────────────────
with tab_forward:
    st.markdown(
        f"## {bioicon('microscope')} Forward Modelling", unsafe_allow_html=True
    )
    st.markdown(
        """
**Gene Expression → L-R Interaction Scores**

Computes ligand x receptor expression product scores using the
OmniPath database (22,234 curated L-R pairs). This tells you
*which interactions are expressed*, not which ones are *causal*.
"""
    )

    if st.session_state.adata is None:
        st.warning("Upload and preprocess data first (Upload & Analyse tab)")
    else:
        adata = st.session_state.adata
        st.info(f"Data ready: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        if st.button("Run Forward Analysis", type="primary"):
            with st.spinner("Loading L-R database & scoring …"):
                lr_db = load_lr_database()
                expr_scores = compute_expression_scores(adata, lr_db)

            if len(expr_scores) == 0:
                st.error(
                    "No L-R pairs matched your gene names. "
                    "Check that you have HGNC symbols."
                )
            else:
                st.session_state.upload_results = expr_scores
                st.session_state.upload_causal = None
                st.success(f"{len(expr_scores)} L-R pairs scored")

                fig = make_lr_bar_chart(
                    expr_scores, "mean_score",
                    "Top L-R Interactions (Expression)", 20,
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = make_network_graph(expr_scores, "mean_score", 30)
                st.plotly_chart(fig2, use_container_width=True)


# ── TAB 4: Inverse / Causal ───────────────────────────────────────
with tab_inverse:
    st.markdown(
        f"## {bioicon('receptor')} Inverse Modelling — Causal L-R Inference",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
**L-R Interactions → Cell Fate Prediction → Counterfactual Scoring**

This runs the full GRAIL-Heart GNN pipeline:
1. Build spatial graph from your data
2. Run forward + inverse pass through the trained model
3. For each edge, measure how much removing it changes predicted cell fates
4. Aggregate per-edge causal scores to per-L-R-pair scores
"""
    )

    if st.session_state.adata is None:
        st.warning("Upload and preprocess data first (Upload & Analyse tab)")
    elif model is None:
        st.error("Model not loaded — check sidebar for errors")
    else:
        adata = st.session_state.adata
        has_spatial = "spatial" in adata.obsm or "X_spatial" in adata.obsm

        st.info(
            f"Data: {adata.n_obs:,} cells x {adata.n_vars:,} genes | "
            f"Spatial: {'Yes' if has_spatial else 'No'} | "
            f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params"
        )

        if not has_spatial:
            st.warning(
                "No spatial coordinates found. A random graph will be used. "
                "For best results, upload a .h5ad with `obsm['spatial']`."
            )

        if st.button("Run Causal Inference", type="primary"):
            progress = st.progress(0, text="Building spatial graph …")

            coords = _get_spatial_coords(adata)
            if not has_spatial:
                coords = np.random.randn(adata.n_obs, 2).astype(np.float32) * 100

            k = adata.uns.get("k_neighbors", 6)
            edge_index = build_spatial_graph(coords, k=k)
            progress.progress(20, text=f"Graph: {edge_index.shape[1]:,} edges")

            progress.progress(30, text="Computing expression scores …")
            lr_db = load_lr_database()
            expr_scores = compute_expression_scores(adata, lr_db)

            if len(expr_scores) == 0:
                st.error("No L-R pairs matched. Ensure HGNC gene symbols.")
            else:
                progress.progress(50, text="Running GNN forward + inverse pass …")
                try:
                    device = torch.device("cpu")  # CPU-only on HF Spaces
                    model.to(device)
                    model_out = run_model_inference(model, adata, edge_index)

                    progress.progress(80, text="Combining scores …")
                    combined = combine_scores(
                        expr_scores, model_out, adata, edge_index
                    )
                    st.session_state.upload_results = combined
                    st.session_state.upload_causal = model_out

                    progress.progress(100, text="Done!")

                    st.success(
                        f"Causal inference complete — {len(combined)} L-R pairs, "
                        f"{edge_index.shape[1]:,} edges"
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        fig = make_lr_bar_chart(
                            combined, "causal_score",
                            "Top Causal L-R Interactions", 15,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = make_network_graph(combined, "causal_score", 30)
                        st.plotly_chart(fig, use_container_width=True)

                    if "pathway_activation" in model_out:
                        st.markdown("### Pathway Activation (mean across cells)")
                        pa = model_out["pathway_activation"].mean(axis=0)
                        fig = px.bar(
                            x=pa,
                            y=[f"Pathway {i}" for i in range(len(pa))],
                            orientation="h",
                            title="Mean Pathway Activation",
                        )
                        fig.update_layout(
                            yaxis=dict(categoryorder="total ascending")
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ── TAB 5: Results ─────────────────────────────────────────────────
with tab_results:
    st.markdown(
        f"## {bioicon('heatmap')} Results & Download", unsafe_allow_html=True
    )

    if st.session_state.upload_results is None:
        st.info("Run Forward or Inverse analysis to see results here.")
    else:
        df = st.session_state.upload_results

        has_causal = "causal_score" in df.columns and df["causal_score"].sum() > 0
        analysis_type = "Inverse (Causal)" if has_causal else "Forward (Expression)"
        st.markdown(f"**Analysis type:** {analysis_type}")

        c1, c2, c3 = st.columns(3)
        c1.metric("L-R Pairs Scored", f"{len(df):,}")
        if has_causal:
            c2.metric(
                "Top Causal Pair",
                f"{df.iloc[0]['ligand']}→{df.iloc[0]['receptor']}",
            )
            c3.metric("Max Causal Score", f"{df.iloc[0]['causal_score']:.3f}")
        else:
            c2.metric(
                "Top Expression Pair",
                f"{df.iloc[0]['ligand']}→{df.iloc[0]['receptor']}",
            )
            c3.metric("Max Score", f"{df.iloc[0]['mean_score']:.4f}")

        st.markdown("### All Scored L-R Pairs")
        score_col = "causal_score" if has_causal else "mean_score"
        display_cols = [
            c
            for c in [
                "ligand", "receptor", "pathway", "mean_score",
                "causal_score", "ligand_expr", "receptor_expr",
            ]
            if c in df.columns
        ]
        st.dataframe(
            df[display_cols].head(200), use_container_width=True, height=400
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig = make_lr_bar_chart(df, score_col, f"Top 20 by {score_col}", 20)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            pw_fig = make_pathway_chart(df, score_col)
            if pw_fig:
                st.plotly_chart(pw_fig, use_container_width=True)

        fig = make_network_graph(df, score_col, 40)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Download")
        c1, c2 = st.columns(2)
        with c1:
            csv = df.to_csv(index=False)
            st.download_button(
                "Download L-R Scores (CSV)",
                data=csv,
                file_name="grail_heart_lr_scores.csv",
                mime="text/csv",
            )
        with c2:
            if has_causal and st.session_state.upload_causal:
                out = st.session_state.upload_causal
                if "causal_scores" in out:
                    causal_csv = pd.DataFrame({
                        "edge_idx": range(len(out["causal_scores"])),
                        "causal_score": out["causal_scores"],
                    }).to_csv(index=False)
                    st.download_button(
                        "Download Per-Edge Causal Scores (CSV)",
                        data=causal_csv,
                        file_name="grail_heart_edge_causal_scores.csv",
                        mime="text/csv",
                    )

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "GRAIL-Heart v2.0 — Graph-based Reconstruction of Artificial Intercellular Links | "
    "[GitHub](https://github.com/Tumo505/GRAIL-Heart) | "
    "[Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6137179) | "
    "Trained on Heart Cell Atlas v2 (42,654 cells, 6 regions)"
)
