#!/usr/bin/env python3
"""
Generate all bundled assets for Hugging Face Spaces deployment.

This script creates:
  1. model/best.pt          — Stripped checkpoint (weights only, ~50 MB)
  2. demo_data/tables/*.csv — Pre-computed L-R scores (copied from outputs/)
  3. demo_data/demo_lv_500cells.h5ad — 500-cell LV subsample for demo
  4. demo_data/lr_database_cache.csv — Cached OmniPath L-R database

Run from the project root:
    python huggingface/prepare_demo_data.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

# ── Resolve paths ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HF_DIR = SCRIPT_DIR  # huggingface/

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Output directories ─────────────────────────────────────────────
MODEL_DIR = HF_DIR / "model"
DEMO_DIR = HF_DIR / "demo_data"
TABLES_DIR = DEMO_DIR / "tables"

MODEL_DIR.mkdir(exist_ok=True)
DEMO_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def strip_checkpoint():
    """Create a weights-only checkpoint (~50 MB instead of ~125 MB)."""
    import torch

    src = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
    dst = MODEL_DIR / "best.pt"

    if not src.exists():
        print(f"  [SKIP] Checkpoint not found: {src}")
        return

    print(f"  Loading full checkpoint ({src.stat().st_size / 1e6:.1f} MB) …")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "epoch": ckpt.get("epoch", 0),
    }
    torch.save(slim, str(dst))
    print(f"  Saved stripped checkpoint: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def copy_precomputed_tables():
    """Copy pre-computed L-R score CSVs."""
    src_dir = PROJECT_ROOT / "outputs" / "enhanced_analysis" / "tables"
    if not src_dir.exists():
        print(f"  [SKIP] Tables not found: {src_dir}")
        return

    count = 0
    for csv_file in src_dir.glob("*.csv"):
        shutil.copy2(csv_file, TABLES_DIR / csv_file.name)
        count += 1
    print(f"  Copied {count} CSV files to {TABLES_DIR}")


def create_demo_adata():
    """Create a 500-cell LV subsample as demo data.

    Tries multiple possible file locations for the HCA v2 LV data.
    Falls back to creating a synthetic demo if no real data is found.
    """
    import scanpy as sc
    import anndata as ad

    dst = DEMO_DIR / "demo_lv_500cells.h5ad"
    N_CELLS = 500

    # Try to find real LV data
    candidates = [
        PROJECT_ROOT / "data" / "HeartCellAtlasv2" / "visium-OCT_LV_raw.h5ad",
        PROJECT_ROOT / "data" / "HeartCellAtlasv2" / "LV.h5ad",
        PROJECT_ROOT / "data" / "HeartCellAtlasv2" / "visium-OCT_LV.h5ad",
    ]

    # Also try any file with 'LV' in the name
    hca_dir = PROJECT_ROOT / "data" / "HeartCellAtlasv2"
    if hca_dir.exists():
        candidates.extend(sorted(hca_dir.glob("*LV*")))

    adata = None
    for path in candidates:
        if path.exists() and path.suffix == ".h5ad":
            print(f"  Loading LV data from {path.name} …")
            try:
                adata = sc.read_h5ad(str(path))
                print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
                break
            except Exception as e:
                print(f"  [WARN] Failed to read {path.name}: {e}")

    if adata is not None:
        import scipy.sparse as sp

        # Convert Ensembl IDs → HGNC symbols if a SYMBOL column exists
        sample = list(adata.var_names[:200])
        ens_frac = sum(1 for g in sample if str(g).startswith("ENS")) / max(len(sample), 1)
        if ens_frac > 0.5:
            for col in ["SYMBOL", "gene_symbol", "gene_symbols", "gene_name",
                        "gene_short_name", "symbol", "name", "feature_name"]:
                if col in adata.var.columns:
                    symbols = adata.var[col].astype(str).values
                    valid = np.array([
                        s != "" and s != "nan" and s != "None" and not s.startswith("ENS")
                        for s in symbols
                    ])
                    if valid.sum() >= 500:
                        adata = adata[:, valid].copy()
                        adata.var_names = adata.var[col].values.astype(str)
                        adata.var_names_make_unique()
                        print(f"  Converted var_names to HGNC via '{col}' column")
                    break

        # Subsample
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, min(N_CELLS, adata.n_obs), replace=False)
        demo = adata[sorted(idx)].copy()

        # Basic preprocessing if not already done
        xmax = demo.X.max()
        if hasattr(xmax, "item"):
            xmax = xmax.item()
        if xmax > 50:
            sc.pp.normalize_total(demo, target_sum=1e4)
            sc.pp.log1p(demo)

        # Store log-normalised for L-R scoring
        demo.layers["log_norm"] = demo.X.copy()

        # HVG selection to keep file small
        if demo.n_vars > 2500:
            sc.pp.highly_variable_genes(demo, n_top_genes=2000, subset=True)

        # Scale for model input — keep result as dense float32 (small: 500x2000)
        sc.pp.scale(demo, max_value=10)

        # Ensure X is dense float32 (500 cells x 2000 genes = 4 MB)
        if sp.issparse(demo.X):
            demo.X = np.array(demo.X.todense(), dtype=np.float32)
        else:
            demo.X = np.array(demo.X, dtype=np.float32)

        # Ensure log_norm layer is also float32
        if sp.issparse(demo.layers["log_norm"]):
            demo.layers["log_norm"] = np.array(
                demo.layers["log_norm"].todense(), dtype=np.float32
            )
        else:
            demo.layers["log_norm"] = np.array(
                demo.layers["log_norm"], dtype=np.float32
            )

        # Strip bulky metadata to minimise file size
        demo.uns = {}
        demo.varm = {}
        demo.obsp = {}
        demo.varp = {}

        demo.write_h5ad(str(dst), compression="gzip")
        size_mb = dst.stat().st_size / 1e6
        print(
            f"  Created demo: {demo.n_obs} cells x {demo.n_vars} genes "
            f"({size_mb:.1f} MB)"
        )
    else:
        # Fallback: create a synthetic demo so the app still loads
        print("  [WARN] No LV data found — creating synthetic demo")
        n_genes = 2000
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        # Sprinkle in some real cardiac gene names for L-R matching
        real_genes = [
            "TIMP1", "MMP2", "TIMP2", "TGFB1", "TGFBR2", "BMP2", "BMPR1A",
            "VEGFA", "FLT1", "FN1", "ITGA5", "COL1A1", "DDR2", "COL4A1",
            "COL4A2", "SERPING1", "C1S", "CFD", "C3", "THBS1", "CD36",
            "S100A8", "ITGB2", "CXCL10", "CCR7", "CCL5", "DPP4",
        ]
        for i, g in enumerate(real_genes):
            if i < n_genes:
                gene_names[i] = g

        X = np.random.randn(N_CELLS, n_genes).astype(np.float32)
        coords = np.random.randn(N_CELLS, 2).astype(np.float32) * 500

        demo = ad.AnnData(
            X=X,
            obs={"cell_type": np.random.choice(
                ["Cardiomyocyte", "Fibroblast", "Endothelial", "Macrophage"],
                N_CELLS,
            )},
        )
        demo.var_names = gene_names
        demo.obsm["spatial"] = coords
        demo.layers["log_norm"] = np.abs(X) * 0.5  # non-negative for scoring

        demo.write_h5ad(str(dst))
        size_mb = dst.stat().st_size / 1e6
        print(
            f"  Created synthetic demo: {N_CELLS} cells x {n_genes} genes "
            f"({size_mb:.1f} MB)"
        )


def cache_lr_database():
    """Cache the OmniPath L-R database to avoid network calls on HF."""
    dst = DEMO_DIR / "lr_database_cache.csv"

    # Try to copy existing cache from project data/ dir
    project_cache = PROJECT_ROOT / "data" / "lr_database_cache.csv"
    if project_cache.exists():
        shutil.copy2(project_cache, dst)
        print(f"  Copied LR cache from {project_cache.name}")
        return

    # Otherwise, download and annotate
    try:
        from grail_heart.data.cellchat_database import (
            get_omnipath_lr_database,
            annotate_cardiac_pathways,
        )

        print("  Downloading OmniPath L-R database …")
        lr = get_omnipath_lr_database()
        lr = annotate_cardiac_pathways(lr)
        lr.to_csv(str(dst), index=False)
        print(f"  Cached {len(lr)} L-R pairs to {dst.name}")
    except Exception as e:
        print(f"  [WARN] Could not cache L-R database: {e}")


def copy_src_modules():
    """Copy the minimal src/grail_heart modules needed for inference."""
    src = PROJECT_ROOT / "src" / "grail_heart"
    dst = HF_DIR / "src" / "grail_heart"

    if dst.exists():
        shutil.rmtree(dst)

    # Copy models/ and data/ (needed by app.py)
    for subdir in ["models", "data"]:
        src_sub = src / subdir
        dst_sub = dst / subdir
        if src_sub.exists():
            shutil.copytree(src_sub, dst_sub)

    # Copy __init__.py
    init_src = src / "__init__.py"
    if init_src.exists():
        shutil.copy2(init_src, dst / "__init__.py")
    else:
        (dst / "__init__.py").write_text("")

    # Copy utils/ if it exists (models may import from it)
    utils_src = src / "utils"
    if utils_src.exists():
        shutil.copytree(utils_src, dst / "utils")

    # Count files
    n_files = sum(1 for _ in dst.rglob("*.py"))
    print(f"  Copied {n_files} Python files to {dst.relative_to(HF_DIR)}")


def main():
    print("=" * 60)
    print("GRAIL-Heart — Preparing Hugging Face Spaces deployment")
    print("=" * 60)

    print("\n[1/5] Stripping checkpoint …")
    strip_checkpoint()

    print("\n[2/5] Copying pre-computed score tables …")
    copy_precomputed_tables()

    print("\n[3/5] Creating demo AnnData …")
    create_demo_adata()

    print("\n[4/5] Caching L-R database …")
    cache_lr_database()

    print("\n[5/5] Copying source modules …")
    copy_src_modules()

    print("\n" + "=" * 60)
    print("Done! HF Space contents are in:")
    print(f"  {HF_DIR}")
    print()
    print("Directory structure:")
    for p in sorted(HF_DIR.rglob("*")):
        if p.is_file() and "__pycache__" not in str(p):
            rel = p.relative_to(HF_DIR)
            size = p.stat().st_size
            if size > 1e6:
                print(f"  {rel}  ({size / 1e6:.1f} MB)")
            else:
                print(f"  {rel}  ({size / 1e3:.0f} KB)")

    print()
    print("Next steps:")
    print("  1. cd huggingface/")
    print("  2. huggingface-cli login")
    print("  3. huggingface-cli repo create grail-heart-demo --type space --space-sdk streamlit")
    print("  4. git clone https://huggingface.co/spaces/YOUR_USERNAME/grail-heart-demo")
    print("  5. Copy all files from huggingface/ into the cloned repo")
    print("  6. git lfs install && git add . && git commit -m 'Initial deploy' && git push")


if __name__ == "__main__":
    main()
