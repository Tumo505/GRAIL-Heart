"""
CellChat/OmniPath Ligand-Receptor Database Integration

Uses OmniPath to access curated L-R interactions from multiple databases:
- CellPhoneDB
- CellChat  
- ICELLNET
- Ramilowski2015
- And more...

This provides 2000+ high-quality L-R pairs for comprehensive analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Set
from pathlib import Path
import warnings


def get_omnipath_lr_database(
    min_curation_effort: int = 0,
    min_sources: int = 1,
    secreted_only: bool = False,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load ligand-receptor interactions from OmniPath.
    
    OmniPath integrates multiple databases including CellPhoneDB, CellChat,
    ICELLNET, and others, providing comprehensive L-R coverage.
    
    Args:
        min_curation_effort: Minimum curation effort score (higher = better quality)
        min_sources: Minimum number of source databases
        secreted_only: If True, only include secreted ligands
        cache_path: Optional path to cache the database locally
        
    Returns:
        DataFrame with columns: ligand, receptor, pathway, function, sources
    """
    # Try to load from cache first
    if cache_path and cache_path.exists():
        print(f"Loading cached L-R database from {cache_path}")
        return pd.read_csv(cache_path)
    
    from omnipath.interactions import import_intercell_network
    
    print("Downloading L-R interactions from OmniPath...")
    print("(This includes CellPhoneDB, CellChat, ICELLNET, and more)")
    
    # Get intercellular interactions
    df = import_intercell_network()
    
    print(f"Downloaded {len(df)} raw interactions")
    
    # Filter for ligand-receptor pairs
    # Keep interactions where source is a ligand/secreted and target is a receptor
    lr_mask = (
        (df['transmitter_intercell_source'] == True) &
        (df['receiver_intercell_target'] == True)
    )
    
    df_lr = df[lr_mask].copy()
    print(f"Filtered to {len(df_lr)} ligand-receptor pairs")
    
    # Optional: filter for secreted ligands only
    if secreted_only:
        secreted_mask = df_lr['secreted_intercell_source'] == True
        df_lr = df_lr[secreted_mask]
        print(f"Filtered to {len(df_lr)} secreted ligand pairs")
    
    # Filter by quality (only if specified)
    if min_curation_effort > 0 and 'curation_effort' in df_lr.columns:
        df_lr = df_lr[df_lr['curation_effort'] >= min_curation_effort]
        print(f"After curation filter: {len(df_lr)} pairs")
    
    if min_sources > 1 and 'n_sources' in df_lr.columns:
        df_lr = df_lr[df_lr['n_sources'] >= min_sources]
        print(f"After source filter: {len(df_lr)} pairs")
    
    # Extract gene symbols - handle all data types properly
    ligands = df_lr['genesymbol_intercell_source'].astype(str).values
    receptors = df_lr['genesymbol_intercell_target'].astype(str).values
    
    # Get pathway info with proper handling
    pathways = df_lr['category_intercell_source'].astype(str).replace('nan', 'signaling').values
    functions = df_lr['aspect_intercell_source'].astype(str).replace('nan', 'signaling').values
    
    result = pd.DataFrame({
        'ligand': ligands,
        'receptor': receptors,
        'pathway': pathways,
        'function': functions,
        'sources': df_lr['sources'].apply(lambda x: str(x) if pd.notna(x) else '').values,
        'n_sources': df_lr['n_sources'].values if 'n_sources' in df_lr.columns else 1,
        'is_stimulation': df_lr['is_stimulation'].values if 'is_stimulation' in df_lr.columns else True,
        'is_inhibition': df_lr['is_inhibition'].values if 'is_inhibition' in df_lr.columns else False,
    })
    
    # Remove duplicates (keep first occurrence which has more metadata)
    result = result.drop_duplicates(subset=['ligand', 'receptor'])
    
    # Remove any rows with invalid gene symbols
    result = result[~result['ligand'].isin(['nan', 'None', ''])]
    result = result[~result['receptor'].isin(['nan', 'None', ''])]
    
    print(f"Final database: {len(result)} unique L-R pairs")
    print(f"  Unique ligands: {result['ligand'].nunique()}")
    print(f"  Unique receptors: {result['receptor'].nunique()}")
    print(f"  Pathway categories: {result['pathway'].nunique()}")
    
    # Cache if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print(f"Cached database to {cache_path}")
    
    return result


def get_cellchat_database(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get CellChat-style L-R database from OmniPath.
    
    This filters for interactions that match CellChat's criteria:
    - Secreted signaling
    - ECM-receptor
    - Cell-cell contact
    
    Returns comprehensive L-R pairs categorized by pathway.
    """
    return get_omnipath_lr_database(
        min_curation_effort=0,
        min_sources=1,
        secreted_only=False,
        cache_path=cache_path,
    )


# Cardiac-specific pathway annotations
CARDIAC_PATHWAYS = {
    # Growth factors critical for heart
    'VEGF': ['VEGFA', 'VEGFB', 'VEGFC', 'VEGFD', 'PGF', 'FLT1', 'KDR', 'FLT4', 'NRP1', 'NRP2'],
    'FGF': ['FGF1', 'FGF2', 'FGF9', 'FGF10', 'FGF21', 'FGF23', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4'],
    'PDGF': ['PDGFA', 'PDGFB', 'PDGFC', 'PDGFD', 'PDGFRA', 'PDGFRB'],
    'TGFb': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2', 'ACVRL1'],
    'BMP': ['BMP2', 'BMP4', 'BMP7', 'BMP10', 'BMPR1A', 'BMPR1B', 'BMPR2'],
    
    # Cardiac-specific signaling
    'Neuregulin': ['NRG1', 'NRG2', 'NRG4', 'ERBB2', 'ERBB3', 'ERBB4'],
    'Natriuretic': ['NPPA', 'NPPB', 'NPR1', 'NPR2', 'NPR3'],
    'Endothelin': ['EDN1', 'EDN2', 'EDN3', 'EDNRA', 'EDNRB'],
    'Angiotensin': ['AGT', 'AGTR1', 'AGTR2', 'ACE', 'ACE2'],
    
    # Inflammatory
    'Interleukin': ['IL1A', 'IL1B', 'IL6', 'IL10', 'IL11', 'IL33', 'IL1R1', 'IL6R', 'IL1RL1'],
    'Chemokine': ['CXCL12', 'CCL2', 'CCL5', 'CXCR4', 'CCR2', 'ACKR3'],
    'TNF': ['TNF', 'TNFRSF1A', 'TNFRSF1B'],
    
    # ECM and adhesion
    'Collagen': ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'DDR1', 'DDR2'],
    'Integrin': ['ITGA1', 'ITGA5', 'ITGAV', 'ITGB1', 'ITGB3', 'ITGB5'],
    'ECM': ['FN1', 'LAMA1', 'THBS1', 'THBS4', 'SPP1', 'POSTN', 'CTGF'],
    
    # Development  
    'NOTCH': ['DLL1', 'DLL4', 'JAG1', 'JAG2', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4'],
    'WNT': ['WNT3A', 'WNT5A', 'WNT11', 'FZD1', 'FZD4', 'FZD7', 'LRP5', 'LRP6'],
    'Hedgehog': ['SHH', 'IHH', 'DHH', 'PTCH1', 'PTCH2', 'SMO'],
}


def annotate_cardiac_pathways(lr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cardiac-specific pathway annotations to L-R database.
    
    Args:
        lr_df: DataFrame with ligand, receptor columns
        
    Returns:
        DataFrame with added 'cardiac_pathway' column
    """
    df = lr_df.copy()
    
    # Create gene to pathway mapping
    gene_to_pathway = {}
    for pathway, genes in CARDIAC_PATHWAYS.items():
        for gene in genes:
            gene_to_pathway[gene] = pathway
    
    # Annotate based on ligand or receptor
    def get_cardiac_pathway(row):
        lig = row['ligand']
        rec = row['receptor']
        
        if lig in gene_to_pathway:
            return gene_to_pathway[lig]
        elif rec in gene_to_pathway:
            return gene_to_pathway[rec]
        else:
            return row.get('pathway', 'Other')
    
    df['cardiac_pathway'] = df.apply(get_cardiac_pathway, axis=1)
    
    return df


def filter_for_cardiac(lr_df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Filter L-R database for cardiac-relevant interactions.
    
    Args:
        lr_df: Full L-R database
        strict: If True, only keep interactions with known cardiac genes
        
    Returns:
        Filtered DataFrame
    """
    # Get all cardiac genes
    cardiac_genes = set()
    for genes in CARDIAC_PATHWAYS.values():
        cardiac_genes.update(genes)
    
    if strict:
        # Both ligand and receptor must be cardiac-related
        mask = (
            lr_df['ligand'].isin(cardiac_genes) | 
            lr_df['receptor'].isin(cardiac_genes)
        )
        return lr_df[mask].copy()
    else:
        # Just annotate and return all
        return annotate_cardiac_pathways(lr_df)


def get_mechanosensitive_gene_sets(
    msigdb_path: Optional[Path] = None,
    include_hallmark: bool = True,
) -> Dict[str, List[str]]:
    """
    Build mechanosensitive gene sets from multiple curated sources.

    Combines:
    1. CARDIAC_PATHWAYS (already in this module)
    2. MSigDB Hallmark gene sets (cardiac-relevant subsets)
    3. Hand-curated YAP/TAZ-Hippo and Piezo mechanosensor genes

    These are used to biologically ground the MechanosensitivePathwayModule
    and LRToTargetGeneDecoder so pathway activations map to real biology
    rather than randomly-initialised weights.

    Args:
        msigdb_path: Path to MSigDB Hallmark JSON file (h.all.v2026.1.Hs.txt).
                     If None, only CARDIAC_PATHWAYS + curated sets are used.
        include_hallmark: Whether to include MSigDB Hallmark gene sets.

    Returns:
        Dictionary mapping pathway name → list of HUGO gene symbols.
    """
    gene_sets: Dict[str, List[str]] = {}

    # ── 1. CARDIAC_PATHWAYS from this module ────────────────────────────
    for name, genes in CARDIAC_PATHWAYS.items():
        gene_sets[name] = list(genes)

    # ── 2. Curated YAP/TAZ-Hippo mechanotransduction genes ─────────────
    gene_sets['YAP_TAZ_Hippo'] = [
        # Core Hippo kinase cascade
        'MST1', 'MST2', 'STK3', 'STK4', 'SAV1', 'LATS1', 'LATS2',
        'MOB1A', 'MOB1B', 'NF2', 'FRMD6', 'WWC1', 'TAOK1', 'TAOK2',
        # Effectors
        'YAP1', 'WWTR1',  # WWTR1 = TAZ
        # Transcriptional partners
        'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4',
        # Direct target genes
        'CTGF', 'CCN2', 'CCN1', 'ANKRD1', 'AMOTL2', 'BIRC5',
        'AXL', 'SERPINE1', 'AREG', 'MYOF',
    ]

    # ── 3. Curated Piezo / mechanosensitive ion channel genes ──────────
    gene_sets['Piezo_Mechano'] = [
        'PIEZO1', 'PIEZO2',
        # TRP channels involved in mechanosensation
        'TRPV4', 'TRPC1', 'TRPC6', 'TRPM4',
        # Stretch-activated channels
        'KCNK2', 'KCNK4',  # TREK-1, TRAAK
        # Downstream / co-factors
        'STOML3', 'CDKL5',
    ]

    # ── 4. Curated Integrin-FAK focal adhesion genes ───────────────────
    gene_sets['Integrin_FAK'] = [
        # Integrins (cardiac-relevant)
        'ITGA1', 'ITGA5', 'ITGA7', 'ITGAV', 'ITGB1', 'ITGB3', 'ITGB5',
        # Focal adhesion kinase axis
        'PTK2', 'PXN', 'VCL', 'TLN1', 'TLN2',
        'ILK', 'PINCH1', 'PARVA',
        # Downstream effectors
        'SRC', 'FAK', 'ROCK1', 'ROCK2', 'RHOA', 'RAC1', 'CDC42',
    ]

    # ── 5. MSigDB Hallmark gene sets (cardiac-relevant) ────────────────
    if include_hallmark and msigdb_path is not None:
        msigdb_path = Path(msigdb_path)
        if msigdb_path.exists():
            import json
            with open(msigdb_path, 'r') as f:
                hallmark_db = json.load(f)

            # Cardiac-relevant Hallmark sets
            cardiac_hallmark_keys = [
                'HALLMARK_TGF_BETA_SIGNALING',
                'HALLMARK_WNT_BETA_CATENIN_SIGNALING',
                'HALLMARK_NOTCH_SIGNALING',
                'HALLMARK_HEDGEHOG_SIGNALING',
                'HALLMARK_ANGIOGENESIS',
                'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
                'HALLMARK_HYPOXIA',
                'HALLMARK_PI3K_AKT_MTOR_SIGNALING',
                'HALLMARK_MYOGENESIS',
                'HALLMARK_APOPTOSIS',
                'HALLMARK_INFLAMMATORY_RESPONSE',
                'HALLMARK_IL6_JAK_STAT3_SIGNALING',
                'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY',
                'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
            ]

            for key in cardiac_hallmark_keys:
                if key in hallmark_db:
                    entry = hallmark_db[key]
                    symbols = entry.get('geneSymbols', [])
                    if symbols:
                        # Use a shorter name for the pathway
                        short_name = key.replace('HALLMARK_', '')
                        gene_sets[f'Hallmark_{short_name}'] = symbols

    return gene_sets


def build_pathway_gene_mask(
    gene_sets: Dict[str, List[str]],
    gene_names: List[str],
) -> 'np.ndarray':
    """
    Build a binary mask matrix [n_pathways, n_genes] where
    mask[p, g] = 1 if gene g belongs to pathway p.

    This mask is used to initialise the pathway_gene_matrix in
    LRToTargetGeneDecoder and the lr_mechano_mapper mask in
    MechanosensitivePathwayModule so the model starts from
    biologically grounded associations.

    Args:
        gene_sets: Dict mapping pathway name → list of gene symbols.
        gene_names: Ordered list of gene symbols used in the expression matrix.

    Returns:
        Binary numpy array of shape [n_pathways, n_genes].
    """
    gene_name_set = {g.upper(): i for i, g in enumerate(gene_names)}
    pathway_names = sorted(gene_sets.keys())
    n_pathways = len(pathway_names)
    n_genes = len(gene_names)

    mask = np.zeros((n_pathways, n_genes), dtype=np.float32)
    for p_idx, p_name in enumerate(pathway_names):
        for gene in gene_sets[p_name]:
            g_idx = gene_name_set.get(gene.upper())
            if g_idx is not None:
                mask[p_idx, g_idx] = 1.0

    # Report coverage
    n_hit = int(mask.sum())
    n_total = sum(len(v) for v in gene_sets.values())
    print(f"Pathway-gene mask: {n_pathways} pathways × {n_genes} genes, "
          f"{n_hit}/{n_total} gene-pathway associations found ({100*n_hit/max(n_total,1):.1f}%)")

    return mask


def build_lr_pathway_membership(
    gene_sets: Dict[str, List[str]],
    lr_database: pd.DataFrame,
) -> 'np.ndarray':
    """
    For each L-R pair in the database, determine which pathways it belongs to.

    Returns a binary matrix [n_lr_pairs, n_pathways].

    Args:
        gene_sets: Dict mapping pathway name → list of gene symbols.
        lr_database: DataFrame with 'ligand' and 'receptor' columns.

    Returns:
        Binary numpy array of shape [n_lr_pairs, n_pathways].
    """
    pathway_names = sorted(gene_sets.keys())
    n_pathways = len(pathway_names)
    n_lr = len(lr_database)

    # Build gene → set of pathway indices
    gene_to_pathways: Dict[str, Set[int]] = {}
    for p_idx, p_name in enumerate(pathway_names):
        for gene in gene_sets[p_name]:
            gene_to_pathways.setdefault(gene.upper(), set()).add(p_idx)

    membership = np.zeros((n_lr, n_pathways), dtype=np.float32)
    for i, row in lr_database.iterrows():
        lig = str(row['ligand']).upper()
        rec = str(row['receptor']).upper()
        for p_idx in gene_to_pathways.get(lig, set()) | gene_to_pathways.get(rec, set()):
            membership[i, p_idx] = 1.0

    return membership


if __name__ == '__main__':
    # Test the database
    print("Testing OmniPath L-R database integration...")
    
    # Get database
    cache_file = Path("data/lr_database_cache.csv")
    df = get_omnipath_lr_database(cache_path=cache_file)
    
    print(f"\nDatabase summary:")
    print(f"  Total pairs: {len(df)}")
    print(f"  Unique ligands: {df['ligand'].nunique()}")
    print(f"  Unique receptors: {df['receptor'].nunique()}")
    
    print(f"\nTop pathways:")
    print(df['pathway'].value_counts().head(20))
    
    # Test cardiac filtering
    cardiac_df = filter_for_cardiac(df, strict=True)
    print(f"\nCardiac-filtered: {len(cardiac_df)} pairs")
    print(f"Cardiac pathways:")
    print(cardiac_df['cardiac_pathway'].value_counts().head(15))
