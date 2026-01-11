#!/usr/bin/env python3
"""
Generate network data JSON files from GRAIL-Heart analysis outputs.
These JSON files are then loaded by the interactive network explorer.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_lr_scores(analysis_dir: Path) -> dict:
    """Load L-R scores from all regions."""
    regions = ['AX', 'LA', 'LV', 'RA', 'RV', 'SP']
    all_data = {}
    
    for region in regions:
        lr_file = analysis_dir / 'tables' / f'{region}_lr_scores.csv'
        if lr_file.exists():
            df = pd.read_csv(lr_file)
            all_data[region] = df
            print(f"Loaded {len(df)} L-R pairs for {region}")
        else:
            print(f"Warning: {lr_file} not found")
    
    return all_data

def build_network_data(df: pd.DataFrame, top_n: int = None, min_score: float = 0.0) -> dict:
    """Build network nodes and edges from L-R score dataframe.
    
    Args:
        df: DataFrame with L-R pairs
        top_n: If set, only include top N edges by score. None = include all.
        min_score: Minimum score threshold
    """
    
    # Detect column names
    if 'ligand' in df.columns:
        source_col, target_col = 'ligand', 'receptor'
    elif 'source' in df.columns:
        source_col, target_col = 'source', 'target'
    else:
        raise ValueError(f"Unknown columns: {df.columns.tolist()}")
    
    # Score column
    score_col = None
    for col in ['causal_score', 'score', 'weight', 'mean_score']:
        if col in df.columns:
            score_col = col
            break
    
    if score_col is None:
        # Use first numeric column
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            score_col = numeric_cols[0]
        else:
            df['score'] = 1.0
            score_col = 'score'
    
    # Filter and sort
    df_filtered = df[df[score_col] >= min_score].copy()
    df_filtered = df_filtered.sort_values(score_col, ascending=False)
    
    # Apply top_n limit if specified
    if top_n is not None:
        df_filtered = df_filtered.head(top_n)
    
    # Build node set and calculate degrees
    node_info = defaultdict(lambda: {'type': set(), 'degree': 0})
    
    for _, row in df_filtered.iterrows():
        source = row[source_col]
        target = row[target_col]
        
        node_info[source]['type'].add('ligand')
        node_info[source]['degree'] += 1
        
        node_info[target]['type'].add('receptor')
        node_info[target]['degree'] += 1
    
    # Create nodes list
    nodes = []
    for gene, info in node_info.items():
        if 'ligand' in info['type'] and 'receptor' in info['type']:
            node_type = 'dual'
        elif 'ligand' in info['type']:
            node_type = 'ligand'
        else:
            node_type = 'receptor'
        
        nodes.append({
            'id': gene,
            'type': node_type,
            'degree': info['degree']
        })
    
    # Create edges list
    edges = []
    for _, row in df_filtered.iterrows():
        edges.append({
            'source': row[source_col],
            'target': row[target_col],
            'weight': round(float(row[score_col]), 3)
        })
    
    return {
        'nodes': nodes,
        'edges': edges
    }

def build_integrated_network(all_region_data: dict, top_n: int = None) -> dict:
    """Build integrated network from all regions.
    
    Args:
        all_region_data: Dict of region -> DataFrame
        top_n: If set, only include top N edges. None = include all.
    """
    
    # Combine all data
    combined_edges = []
    
    for region, df in all_region_data.items():
        # Detect columns
        if 'ligand' in df.columns:
            source_col, target_col = 'ligand', 'receptor'
        elif 'source' in df.columns:
            source_col, target_col = 'source', 'target'
        else:
            continue
        
        # Score column
        score_col = None
        for col in ['causal_score', 'score', 'weight', 'mean_score']:
            if col in df.columns:
                score_col = col
                break
        
        if score_col is None:
            continue
        
        for _, row in df.iterrows():
            combined_edges.append({
                'source': row[source_col],
                'target': row[target_col],
                'weight': float(row[score_col]),
                'region': region
            })
    
    # Sort by weight and take top N (if specified)
    combined_edges.sort(key=lambda x: x['weight'], reverse=True)
    if top_n is not None:
        top_edges = combined_edges[:top_n]
    else:
        top_edges = combined_edges
    
    # Build nodes
    node_info = defaultdict(lambda: {'type': set(), 'degree': 0, 'regions': set()})
    
    for edge in top_edges:
        node_info[edge['source']]['type'].add('ligand')
        node_info[edge['source']]['degree'] += 1
        node_info[edge['source']]['regions'].add(edge['region'])
        
        node_info[edge['target']]['type'].add('receptor')
        node_info[edge['target']]['degree'] += 1
        node_info[edge['target']]['regions'].add(edge['region'])
    
    nodes = []
    for gene, info in node_info.items():
        if 'ligand' in info['type'] and 'receptor' in info['type']:
            node_type = 'dual'
        elif 'ligand' in info['type']:
            node_type = 'ligand'
        else:
            node_type = 'receptor'
        
        nodes.append({
            'id': gene,
            'type': node_type,
            'degree': info['degree'],
            'regions': list(info['regions'])
        })
    
    edges = [{
        'source': e['source'],
        'target': e['target'],
        'weight': round(e['weight'], 3),
        'region': e['region']
    } for e in top_edges]
    
    return {
        'nodes': nodes,
        'edges': edges
    }

def main():
    # Paths
    project_dir = Path(__file__).parent.parent
    analysis_dir = project_dir / 'outputs' / 'enhanced_analysis'
    output_dir = project_dir / 'outputs' / 'cytoscape' / 'data'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading L-R scores from analysis outputs...")
    all_data = load_lr_scores(analysis_dir)
    
    if not all_data:
        print("No data found! Check paths.")
        return
    
    # Generate per-region networks (ALL edges)
    all_networks = {}
    
    for region, df in all_data.items():
        print(f"\nProcessing {region}...")
        # Include ALL edges (top_n=None), no minimum score filter
        network = build_network_data(df, top_n=None, min_score=0.0)
        all_networks[region] = network
        
        # Save individual region file
        region_file = output_dir / f'{region}_network.json'
        with open(region_file, 'w') as f:
            json.dump(network, f, indent=2)
        print(f"  Saved {region_file.name}: {len(network['nodes'])} nodes, {len(network['edges'])} edges")
    
    # Generate integrated network (ALL edges from all regions)
    print("\nBuilding integrated network...")
    integrated = build_integrated_network(all_data, top_n=None)
    all_networks['integrated'] = integrated
    
    integrated_file = output_dir / 'integrated_network.json'
    with open(integrated_file, 'w') as f:
        json.dump(integrated, f, indent=2)
    print(f"Saved {integrated_file.name}: {len(integrated['nodes'])} nodes, {len(integrated['edges'])} edges")
    
    # Save combined file for the explorer
    combined_file = output_dir / 'all_networks.json'
    with open(combined_file, 'w') as f:
        json.dump(all_networks, f)
    print(f"\nSaved combined data to {combined_file.name}")
    
    # Also generate region metadata
    metadata = {
        'integrated': {'name': 'Integrated Network', 'cells': 42654, 'description': 'Combined L-R interactions across all cardiac regions'},
        'AX': {'name': 'Apex', 'cells': 6497, 'description': 'Cardiac apex - tip of the heart'},
        'LA': {'name': 'Left Atrium', 'cells': 5822, 'description': 'Upper left chamber receiving oxygenated blood'},
        'LV': {'name': 'Left Ventricle', 'cells': 9626, 'description': 'Main pumping chamber to systemic circulation'},
        'RA': {'name': 'Right Atrium', 'cells': 7027, 'description': 'Upper right chamber receiving deoxygenated blood'},
        'RV': {'name': 'Right Ventricle', 'cells': 5039, 'description': 'Pumping chamber to pulmonary circulation'},
        'SP': {'name': 'Septum', 'cells': 8643, 'description': 'Muscular wall separating left and right chambers'}
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAll network data generated in {output_dir}")
    print("\nTo update the explorer, run:")
    print("  python scripts/update_explorer.py")

if __name__ == '__main__':
    main()
