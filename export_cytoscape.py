#!/usr/bin/env python3
"""
Export GRAIL-Heart network data to Cytoscape-compatible formats.
Generates GraphML, edge lists, and node attribute files for interactive visualization.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


def create_graphml(nodes_df, edges_df, output_path, network_name="GRAIL-Heart"):
    """Create GraphML file for Cytoscape import."""
    
    # Create root element with namespaces
    graphml = ET.Element('graphml')
    graphml.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
    graphml.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    graphml.set('xsi:schemaLocation', 
                'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd')
    
    # Define node attributes
    node_attrs = [
        ('node_type', 'string', 'unknown'),
        ('gene_name', 'string', ''),
        ('expression', 'double', '0.0'),
        ('degree', 'int', '0'),
        ('region', 'string', ''),
        ('is_ligand', 'boolean', 'false'),
        ('is_receptor', 'boolean', 'false'),
    ]
    
    for attr_name, attr_type, default in node_attrs:
        key = ET.SubElement(graphml, 'key')
        key.set('id', attr_name)
        key.set('for', 'node')
        key.set('attr.name', attr_name)
        key.set('attr.type', attr_type)
        default_elem = ET.SubElement(key, 'default')
        default_elem.text = default
    
    # Define edge attributes
    edge_attrs = [
        ('interaction_type', 'string', 'L-R'),
        ('causal_score', 'double', '0.0'),
        ('ligand', 'string', ''),
        ('receptor', 'string', ''),
        ('pathway', 'string', ''),
        ('function', 'string', ''),
        ('weight', 'double', '1.0'),
    ]
    
    for attr_name, attr_type, default in edge_attrs:
        key = ET.SubElement(graphml, 'key')
        key.set('id', attr_name)
        key.set('for', 'edge')
        key.set('attr.name', attr_name)
        key.set('attr.type', attr_type)
        default_elem = ET.SubElement(key, 'default')
        default_elem.text = default
    
    # Create graph
    graph = ET.SubElement(graphml, 'graph')
    graph.set('id', network_name)
    graph.set('edgedefault', 'directed')
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        node = ET.SubElement(graph, 'node')
        node.set('id', str(row['id']))
        
        for col in nodes_df.columns:
            if col != 'id':
                data = ET.SubElement(node, 'data')
                data.set('key', col)
                data.text = str(row[col])
    
    # Add edges
    for idx, row in edges_df.iterrows():
        edge = ET.SubElement(graph, 'edge')
        edge.set('id', f"e{idx}")
        edge.set('source', str(row['source']))
        edge.set('target', str(row['target']))
        
        for col in edges_df.columns:
            if col not in ['source', 'target']:
                data = ET.SubElement(edge, 'data')
                data.set('key', col)
                data.text = str(row[col])
    
    # Pretty print
    xml_str = minidom.parseString(ET.tostring(graphml)).toprettyxml(indent="  ")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    return output_path


def create_cytoscape_json(nodes_df, edges_df, output_path, network_name="GRAIL-Heart"):
    """Create Cytoscape.js compatible JSON format."""
    
    elements = {
        "nodes": [],
        "edges": []
    }
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        node_data = {"id": str(row['id'])}
        for col in nodes_df.columns:
            if col != 'id':
                node_data[col] = row[col] if pd.notna(row[col]) else None
        elements["nodes"].append({"data": node_data})
    
    # Add edges
    for idx, row in edges_df.iterrows():
        edge_data = {
            "id": f"e{idx}",
            "source": str(row['source']),
            "target": str(row['target'])
        }
        for col in edges_df.columns:
            if col not in ['source', 'target']:
                edge_data[col] = row[col] if pd.notna(row[col]) else None
        elements["edges"].append({"data": edge_data})
    
    cytoscape_data = {
        "format_version": "1.0",
        "generated_by": "GRAIL-Heart",
        "target_cytoscapejs_version": "~3.0",
        "data": {
            "name": network_name,
            "shared_name": network_name
        },
        "elements": elements
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cytoscape_data, f, indent=2)
    
    return output_path


def create_sif_format(edges_df, output_path):
    """Create Simple Interaction Format (SIF) for Cytoscape."""
    
    with open(output_path, 'w') as f:
        for _, row in edges_df.iterrows():
            interaction_type = row.get('interaction_type', 'interacts')
            f.write(f"{row['source']}\t{interaction_type}\t{row['target']}\n")
    
    return output_path


def create_node_attributes(nodes_df, output_path):
    """Create node attribute file for Cytoscape."""
    
    nodes_df.to_csv(output_path, sep='\t', index=False)
    return output_path


def create_edge_attributes(edges_df, output_path):
    """Create edge attribute file for Cytoscape."""
    
    # Create edge name column
    edges_df = edges_df.copy()
    edges_df['edge_name'] = edges_df['source'] + ' (interacts) ' + edges_df['target']
    
    edges_df.to_csv(output_path, sep='\t', index=False)
    return output_path


def load_causal_data(causal_dir):
    """Load causal analysis data from GRAIL-Heart output."""
    
    # Causal edges files only have edge_idx and causal_score
    # We need to use L-R scores instead which have ligand/receptor info
    return pd.DataFrame(), []


def load_lr_scores(tables_dir):
    """Load L-R scores from GRAIL-Heart tables."""
    
    all_scores = []
    
    for csv_file in Path(tables_dir).glob('*_lr_scores.csv'):
        region = csv_file.stem.replace('_lr_scores', '')
        df = pd.read_csv(csv_file)
        df['region'] = region
        all_scores.append(df)
    
    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
    else:
        scores_df = pd.DataFrame()
    
    return scores_df


def build_network_from_edges(edges_df, score_threshold=0.5):
    """Build node and edge DataFrames for network visualization."""
    
    # Filter by score threshold
    score_col = None
    for col in ['causal_score', 'mean_score', 'score']:
        if col in edges_df.columns:
            score_col = col
            break
    
    if score_col:
        edges_df = edges_df[edges_df[score_col] >= score_threshold].copy()
    
    # Extract unique nodes (ligands and receptors)
    ligands = set()
    receptors = set()
    
    # Determine column names
    ligand_col = None
    receptor_col = None
    
    for col in ['ligand', 'source', 'gene_a']:
        if col in edges_df.columns:
            ligand_col = col
            break
    
    for col in ['receptor', 'target', 'gene_b']:
        if col in edges_df.columns:
            receptor_col = col
            break
    
    if ligand_col is None or receptor_col is None:
        print(f"    Warning: Could not find ligand/receptor columns. Available: {list(edges_df.columns)}")
        return pd.DataFrame(), pd.DataFrame()
    
    for _, row in edges_df.iterrows():
        ligands.add(row[ligand_col])
        receptors.add(row[receptor_col])
    
    # Create nodes DataFrame
    nodes = []
    all_genes = ligands | receptors
    
    for gene in all_genes:
        is_ligand = gene in ligands
        is_receptor = gene in receptors
        
        if is_ligand and is_receptor:
            node_type = 'both'
        elif is_ligand:
            node_type = 'ligand'
        else:
            node_type = 'receptor'
        
        # Calculate degree
        out_degree = len(edges_df[edges_df[ligand_col] == gene])
        in_degree = len(edges_df[edges_df[receptor_col] == gene])
        
        nodes.append({
            'id': gene,
            'gene_name': gene,
            'node_type': node_type,
            'is_ligand': is_ligand,
            'is_receptor': is_receptor,
            'degree': out_degree + in_degree,
            'out_degree': out_degree,
            'in_degree': in_degree
        })
    
    nodes_df = pd.DataFrame(nodes)
    
    # Create edges DataFrame
    edge_records = []
    for _, row in edges_df.iterrows():
        edge_record = {
            'source': row[ligand_col],
            'target': row[receptor_col],
            'interaction_type': 'L-R',
        }
        
        # Add available attributes
        if 'causal_score' in row:
            edge_record['causal_score'] = row['causal_score']
            edge_record['weight'] = row['causal_score']
        elif 'score' in row:
            edge_record['causal_score'] = row['score']
            edge_record['weight'] = row['score']
        
        if 'pathway' in row:
            edge_record['pathway'] = row['pathway']
        if 'function' in row:
            edge_record['function'] = row['function']
        if 'region' in row:
            edge_record['region'] = row['region']
        
        edge_record['ligand'] = row[ligand_col]
        edge_record['receptor'] = row[receptor_col]
        
        edge_records.append(edge_record)
    
    edges_out_df = pd.DataFrame(edge_records)
    
    return nodes_df, edges_out_df


def create_html_visualization(nodes_df, edges_df, output_path, title="GRAIL-Heart Network"):
    """Create interactive HTML visualization using Cytoscape.js."""
    
    # Prepare data for JavaScript
    elements = []
    
    for _, row in nodes_df.iterrows():
        node_data = {
            "id": str(row['id']),
            "label": str(row['gene_name']),
            "type": row['node_type'],
            "degree": int(row['degree'])
        }
        elements.append({"data": node_data, "group": "nodes"})
    
    for idx, row in edges_df.iterrows():
        edge_data = {
            "id": f"e{idx}",
            "source": str(row['source']),
            "target": str(row['target']),
            "weight": float(row.get('weight', 1.0)),
            "causal_score": float(row.get('causal_score', 0.0))
        }
        elements.append({"data": edge_data, "group": "edges"})
    
    html_template = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #1a1a2e;
            color: #eee;
        }}
        #header {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #e94560;
        }}
        h1 {{
            margin: 0;
            color: #e94560;
            font-size: 2em;
        }}
        .subtitle {{
            color: #aaa;
            margin-top: 5px;
        }}
        #cy {{
            width: 100%;
            height: calc(100vh - 180px);
            background: #0f0f23;
        }}
        #controls {{
            padding: 15px 20px;
            background: #16213e;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        label {{
            color: #aaa;
            font-size: 0.9em;
        }}
        select, input[type="range"] {{
            background: #1a1a2e;
            border: 1px solid #e94560;
            color: #eee;
            padding: 5px 10px;
            border-radius: 4px;
        }}
        button {{
            background: #e94560;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }}
        button:hover {{
            background: #ff6b6b;
        }}
        #info {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(22, 33, 62, 0.95);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e94560;
            max-width: 300px;
            display: none;
        }}
        #info h3 {{
            margin: 0 0 10px 0;
            color: #e94560;
        }}
        #info p {{
            margin: 5px 0;
            font-size: 0.9em;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-left: auto;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.85em;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .stats {{
            color: #888;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1> {title}</h1>
        <div class="subtitle">Interactive Ligand-Receptor Network Visualization</div>
    </div>
    
    <div id="controls">
        <div class="control-group">
            <label>Layout:</label>
            <select id="layout-select">
                <option value="cose">CoSE (Force-directed)</option>
                <option value="circle">Circle</option>
                <option value="concentric">Concentric</option>
                <option value="breadthfirst">Hierarchical</option>
                <option value="grid">Grid</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>Min Score:</label>
            <input type="range" id="score-filter" min="0" max="1" step="0.1" value="0">
            <span id="score-value">0.0</span>
        </div>
        
        <button onclick="cy.fit()">Fit View</button>
        <button onclick="exportPNG()">Export PNG</button>
        
        <div class="stats" id="stats"></div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #e94560;"></div>
                <span>Ligand</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #4ecdc4;"></div>
                <span>Receptor</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ffe66d;"></div>
                <span>Both</span>
            </div>
        </div>
    </div>
    
    <div id="cy"></div>
    
    <div id="info">
        <h3 id="info-title"></h3>
        <div id="info-content"></div>
    </div>
    
    <script>
        const elements = {json.dumps(elements)};
        
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'color': '#fff',
                        'text-outline-color': '#000',
                        'text-outline-width': 1,
                        'width': 'mapData(degree, 1, 50, 20, 60)',
                        'height': 'mapData(degree, 1, 50, 20, 60)',
                        'background-color': function(ele) {{
                            const type = ele.data('type');
                            if (type === 'ligand') return '#e94560';
                            if (type === 'receptor') return '#4ecdc4';
                            return '#ffe66d';
                        }}
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 'mapData(weight, 0, 2, 1, 5)',
                        'line-color': '#555',
                        'target-arrow-color': '#555',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.7
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 3,
                        'border-color': '#fff'
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                        'line-color': '#e94560',
                        'target-arrow-color': '#e94560',
                        'width': 4
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: false,
                nodeRepulsion: 8000,
                idealEdgeLength: 100
            }}
        }});
        
        // Update stats
        function updateStats() {{
            const visibleNodes = cy.nodes(':visible').length;
            const visibleEdges = cy.edges(':visible').length;
            document.getElementById('stats').textContent = 
                `${{visibleNodes}} nodes, ${{visibleEdges}} edges`;
        }}
        updateStats();
        
        // Layout change
        document.getElementById('layout-select').addEventListener('change', function(e) {{
            cy.layout({{ name: e.target.value, animate: true }}).run();
        }});
        
        // Score filter
        document.getElementById('score-filter').addEventListener('input', function(e) {{
            const threshold = parseFloat(e.target.value);
            document.getElementById('score-value').textContent = threshold.toFixed(1);
            
            cy.edges().forEach(edge => {{
                const score = edge.data('causal_score') || 0;
                if (score >= threshold) {{
                    edge.show();
                }} else {{
                    edge.hide();
                }}
            }});
            
            // Hide orphan nodes
            cy.nodes().forEach(node => {{
                const connectedEdges = node.connectedEdges(':visible');
                if (connectedEdges.length === 0) {{
                    node.hide();
                }} else {{
                    node.show();
                }}
            }});
            
            updateStats();
        }});
        
        // Node click info
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const info = document.getElementById('info');
            document.getElementById('info-title').textContent = node.data('label');
            document.getElementById('info-content').innerHTML = `
                <p><strong>Type:</strong> ${{node.data('type')}}</p>
                <p><strong>Connections:</strong> ${{node.data('degree')}}</p>
                <p><strong>Outgoing:</strong> ${{node.outgoers('edge').length}}</p>
                <p><strong>Incoming:</strong> ${{node.incomers('edge').length}}</p>
            `;
            info.style.display = 'block';
        }});
        
        // Edge click info
        cy.on('tap', 'edge', function(evt) {{
            const edge = evt.target;
            const info = document.getElementById('info');
            document.getElementById('info-title').textContent = 
                `${{edge.data('source')}} → ${{edge.data('target')}}`;
            document.getElementById('info-content').innerHTML = `
                <p><strong>Causal Score:</strong> ${{(edge.data('causal_score') || 0).toFixed(3)}}</p>
                <p><strong>Type:</strong> L-R Interaction</p>
            `;
            info.style.display = 'block';
        }});
        
        // Click background to hide info
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                document.getElementById('info').style.display = 'none';
            }}
        }});
        
        // Export PNG
        function exportPNG() {{
            const png = cy.png({{ full: true, scale: 2 }});
            const link = document.createElement('a');
            link.href = png;
            link.download = 'grail_heart_network.png';
            link.click();
        }}
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export GRAIL-Heart networks to Cytoscape')
    parser.add_argument('--input_dir', type=str, default='outputs/enhanced_analysis',
                        help='Input directory with GRAIL-Heart analysis results')
    parser.add_argument('--output_dir', type=str, default='outputs/cytoscape',
                        help='Output directory for Cytoscape files')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Minimum causal score threshold for edges')
    parser.add_argument('--format', type=str, nargs='+', 
                        default=['graphml', 'json', 'sif', 'html'],
                        choices=['graphml', 'json', 'sif', 'html', 'all'],
                        help='Output formats')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GRAIL-Heart Cytoscape Export")
    print("=" * 60)
    
    # Load data
    print("\nLoading L-R scores data...")
    causal_dir = input_dir / 'causal_analysis'
    tables_dir = input_dir / 'tables'
    
    # Load L-R scores directly (they have ligand/receptor info)
    edges_df = load_lr_scores(tables_dir)
    if not edges_df.empty:
        regions = edges_df['region'].unique().tolist() if 'region' in edges_df.columns else []
        print(f"  Loaded {len(edges_df)} L-R pairs from {len(regions)} regions")
    else:
        edges_df = pd.DataFrame()
        regions = []
    
    if edges_df.empty:
        print("ERROR: No network data found!")
        return
    
    # Load cross-region comparison for integrated network
    cross_region_path = tables_dir / 'cross_region_comparison.csv'
    if cross_region_path.exists():
        cross_region_df = pd.read_csv(cross_region_path)
        print(f"  Loaded cross-region comparison with {len(cross_region_df)} L-R pairs")
    else:
        cross_region_df = None
    
    formats = args.format
    if 'all' in formats:
        formats = ['graphml', 'json', 'sif', 'html']
    
    # Export per-region networks
    print(f"\nExporting networks (threshold: {args.score_threshold})...")
    
    for region in regions:
        region_edges = edges_df[edges_df['region'] == region]
        if region_edges.empty:
            continue
        
        nodes_df, edges_out_df = build_network_from_edges(region_edges, args.score_threshold)
        print(f"\n  {region}: {len(nodes_df)} nodes, {len(edges_out_df)} edges")
        
        if nodes_df.empty:
            continue
        
        region_dir = output_dir / region
        region_dir.mkdir(exist_ok=True)
        
        if 'graphml' in formats:
            path = create_graphml(nodes_df, edges_out_df, 
                                  region_dir / f'{region}_network.graphml',
                                  f'GRAIL-Heart_{region}')
            print(f"    Created: {path.name}")
        
        if 'json' in formats:
            path = create_cytoscape_json(nodes_df, edges_out_df,
                                         region_dir / f'{region}_network.cyjs',
                                         f'GRAIL-Heart_{region}')
            print(f"    Created: {path.name}")
        
        if 'sif' in formats:
            path = create_sif_format(edges_out_df, region_dir / f'{region}_network.sif')
            print(f"    Created: {path.name}")
            
            # Also create attribute files
            create_node_attributes(nodes_df, region_dir / f'{region}_node_attributes.txt')
            create_edge_attributes(edges_out_df, region_dir / f'{region}_edge_attributes.txt')
        
        if 'html' in formats:
            path = create_html_visualization(nodes_df, edges_out_df,
                                             region_dir / f'{region}_interactive.html',
                                             f'GRAIL-Heart {region} Network')
            print(f"    Created: {path.name}")
    
    # Export integrated network (all regions combined)
    print("\n  Creating integrated network (all regions)...")
    
    if cross_region_df is not None:
        # Use cross-region data for integrated network
        integrated_edges = []
        for _, row in cross_region_df.iterrows():
            avg_score = row.get('mean_score', 0)
            if avg_score >= args.score_threshold:
                integrated_edges.append({
                    'source': row['ligand'],
                    'target': row['receptor'],
                    'ligand': row['ligand'],
                    'receptor': row['receptor'],
                    'causal_score': avg_score,
                    'pathway': row.get('pathway', ''),
                    'function': row.get('function', ''),
                    'region': 'integrated'
                })
        integrated_edges_df = pd.DataFrame(integrated_edges)
    else:
        integrated_edges_df = edges_df.copy()
    
    if not integrated_edges_df.empty:
        nodes_df, edges_out_df = build_network_from_edges(integrated_edges_df, args.score_threshold)
        print(f"  Integrated: {len(nodes_df)} nodes, {len(edges_out_df)} edges")
        
        if 'graphml' in formats:
            path = create_graphml(nodes_df, edges_out_df,
                                  output_dir / 'integrated_network.graphml',
                                  'GRAIL-Heart_Integrated')
            print(f"    Created: {path.name}")
        
        if 'json' in formats:
            path = create_cytoscape_json(nodes_df, edges_out_df,
                                         output_dir / 'integrated_network.cyjs',
                                         'GRAIL-Heart_Integrated')
            print(f"    Created: {path.name}")
        
        if 'sif' in formats:
            path = create_sif_format(edges_out_df, output_dir / 'integrated_network.sif')
            print(f"    Created: {path.name}")
            create_node_attributes(nodes_df, output_dir / 'integrated_node_attributes.txt')
            create_edge_attributes(edges_out_df, output_dir / 'integrated_edge_attributes.txt')
        
        if 'html' in formats:
            path = create_html_visualization(nodes_df, edges_out_df,
                                             output_dir / 'integrated_interactive.html',
                                             'GRAIL-Heart Integrated Cardiac Network')
            print(f"    Created: {path.name}")
    
    # Create README for Cytoscape import instructions
    readme_content = """# GRAIL-Heart Cytoscape Export

## Files Generated

### Per-Region Networks
Each cardiac region has its own folder with:
- `{region}_network.graphml` - GraphML format (recommended for Cytoscape Desktop)
- `{region}_network.cyjs` - Cytoscape.js JSON format
- `{region}_network.sif` - Simple Interaction Format
- `{region}_node_attributes.txt` - Node attributes (tab-delimited)
- `{region}_edge_attributes.txt` - Edge attributes (tab-delimited)
- `{region}_interactive.html` - Standalone interactive visualization

### Integrated Network
- `integrated_network.graphml` - Combined network from all regions
- `integrated_network.cyjs` - JSON format
- `integrated_network.sif` - SIF format
- `integrated_interactive.html` - Interactive HTML visualization

## Import Instructions

### Cytoscape Desktop (Recommended)

1. Open Cytoscape (https://cytoscape.org/)
2. File → Import → Network from File
3. Select the `.graphml` file
4. The network will be loaded with all attributes

### Import SIF with Attributes

1. File → Import → Network from File → Select `.sif` file
2. File → Import → Table from File → Select `_node_attributes.txt`
   - Import Data As: Node Table Columns
   - Key Column: id
3. File → Import → Table from File → Select `_edge_attributes.txt`
   - Import Data As: Edge Table Columns

### Interactive HTML

Simply open the `.html` files in a web browser for immediate visualization.
No installation required!

## Network Attributes

### Node Attributes
- `gene_name`: Gene symbol
- `node_type`: 'ligand', 'receptor', or 'both'
- `degree`: Total connections
- `out_degree`: Outgoing edges (ligand role)
- `in_degree`: Incoming edges (receptor role)

### Edge Attributes
- `causal_score`: GRAIL-Heart causal inference score (0-2)
- `pathway`: Signaling pathway category
- `function`: Biological function
- `region`: Cardiac region (for per-region networks)

## Visualization Tips

1. **Size nodes by degree**: Visual Mapping → Node Size → degree
2. **Color nodes by type**: Visual Mapping → Node Fill Color → node_type
3. **Edge width by score**: Visual Mapping → Edge Width → causal_score
4. **Layout**: Layout → Prefuse Force Directed or yFiles Organic

## Citation

If you use these networks, please cite:
GRAIL-Heart: Graph-based Reasoning for Analyzing Intercellular 
Ligand-receptor interactions in Heart tissue

Generated: """ + str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
    
    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Open the integrated HTML in browser
    html_path = output_dir / 'integrated_interactive.html'
    if html_path.exists():
        print(f"\nOpen interactive visualization: {html_path}")


if __name__ == '__main__':
    main()
