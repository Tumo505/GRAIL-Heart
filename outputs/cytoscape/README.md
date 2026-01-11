# GRAIL-Heart Cytoscape Export

## 🌐 Live Interactive Explorer

**[Open GRAIL-Heart Network Explorer](https://tumo505.github.io/GRAIL-Heart/outputs/cytoscape/index.html)**

The unified interactive explorer allows you to:
- Switch between all 6 cardiac regions + integrated network
- Search and filter genes by name
- Adjust visualization parameters
- Export networks as SVG/PNG
- View detailed node/edge information on hover

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
