---
title: GRAIL-Heart
emoji: "\U0001FAC0"
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - spatial-transcriptomics
  - graph-neural-networks
  - cardiac-biology
  - cell-communication
  - inverse-modelling
  - causal-inference
---

# GRAIL-Heart: Inverse Modelling of Cardiac Cell-Cell Communication

Interactive web application for exploring causal ligand-receptor (L-R)
interaction analysis in cardiac spatial transcriptomics using the
GRAIL-Heart graph neural network.

## Features

- **Atlas Explorer** — Browse pre-computed results across 6 Heart Cell Atlas v2 regions
- **Upload & Analyse** — Upload your own spatial transcriptomics data (.h5ad, .h5, .csv)
- **Forward Model** — Compute expression-based L-R interaction scores
- **Inverse / Causal** — Run the full GNN inverse inference pipeline
- **Results** — View, compare, and download all results

## Demo Data

Click **Load Demo Data** in the Upload tab to load a 500-cell subsample from the
left ventricle (LV) of the Heart Cell Atlas v2, pre-processed and ready for
both forward and inverse analysis.

## Links

- **Preprint:** [SSRN 6137179](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6137179)
- **Code:** [github.com/Tumo505/GRAIL-Heart](https://github.com/Tumo505/GRAIL-Heart)
- **Network Explorer:** [Interactive Cytoscape](https://tumo505.github.io/GRAIL-Heart/outputs/cytoscape/index.html)

## Citation

If you use GRAIL-Heart in your research, please cite:

```
Tumo Kgabeng, Lulu Wang, Harry Ngwangwa, Thanyani Pandelani, "GRAIL-Heart: A Graph Attention Network for Inferring Ligand-Receptor Interactions in Spatial Tran-scriptomics," 2026. Available: https://papers.ssrn.com/abstract=6137179. DOI: 10.2139/ssrn.6137179.
```
