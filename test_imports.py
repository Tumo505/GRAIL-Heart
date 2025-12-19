"""Quick test of GRAIL-Heart imports and model instantiation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing imports...")

# Test data imports
from grail_heart.data import (
    SpatialTranscriptomicsDataset, 
    SpatialGraphBuilder, 
    LigandReceptorDatabase
)
print("  Data module: OK")

# Test model imports
from grail_heart.models import (
    GRAILHeart, 
    create_grail_heart,
    GeneEncoder,
    GATStack,
)
print("  Models module: OK")

# Test training imports
from grail_heart.training import (
    GRAILHeartTrainer, 
    GRAILHeartLoss,
    create_optimizer,
)
print("  Training module: OK")

# Test PyTorch
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Test model creation
print("\nCreating test model...")
model = create_grail_heart(
    n_genes=2000,
    config={
        'hidden_dim': 128,
        'n_gat_layers': 2,
        'n_heads': 4,
    }
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Test forward pass
print("\nTesting forward pass...")
from torch_geometric.data import Data

# Create dummy graph data
n_cells = 100
n_genes = 2000
n_edges = 500

x = torch.randn(n_cells, n_genes)
edge_index = torch.randint(0, n_cells, (2, n_edges))
pos = torch.rand(n_cells, 2)

data = Data(x=x, edge_index=edge_index, pos=pos)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

with torch.no_grad():
    outputs = model(data)
    
print(f"Output keys: {list(outputs.keys())}")
if 'lr_scores' in outputs:
    print(f"LR scores shape: {outputs['lr_scores'].shape}")
if 'reconstruction' in outputs:
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")

print("\n=== All tests passed! ===")
