"""
Graph Attention Network Layers for GRAIL-Heart

Custom GAT layers with edge type awareness and attention mechanisms
designed for spatial transcriptomics and L-R interaction modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops
from typing import Optional, Tuple, Union


class EdgeTypeGATConv(MessagePassing):
    """
    Graph Attention Convolution with Edge Type Awareness.
    
    Extends standard GAT to handle multiple edge types (spatial vs L-R),
    with separate attention mechanisms for each type.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads
        n_edge_types: Number of edge types
        concat: Whether to concatenate heads or average
        dropout: Attention dropout rate
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        n_edge_types: int = 2,
        concat: bool = True,
        dropout: float = 0.1,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.n_edge_types = n_edge_types
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        # Linear transformations for each head
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters per edge type
        self.att_src = nn.Parameter(torch.Tensor(n_edge_types, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(n_edge_types, heads, out_channels))
        
        # Edge type embedding
        self.edge_type_embed = nn.Embedding(n_edge_types, heads * out_channels)
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.edge_type_embed.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with edge-type aware attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge type indices [num_edges]
            edge_weight: Edge weights [num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            Updated node features [num_nodes, out_channels * heads]
            Optionally attention weights
        """
        H, C = self.heads, self.out_channels
        
        # Default edge types to 0 (spatial)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
            
        # Linear transformations
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)
        
        # Add self loops
        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, edge_type = self._add_self_loops_with_type(
                edge_index, edge_type, num_nodes
            )
            if edge_weight is not None:
                edge_weight = torch.cat([
                    edge_weight,
                    torch.ones(num_nodes, device=edge_weight.device)
                ])
                
        # Message passing
        out = self.propagate(
            edge_index, 
            x=(x_src, x_dst),
            edge_type=edge_type,
            edge_weight=edge_weight,
            size=None,
        )
        
        # Reshape output
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
            
        if self.bias is not None:
            out = out + self.bias
            
        if return_attention:
            return out, self._alpha
        return out
    
    def _add_self_loops_with_type(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add self-loops with edge type 0."""
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_type = torch.zeros(num_nodes, dtype=torch.long, device=edge_type.device)
        
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_type = torch.cat([edge_type, loop_type])
        
        return edge_index, edge_type
    
    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """Compute messages with edge-type aware attention."""
        
        # Get attention parameters for each edge's type
        att_src = self.att_src[edge_type]  # [E, H, C]
        att_dst = self.att_dst[edge_type]  # [E, H, C]
        
        # Compute attention scores
        alpha = (x_j * att_src).sum(dim=-1) + (x_i * att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight by edge weight if provided
        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)
            
        # Weighted message
        return x_j * alpha.unsqueeze(-1)


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer with residual connections.
    
    Wraps EdgeTypeGATConv with layer normalization, feedforward
    network, and residual connections following transformer architecture.
    
    Args:
        in_channels: Input dimension
        out_channels: Output dimension per head
        heads: Number of attention heads
        n_edge_types: Number of edge types
        dropout: Dropout rate
        concat: Whether to concatenate heads
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        n_edge_types: int = 2,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        
        self.concat = concat
        hidden_dim = out_channels * heads if concat else out_channels
        
        # Attention layer
        self.gat = EdgeTypeGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            n_edge_types=n_edge_types,
            concat=concat,
            dropout=dropout,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Projection for residual if dimensions don't match
        self.residual_proj = None
        if in_channels != hidden_dim:
            self.residual_proj = nn.Linear(in_channels, hidden_dim)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices
            edge_type: Edge type indices
            edge_weight: Edge weights
            
        Returns:
            Updated node features
        """
        # Project residual if needed
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
            
        # Attention + residual
        h = self.gat(x, edge_index, edge_type, edge_weight)
        h = self.norm1(h + residual)
        
        # FFN + residual
        h = self.norm2(h + self.ffn(h))
        
        return h


class GATStack(nn.Module):
    """
    Stacked GAT layers for deep graph learning.
    
    Args:
        in_channels: Input dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GAT layers
        heads: Number of attention heads
        n_edge_types: Number of edge types
        dropout: Dropout rate
        jk: Jumping knowledge mode ('cat', 'max', 'lstm', None)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 8,
        n_edge_types: int = 2,
        dropout: float = 0.1,
        jk: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.jk = jk
        
        # First layer
        self.layers = nn.ModuleList([
            MultiHeadGATLayer(
                in_channels=in_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                n_edge_types=n_edge_types,
                dropout=dropout,
                concat=True,
            )
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                MultiHeadGATLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    n_edge_types=n_edge_types,
                    dropout=dropout,
                    concat=True,
                )
            )
            
        # Final layer
        self.layers.append(
            MultiHeadGATLayer(
                in_channels=hidden_channels,
                out_channels=out_channels // heads if jk else out_channels,
                heads=heads if jk else 1,
                n_edge_types=n_edge_types,
                dropout=dropout,
                concat=jk is not None,
            )
        )
        
        # Jumping knowledge aggregation
        if jk == 'cat':
            # Concatenate all layer outputs
            jk_dim = hidden_channels * (num_layers - 1) + (out_channels if jk else out_channels * heads)
            self.jk_proj = nn.Linear(jk_dim, out_channels)
        elif jk == 'max':
            self.jk_proj = None
        elif jk == 'lstm':
            self.jk_lstm = nn.LSTM(
                hidden_channels, out_channels, 
                batch_first=True, bidirectional=True
            )
            self.jk_proj = nn.Linear(out_channels * 2, out_channels)
        else:
            self.jk_proj = None
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through stacked GAT layers.
        
        Args:
            x: Input node features
            edge_index: Edge indices
            edge_type: Edge type indices
            edge_weight: Edge weights
            
        Returns:
            Output node features
        """
        xs = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type, edge_weight)
            if self.jk:
                xs.append(x)
                
        if self.jk == 'cat':
            x = torch.cat(xs, dim=-1)
            x = self.jk_proj(x)
        elif self.jk == 'max':
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.jk == 'lstm':
            x = torch.stack(xs, dim=1)  # [N, L, D]
            x, _ = self.jk_lstm(x)
            x = self.jk_proj(x[:, -1, :])
            
        return x
