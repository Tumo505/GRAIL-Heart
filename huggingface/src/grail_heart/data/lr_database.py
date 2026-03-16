"""
Ligand-Receptor Database Integration

Interfaces with CellPhoneDB and other L-R databases to provide
ligand-receptor interaction pairs for graph construction and
signaling network inference.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings


class LigandReceptorDatabase:
    """
    Interface for ligand-receptor interaction databases.
    
    Supports:
    - CellPhoneDB v5
    - Custom L-R pair files
    - Built-in cardiac-specific interactions
    
    Args:
        source: Database source ('cellphonedb', 'custom', 'builtin')
        custom_path: Path to custom L-R pairs file
        species: Species for database ('human', 'mouse')
    """
    
    def __init__(
        self,
        source: str = 'cellphonedb',
        custom_path: Optional[Path] = None,
        species: str = 'human',
    ):
        self.source = source
        self.custom_path = custom_path
        self.species = species
        
        self.lr_pairs = None
        self.gene_info = None
        
        self._load_database()
        
    def _load_database(self):
        """Load L-R database based on source."""
        
        if self.source == 'cellphonedb':
            self._load_cellphonedb()
        elif self.source == 'custom':
            self._load_custom()
        elif self.source == 'builtin':
            self._load_builtin()
        else:
            raise ValueError(f"Unknown source: {self.source}")
            
    def _load_cellphonedb(self):
        """Load CellPhoneDB database."""
        try:
            # Try to import CellPhoneDB
            from cellphonedb.utils import db_utils
            
            # Get database files
            cpdb_file_path = db_utils.get_default_database()
            
            # Load interaction data
            interactions = pd.read_csv(
                Path(cpdb_file_path) / 'interaction_input.csv'
            )
            
            # Load gene info
            genes = pd.read_csv(
                Path(cpdb_file_path) / 'gene_input.csv'
            )
            
            self.gene_info = genes
            
            # Process interactions to get L-R pairs
            self.lr_pairs = self._process_cellphonedb_interactions(interactions)
            
        except ImportError:
            warnings.warn("CellPhoneDB not available, falling back to builtin database")
            self._load_builtin()
        except Exception as e:
            warnings.warn(f"Failed to load CellPhoneDB: {e}, falling back to builtin database")
            self._load_builtin()
            
    def _process_cellphonedb_interactions(
        self, 
        interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Process CellPhoneDB interactions to L-R pairs format."""
        
        # CellPhoneDB has partner_a and partner_b columns
        # These can be ligand-receptor or receptor-ligand
        pairs = []
        
        for _, row in interactions.iterrows():
            partner_a = str(row.get('partner_a', '')).strip()
            partner_b = str(row.get('partner_b', '')).strip()
            
            if not partner_a or not partner_b:
                continue
                
            # Handle complex proteins (e.g., "TGFB1_TGFBR1_TGFBR2")
            if '_' in partner_a:
                partner_a_genes = partner_a.split('_')
            else:
                partner_a_genes = [partner_a]
                
            if '_' in partner_b:
                partner_b_genes = partner_b.split('_')
            else:
                partner_b_genes = [partner_b]
                
            # Create pairs for all combinations
            for a in partner_a_genes:
                for b in partner_b_genes:
                    pairs.append({
                        'ligand': a,
                        'receptor': b,
                        'source': 'cellphonedb',
                        'annotation': row.get('annotation_strategy', 'unknown')
                    })
                    
        return pd.DataFrame(pairs).drop_duplicates()
    
    def _load_custom(self):
        """Load custom L-R pairs from file."""
        if self.custom_path is None:
            raise ValueError("custom_path required for custom source")
            
        path = Path(self.custom_path)
        
        if path.suffix == '.csv':
            self.lr_pairs = pd.read_csv(path)
        elif path.suffix in ['.xlsx', '.xls']:
            self.lr_pairs = pd.read_excel(path)
        elif path.suffix == '.tsv':
            self.lr_pairs = pd.read_csv(path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        # Ensure required columns
        required = {'ligand', 'receptor'}
        if not required.issubset(self.lr_pairs.columns):
            raise ValueError(f"Custom file must have columns: {required}")
    
    def _load_builtin(self):
        """Load built-in cardiac L-R database."""
        
        # Curated list of cardiac-relevant L-R interactions
        cardiac_lr_pairs = [
            # Growth factors and their receptors
            ('VEGFA', 'FLT1'), ('VEGFA', 'KDR'), ('VEGFB', 'FLT1'),
            ('PDGFA', 'PDGFRA'), ('PDGFB', 'PDGFRB'), ('PDGFC', 'PDGFRA'),
            ('FGF1', 'FGFR1'), ('FGF2', 'FGFR1'), ('FGF2', 'FGFR2'),
            ('EGF', 'EGFR'), ('HBEGF', 'EGFR'), ('NRG1', 'ERBB2'), ('NRG1', 'ERBB4'),
            ('IGF1', 'IGF1R'), ('IGF2', 'IGF1R'), ('IGF2', 'IGF2R'),
            ('BMP2', 'BMPR1A'), ('BMP4', 'BMPR1A'), ('BMP10', 'BMPR2'),
            
            # TGF-beta signaling
            ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2'), ('TGFB2', 'TGFBR1'),
            ('TGFB3', 'TGFBR1'), ('GDF15', 'TGFBR2'),
            
            # Wnt signaling
            ('WNT1', 'FZD1'), ('WNT3A', 'FZD2'), ('WNT5A', 'FZD5'), ('WNT11', 'FZD7'),
            ('WNT2', 'LRP5'), ('WNT3', 'LRP6'),
            
            # Notch signaling
            ('DLL1', 'NOTCH1'), ('DLL4', 'NOTCH1'), ('JAG1', 'NOTCH1'),
            ('JAG1', 'NOTCH2'), ('JAG2', 'NOTCH1'), ('JAG2', 'NOTCH3'),
            
            # Cytokines
            ('IL1B', 'IL1R1'), ('IL6', 'IL6R'), ('TNF', 'TNFRSF1A'),
            ('CXCL12', 'CXCR4'), ('CCL2', 'CCR2'), ('CCL5', 'CCR5'),
            
            # ECM interactions
            ('COL1A1', 'ITGA1'), ('COL1A2', 'ITGB1'), ('FN1', 'ITGA5'),
            ('LAMA2', 'ITGA7'), ('LAMB1', 'ITGB1'), ('THBS1', 'CD47'),
            
            # Cardiac-specific
            ('NPPA', 'NPR1'), ('NPPB', 'NPR1'), ('EDN1', 'EDNRA'),
            ('AGT', 'AGTR1'), ('ANGPT1', 'TEK'), ('ANGPT2', 'TEK'),
            ('SEMA3A', 'NRP1'), ('EFNA1', 'EPHA2'), ('EFNB2', 'EPHB4'),
            
            # Nervous system / conduction
            ('BDNF', 'NTRK2'), ('NGF', 'NTRK1'), ('NTF3', 'NTRK3'),
            ('GDNF', 'RET'), ('CNTF', 'CNTFR'),
            
            # Gap junctions and adhesion (paracrine/juxtacrine)
            ('CDH2', 'CDH2'), ('CDH5', 'CDH5'),  # homophilic
        ]
        
        self.lr_pairs = pd.DataFrame(cardiac_lr_pairs, columns=['ligand', 'receptor'])
        self.lr_pairs['source'] = 'builtin_cardiac'
        self.lr_pairs['annotation'] = 'cardiac'
        
        print(f"Loaded {len(self.lr_pairs)} built-in cardiac L-R pairs")
        
    def get_pairs(self, genes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get L-R pairs, optionally filtered to specific genes.
        
        Args:
            genes: Optional list of genes to filter to
            
        Returns:
            DataFrame with ligand, receptor columns
        """
        pairs = self.lr_pairs.copy()
        
        if genes is not None:
            gene_set = set(genes)
            mask = pairs['ligand'].isin(gene_set) | pairs['receptor'].isin(gene_set)
            pairs = pairs[mask]
            
        return pairs
    
    def get_ligands(self) -> Set[str]:
        """Get all ligand genes."""
        return set(self.lr_pairs['ligand'].unique())
    
    def get_receptors(self) -> Set[str]:
        """Get all receptor genes."""
        return set(self.lr_pairs['receptor'].unique())
    
    def get_all_genes(self) -> Set[str]:
        """Get all genes involved in L-R interactions."""
        return self.get_ligands() | self.get_receptors()
    
    def get_partners(self, gene: str) -> Dict[str, List[str]]:
        """
        Get interaction partners for a gene.
        
        Args:
            gene: Gene name
            
        Returns:
            Dictionary with 'as_ligand' and 'as_receptor' partner lists
        """
        partners = {
            'as_ligand': [],
            'as_receptor': []
        }
        
        # Gene as ligand -> find receptors
        mask = self.lr_pairs['ligand'] == gene
        partners['as_ligand'] = self.lr_pairs[mask]['receptor'].tolist()
        
        # Gene as receptor -> find ligands
        mask = self.lr_pairs['receptor'] == gene
        partners['as_receptor'] = self.lr_pairs[mask]['ligand'].tolist()
        
        return partners
    
    def filter_by_expression(
        self,
        expression: pd.DataFrame,
        min_pct: float = 0.1,
    ) -> pd.DataFrame:
        """
        Filter L-R pairs to those expressed in the dataset.
        
        Args:
            expression: Expression matrix (cells x genes)
            min_pct: Minimum percentage of cells expressing gene
            
        Returns:
            Filtered L-R pairs
        """
        # Find expressed genes
        n_cells = expression.shape[0]
        expressed = (expression > 0).sum(axis=0) / n_cells >= min_pct
        expressed_genes = set(expression.columns[expressed])
        
        # Filter pairs
        mask = (
            self.lr_pairs['ligand'].isin(expressed_genes) &
            self.lr_pairs['receptor'].isin(expressed_genes)
        )
        
        filtered = self.lr_pairs[mask]
        print(f"Filtered to {len(filtered)} expressed L-R pairs")
        
        return filtered
    
    def to_edge_dict(self, gene_to_idx: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Convert L-R pairs to edge indices.
        
        Args:
            gene_to_idx: Mapping from gene names to indices
            
        Returns:
            Dictionary with edge indices and attributes
        """
        import torch
        
        edges = []
        
        for _, row in self.lr_pairs.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            
            if ligand in gene_to_idx and receptor in gene_to_idx:
                edges.append((gene_to_idx[ligand], gene_to_idx[receptor]))
                
        if not edges:
            return {
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'n_edges': 0
            }
            
        edge_index = torch.tensor(edges, dtype=torch.long).T
        
        return {
            'edge_index': edge_index,
            'n_edges': edge_index.shape[1]
        }


class CellPhoneDBRunner:
    """
    Wrapper for running CellPhoneDB analysis.
    
    Integrates CellPhoneDB v5 for cell-cell communication analysis
    to identify significant L-R interactions in the dataset.
    """
    
    def __init__(
        self,
        cpdb_version: str = 'v5',
        n_jobs: int = 4,
    ):
        self.cpdb_version = cpdb_version
        self.n_jobs = n_jobs
        
    def run_analysis(
        self,
        counts: pd.DataFrame,
        meta: pd.DataFrame,
        output_dir: Path,
        iterations: int = 1000,
        threshold: float = 0.1,
    ) -> pd.DataFrame:
        """
        Run CellPhoneDB statistical analysis.
        
        Args:
            counts: Gene expression counts (genes x cells)
            meta: Cell metadata with 'cell_type' column
            output_dir: Output directory for results
            iterations: Number of permutations
            threshold: Percentage threshold for filtering
            
        Returns:
            DataFrame with significant interactions
        """
        try:
            from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
        except ImportError:
            raise ImportError("CellPhoneDB not installed. Run: pip install cellphonedb")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        deconvoluted, means, pvalues, significant_means = cpdb_statistical_analysis_method.call(
            cpdb_file_path=None,  # Use default database
            meta_file_path=meta,
            counts_file_path=counts,
            counts_data='hgnc_symbol',
            iterations=iterations,
            threshold=threshold,
            threads=self.n_jobs,
            output_path=str(output_dir),
        )
        
        return significant_means
    
    def filter_significant(
        self,
        results: pd.DataFrame,
        pvalue_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Filter to significant interactions."""
        # Filter based on p-values
        # CellPhoneDB outputs have cell-type pair columns
        
        # Find columns with p-values (cell type pairs)
        pair_cols = [c for c in results.columns if '|' in c]
        
        # Filter rows where any pair is significant
        significant_mask = (results[pair_cols] < pvalue_threshold).any(axis=1)
        
        return results[significant_mask]
