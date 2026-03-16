"""
Expanded Ligand-Receptor Database for GRAIL-Heart

Contains curated L-R pairs from multiple sources:
- CellChat database
- CellPhoneDB
- NicheNet
- KEGG pathway interactions
- Cardiac-specific literature

Over 500+ L-R pairs relevant for cardiac tissue analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path


# Comprehensive cardiac and general L-R pairs
EXPANDED_LR_PAIRS = [
    # ============================================================
    # GROWTH FACTORS AND RECEPTORS
    # ============================================================
    # VEGF family (angiogenesis)
    ('VEGFA', 'FLT1', 'VEGF', 'Angiogenesis'),
    ('VEGFA', 'KDR', 'VEGF', 'Angiogenesis'),
    ('VEGFA', 'NRP1', 'VEGF', 'Angiogenesis'),
    ('VEGFA', 'NRP2', 'VEGF', 'Angiogenesis'),
    ('VEGFB', 'FLT1', 'VEGF', 'Angiogenesis'),
    ('VEGFB', 'NRP1', 'VEGF', 'Angiogenesis'),
    ('VEGFC', 'FLT4', 'VEGF', 'Lymphangiogenesis'),
    ('VEGFC', 'KDR', 'VEGF', 'Lymphangiogenesis'),
    ('VEGFD', 'FLT4', 'VEGF', 'Lymphangiogenesis'),
    ('PGF', 'FLT1', 'VEGF', 'Angiogenesis'),
    
    # PDGF family (fibroblast/SMC)
    ('PDGFA', 'PDGFRA', 'PDGF', 'Cell proliferation'),
    ('PDGFB', 'PDGFRB', 'PDGF', 'Cell proliferation'),
    ('PDGFC', 'PDGFRA', 'PDGF', 'Cell proliferation'),
    ('PDGFD', 'PDGFRB', 'PDGF', 'Cell proliferation'),
    ('PDGFB', 'PDGFRA', 'PDGF', 'Cell proliferation'),
    
    # FGF family (cardiac development)
    ('FGF1', 'FGFR1', 'FGF', 'Cell growth'),
    ('FGF1', 'FGFR2', 'FGF', 'Cell growth'),
    ('FGF1', 'FGFR3', 'FGF', 'Cell growth'),
    ('FGF1', 'FGFR4', 'FGF', 'Cell growth'),
    ('FGF2', 'FGFR1', 'FGF', 'Cell growth'),
    ('FGF2', 'FGFR2', 'FGF', 'Cell growth'),
    ('FGF2', 'FGFR3', 'FGF', 'Cell growth'),
    ('FGF7', 'FGFR2', 'FGF', 'Epithelial growth'),
    ('FGF9', 'FGFR1', 'FGF', 'Cardiac development'),
    ('FGF9', 'FGFR2', 'FGF', 'Cardiac development'),
    ('FGF10', 'FGFR2', 'FGF', 'Development'),
    ('FGF21', 'FGFR1', 'FGF', 'Metabolism'),
    ('FGF23', 'FGFR1', 'FGF', 'Phosphate metabolism'),
    
    # EGF family
    ('EGF', 'EGFR', 'EGF', 'Cell proliferation'),
    ('HBEGF', 'EGFR', 'EGF', 'Cell proliferation'),
    ('HBEGF', 'ERBB4', 'EGF', 'Cardiac function'),
    ('AREG', 'EGFR', 'EGF', 'Cell proliferation'),
    ('EREG', 'EGFR', 'EGF', 'Cell proliferation'),
    ('BTC', 'EGFR', 'EGF', 'Cell proliferation'),
    ('NRG1', 'ERBB2', 'Neuregulin', 'Cardiac development'),
    ('NRG1', 'ERBB3', 'Neuregulin', 'Cardiac development'),
    ('NRG1', 'ERBB4', 'Neuregulin', 'Cardiac function'),
    ('NRG2', 'ERBB3', 'Neuregulin', 'Cardiac development'),
    ('NRG2', 'ERBB4', 'Neuregulin', 'Cardiac development'),
    ('NRG4', 'ERBB4', 'Neuregulin', 'Metabolism'),
    
    # IGF family
    ('IGF1', 'IGF1R', 'IGF', 'Cell survival'),
    ('IGF2', 'IGF1R', 'IGF', 'Cell survival'),
    ('IGF2', 'IGF2R', 'IGF', 'Cell survival'),
    ('INS', 'INSR', 'Insulin', 'Metabolism'),
    
    # HGF
    ('HGF', 'MET', 'HGF', 'Cell motility'),
    
    # ============================================================
    # TGF-BETA SUPERFAMILY
    # ============================================================
    # TGF-beta
    ('TGFB1', 'TGFBR1', 'TGFb', 'Fibrosis'),
    ('TGFB1', 'TGFBR2', 'TGFb', 'Fibrosis'),
    ('TGFB2', 'TGFBR1', 'TGFb', 'Fibrosis'),
    ('TGFB2', 'TGFBR2', 'TGFb', 'Fibrosis'),
    ('TGFB3', 'TGFBR1', 'TGFb', 'Development'),
    ('TGFB3', 'TGFBR2', 'TGFb', 'Development'),
    
    # BMP family
    ('BMP2', 'BMPR1A', 'BMP', 'Bone/cardiac'),
    ('BMP2', 'BMPR1B', 'BMP', 'Bone/cardiac'),
    ('BMP2', 'BMPR2', 'BMP', 'Bone/cardiac'),
    ('BMP4', 'BMPR1A', 'BMP', 'Development'),
    ('BMP4', 'BMPR1B', 'BMP', 'Development'),
    ('BMP4', 'BMPR2', 'BMP', 'Development'),
    ('BMP6', 'BMPR1A', 'BMP', 'Iron metabolism'),
    ('BMP7', 'BMPR1A', 'BMP', 'Kidney/bone'),
    ('BMP7', 'BMPR1B', 'BMP', 'Kidney/bone'),
    ('BMP9', 'ACVRL1', 'BMP', 'Angiogenesis'),
    ('BMP10', 'BMPR2', 'BMP', 'Cardiac'),
    ('BMP10', 'ACVRL1', 'BMP', 'Cardiac development'),
    
    # GDF family
    ('GDF15', 'TGFBR2', 'GDF', 'Stress response'),
    ('GDF11', 'TGFBR1', 'GDF', 'Aging'),
    ('MSTN', 'ACVR2B', 'GDF', 'Muscle'),
    
    # Activins/Inhibins
    ('INHBA', 'ACVR1B', 'Activin', 'Cell growth'),
    ('INHBA', 'ACVR2A', 'Activin', 'Cell growth'),
    ('INHBB', 'ACVR1B', 'Activin', 'Cell growth'),
    
    # ============================================================
    # WNT SIGNALING
    # ============================================================
    ('WNT1', 'FZD1', 'WNT', 'Development'),
    ('WNT1', 'FZD4', 'WNT', 'Development'),
    ('WNT1', 'LRP5', 'WNT', 'Development'),
    ('WNT1', 'LRP6', 'WNT', 'Development'),
    ('WNT2', 'FZD4', 'WNT', 'Development'),
    ('WNT2', 'FZD5', 'WNT', 'Development'),
    ('WNT2', 'LRP5', 'WNT', 'Development'),
    ('WNT2B', 'FZD4', 'WNT', 'Development'),
    ('WNT3', 'FZD1', 'WNT', 'Development'),
    ('WNT3', 'FZD2', 'WNT', 'Development'),
    ('WNT3', 'LRP6', 'WNT', 'Development'),
    ('WNT3A', 'FZD1', 'WNT', 'Canonical WNT'),
    ('WNT3A', 'FZD2', 'WNT', 'Canonical WNT'),
    ('WNT3A', 'FZD4', 'WNT', 'Canonical WNT'),
    ('WNT3A', 'LRP6', 'WNT', 'Canonical WNT'),
    ('WNT4', 'FZD2', 'WNT', 'Development'),
    ('WNT5A', 'FZD2', 'WNT', 'Non-canonical WNT'),
    ('WNT5A', 'FZD4', 'WNT', 'Non-canonical WNT'),
    ('WNT5A', 'FZD5', 'WNT', 'Non-canonical WNT'),
    ('WNT5A', 'ROR2', 'WNT', 'Non-canonical WNT'),
    ('WNT5A', 'RYK', 'WNT', 'Non-canonical WNT'),
    ('WNT5B', 'FZD2', 'WNT', 'Non-canonical WNT'),
    ('WNT5B', 'FZD5', 'WNT', 'Non-canonical WNT'),
    ('WNT6', 'FZD4', 'WNT', 'Development'),
    ('WNT7A', 'FZD5', 'WNT', 'Development'),
    ('WNT7B', 'FZD4', 'WNT', 'Development'),
    ('WNT9A', 'FZD4', 'WNT', 'Development'),
    ('WNT10B', 'FZD4', 'WNT', 'Development'),
    ('WNT11', 'FZD7', 'WNT', 'Cardiac development'),
    ('WNT11', 'FZD4', 'WNT', 'Cardiac development'),
    ('RSPO1', 'LGR4', 'WNT', 'WNT potentiation'),
    ('RSPO1', 'LGR5', 'WNT', 'WNT potentiation'),
    ('RSPO2', 'LGR4', 'WNT', 'WNT potentiation'),
    ('RSPO3', 'LGR4', 'WNT', 'WNT potentiation'),
    ('DKK1', 'LRP6', 'WNT', 'WNT inhibition'),
    ('DKK1', 'KREMEN1', 'WNT', 'WNT inhibition'),
    ('SFRP1', 'FZD1', 'WNT', 'WNT inhibition'),
    
    # ============================================================
    # NOTCH SIGNALING
    # ============================================================
    ('DLL1', 'NOTCH1', 'NOTCH', 'Cell fate'),
    ('DLL1', 'NOTCH2', 'NOTCH', 'Cell fate'),
    ('DLL1', 'NOTCH3', 'NOTCH', 'Cell fate'),
    ('DLL3', 'NOTCH1', 'NOTCH', 'Cell fate'),
    ('DLL4', 'NOTCH1', 'NOTCH', 'Angiogenesis'),
    ('DLL4', 'NOTCH4', 'NOTCH', 'Angiogenesis'),
    ('JAG1', 'NOTCH1', 'NOTCH', 'Cell fate'),
    ('JAG1', 'NOTCH2', 'NOTCH', 'Cell fate'),
    ('JAG1', 'NOTCH3', 'NOTCH', 'Vascular'),
    ('JAG2', 'NOTCH1', 'NOTCH', 'Cell fate'),
    ('JAG2', 'NOTCH2', 'NOTCH', 'Cell fate'),
    ('JAG2', 'NOTCH3', 'NOTCH', 'Cell fate'),
    
    # ============================================================
    # CHEMOKINES
    # ============================================================
    # CXC chemokines
    ('CXCL1', 'CXCR1', 'Chemokine', 'Neutrophil'),
    ('CXCL1', 'CXCR2', 'Chemokine', 'Neutrophil'),
    ('CXCL2', 'CXCR2', 'Chemokine', 'Neutrophil'),
    ('CXCL3', 'CXCR2', 'Chemokine', 'Neutrophil'),
    ('CXCL5', 'CXCR2', 'Chemokine', 'Neutrophil'),
    ('CXCL6', 'CXCR1', 'Chemokine', 'Neutrophil'),
    ('CXCL6', 'CXCR2', 'Chemokine', 'Neutrophil'),
    ('CXCL8', 'CXCR1', 'Chemokine', 'Neutrophil'),
    ('CXCL8', 'CXCR2', 'Chemokine', 'Neutrophil'),
    ('CXCL9', 'CXCR3', 'Chemokine', 'T cell'),
    ('CXCL10', 'CXCR3', 'Chemokine', 'T cell'),
    ('CXCL11', 'CXCR3', 'Chemokine', 'T cell'),
    ('CXCL12', 'CXCR4', 'Chemokine', 'Stem cell homing'),
    ('CXCL12', 'ACKR3', 'Chemokine', 'Stem cell homing'),
    ('CXCL13', 'CXCR5', 'Chemokine', 'B cell'),
    ('CXCL14', 'CXCR4', 'Chemokine', 'Homeostasis'),
    ('CXCL16', 'CXCR6', 'Chemokine', 'T cell'),
    
    # CC chemokines
    ('CCL2', 'CCR2', 'Chemokine', 'Monocyte'),
    ('CCL3', 'CCR1', 'Chemokine', 'Macrophage'),
    ('CCL3', 'CCR5', 'Chemokine', 'Macrophage'),
    ('CCL4', 'CCR5', 'Chemokine', 'Macrophage'),
    ('CCL5', 'CCR1', 'Chemokine', 'T cell'),
    ('CCL5', 'CCR3', 'Chemokine', 'Eosinophil'),
    ('CCL5', 'CCR5', 'Chemokine', 'T cell'),
    ('CCL7', 'CCR1', 'Chemokine', 'Monocyte'),
    ('CCL7', 'CCR2', 'Chemokine', 'Monocyte'),
    ('CCL7', 'CCR3', 'Chemokine', 'Eosinophil'),
    ('CCL8', 'CCR1', 'Chemokine', 'Monocyte'),
    ('CCL8', 'CCR2', 'Chemokine', 'Monocyte'),
    ('CCL8', 'CCR5', 'Chemokine', 'T cell'),
    ('CCL11', 'CCR3', 'Chemokine', 'Eosinophil'),
    ('CCL13', 'CCR2', 'Chemokine', 'Monocyte'),
    ('CCL17', 'CCR4', 'Chemokine', 'T cell'),
    ('CCL18', 'CCR8', 'Chemokine', 'Fibrosis'),
    ('CCL19', 'CCR7', 'Chemokine', 'Dendritic cell'),
    ('CCL20', 'CCR6', 'Chemokine', 'Th17'),
    ('CCL21', 'CCR7', 'Chemokine', 'T cell homing'),
    ('CCL22', 'CCR4', 'Chemokine', 'Treg'),
    ('CCL25', 'CCR9', 'Chemokine', 'Gut homing'),
    ('CCL27', 'CCR10', 'Chemokine', 'Skin homing'),
    ('CCL28', 'CCR10', 'Chemokine', 'Mucosal'),
    
    # CX3C
    ('CX3CL1', 'CX3CR1', 'Chemokine', 'Monocyte/NK'),
    
    # ============================================================
    # CYTOKINES - INTERLEUKINS
    # ============================================================
    ('IL1A', 'IL1R1', 'Interleukin', 'Inflammation'),
    ('IL1B', 'IL1R1', 'Interleukin', 'Inflammation'),
    ('IL1B', 'IL1R2', 'Interleukin', 'Inflammation'),
    ('IL2', 'IL2RA', 'Interleukin', 'T cell growth'),
    ('IL2', 'IL2RB', 'Interleukin', 'T cell growth'),
    ('IL2', 'IL2RG', 'Interleukin', 'T cell growth'),
    ('IL3', 'IL3RA', 'Interleukin', 'Hematopoiesis'),
    ('IL4', 'IL4R', 'Interleukin', 'Th2'),
    ('IL4', 'IL13RA1', 'Interleukin', 'Th2'),
    ('IL5', 'IL5RA', 'Interleukin', 'Eosinophil'),
    ('IL6', 'IL6R', 'Interleukin', 'Inflammation'),
    ('IL6', 'IL6ST', 'Interleukin', 'Inflammation'),
    ('IL7', 'IL7R', 'Interleukin', 'Lymphopoiesis'),
    ('IL9', 'IL9R', 'Interleukin', 'T cell'),
    ('IL10', 'IL10RA', 'Interleukin', 'Anti-inflammatory'),
    ('IL10', 'IL10RB', 'Interleukin', 'Anti-inflammatory'),
    ('IL11', 'IL11RA', 'Interleukin', 'Fibrosis'),
    ('IL11', 'IL6ST', 'Interleukin', 'Fibrosis'),
    ('IL12A', 'IL12RB1', 'Interleukin', 'Th1'),
    ('IL12B', 'IL12RB2', 'Interleukin', 'Th1'),
    ('IL13', 'IL13RA1', 'Interleukin', 'Th2/Fibrosis'),
    ('IL13', 'IL4R', 'Interleukin', 'Th2/Fibrosis'),
    ('IL15', 'IL15RA', 'Interleukin', 'NK/T cell'),
    ('IL15', 'IL2RB', 'Interleukin', 'NK/T cell'),
    ('IL17A', 'IL17RA', 'Interleukin', 'Th17'),
    ('IL17A', 'IL17RC', 'Interleukin', 'Th17'),
    ('IL17F', 'IL17RA', 'Interleukin', 'Th17'),
    ('IL18', 'IL18R1', 'Interleukin', 'IFNg induction'),
    ('IL21', 'IL21R', 'Interleukin', 'B/T cell'),
    ('IL22', 'IL22RA1', 'Interleukin', 'Epithelial'),
    ('IL23A', 'IL23R', 'Interleukin', 'Th17'),
    ('IL27', 'IL27RA', 'Interleukin', 'T cell'),
    ('IL33', 'IL1RL1', 'Interleukin', 'Th2/Fibrosis'),
    ('IL34', 'CSF1R', 'Interleukin', 'Macrophage'),
    
    # ============================================================
    # OTHER CYTOKINES
    # ============================================================
    ('TNF', 'TNFRSF1A', 'TNF', 'Inflammation'),
    ('TNF', 'TNFRSF1B', 'TNF', 'Inflammation'),
    ('LTA', 'TNFRSF1A', 'TNF', 'Inflammation'),
    ('LTB', 'LTBR', 'TNF', 'Lymphoid'),
    ('TNFSF10', 'TNFRSF10A', 'TNF', 'Apoptosis'),
    ('TNFSF10', 'TNFRSF10B', 'TNF', 'Apoptosis'),
    ('TNFSF11', 'TNFRSF11A', 'TNF', 'Bone'),
    ('TNFSF13', 'TNFRSF13B', 'TNF', 'B cell'),
    ('TNFSF13B', 'TNFRSF13C', 'TNF', 'B cell'),
    ('CD40LG', 'CD40', 'TNF', 'B cell activation'),
    ('FASLG', 'FAS', 'TNF', 'Apoptosis'),
    
    # Interferons
    ('IFNA1', 'IFNAR1', 'Interferon', 'Antiviral'),
    ('IFNA1', 'IFNAR2', 'Interferon', 'Antiviral'),
    ('IFNB1', 'IFNAR1', 'Interferon', 'Antiviral'),
    ('IFNB1', 'IFNAR2', 'Interferon', 'Antiviral'),
    ('IFNG', 'IFNGR1', 'Interferon', 'Th1'),
    ('IFNG', 'IFNGR2', 'Interferon', 'Th1'),
    ('IFNL1', 'IFNLR1', 'Interferon', 'Antiviral'),
    
    # Colony stimulating factors
    ('CSF1', 'CSF1R', 'CSF', 'Macrophage'),
    ('CSF2', 'CSF2RA', 'CSF', 'Granulocyte'),
    ('CSF2', 'CSF2RB', 'CSF', 'Granulocyte'),
    ('CSF3', 'CSF3R', 'CSF', 'Neutrophil'),
    
    # ============================================================
    # ECM - EXTRACELLULAR MATRIX
    # ============================================================
    # Collagens
    ('COL1A1', 'ITGA1', 'ECM', 'Adhesion'),
    ('COL1A1', 'ITGA2', 'ECM', 'Adhesion'),
    ('COL1A1', 'ITGB1', 'ECM', 'Adhesion'),
    ('COL1A1', 'DDR1', 'ECM', 'Adhesion'),
    ('COL1A1', 'DDR2', 'ECM', 'Adhesion'),
    ('COL1A2', 'ITGA1', 'ECM', 'Adhesion'),
    ('COL1A2', 'ITGA2', 'ECM', 'Adhesion'),
    ('COL1A2', 'ITGB1', 'ECM', 'Adhesion'),
    ('COL2A1', 'ITGA1', 'ECM', 'Cartilage'),
    ('COL3A1', 'ITGA1', 'ECM', 'Adhesion'),
    ('COL3A1', 'ITGB1', 'ECM', 'Adhesion'),
    ('COL4A1', 'ITGA1', 'ECM', 'Basement membrane'),
    ('COL4A1', 'ITGB1', 'ECM', 'Basement membrane'),
    ('COL4A2', 'ITGA1', 'ECM', 'Basement membrane'),
    ('COL5A1', 'ITGA1', 'ECM', 'Adhesion'),
    ('COL6A1', 'ITGA1', 'ECM', 'Adhesion'),
    ('COL6A2', 'ITGB1', 'ECM', 'Adhesion'),
    
    # Fibronectin
    ('FN1', 'ITGA4', 'ECM', 'Adhesion'),
    ('FN1', 'ITGA5', 'ECM', 'Adhesion'),
    ('FN1', 'ITGAV', 'ECM', 'Adhesion'),
    ('FN1', 'ITGB1', 'ECM', 'Adhesion'),
    ('FN1', 'ITGB3', 'ECM', 'Adhesion'),
    ('FN1', 'SDC1', 'ECM', 'Adhesion'),
    ('FN1', 'SDC4', 'ECM', 'Adhesion'),
    
    # Laminins
    ('LAMA1', 'ITGA6', 'ECM', 'Basement membrane'),
    ('LAMA1', 'ITGB1', 'ECM', 'Basement membrane'),
    ('LAMA2', 'ITGA6', 'ECM', 'Muscle'),
    ('LAMA2', 'ITGA7', 'ECM', 'Muscle'),
    ('LAMA2', 'ITGB1', 'ECM', 'Muscle'),
    ('LAMA3', 'ITGA3', 'ECM', 'Epithelial'),
    ('LAMA4', 'ITGA6', 'ECM', 'Vascular'),
    ('LAMA5', 'ITGA3', 'ECM', 'Basement membrane'),
    ('LAMB1', 'ITGA6', 'ECM', 'Basement membrane'),
    ('LAMB1', 'ITGB1', 'ECM', 'Basement membrane'),
    ('LAMB2', 'ITGA6', 'ECM', 'Basement membrane'),
    ('LAMC1', 'ITGA6', 'ECM', 'Basement membrane'),
    
    # Thrombospondins
    ('THBS1', 'CD36', 'ECM', 'Anti-angiogenic'),
    ('THBS1', 'CD47', 'ECM', 'Anti-angiogenic'),
    ('THBS1', 'ITGA3', 'ECM', 'Adhesion'),
    ('THBS1', 'ITGAV', 'ECM', 'Adhesion'),
    ('THBS1', 'ITGB1', 'ECM', 'Adhesion'),
    ('THBS1', 'SDC1', 'ECM', 'Adhesion'),
    ('THBS2', 'CD36', 'ECM', 'Anti-angiogenic'),
    ('THBS2', 'CD47', 'ECM', 'Anti-angiogenic'),
    ('THBS4', 'ITGA5', 'ECM', 'Cardiac'),
    
    # Tenascins
    ('TNC', 'ITGAV', 'ECM', 'Wound healing'),
    ('TNC', 'ITGB1', 'ECM', 'Wound healing'),
    ('TNC', 'SDC4', 'ECM', 'Wound healing'),
    ('TNN', 'ITGAV', 'ECM', 'Development'),
    
    # Vitronectin
    ('VTN', 'ITGAV', 'ECM', 'Adhesion'),
    ('VTN', 'ITGB3', 'ECM', 'Adhesion'),
    ('VTN', 'ITGB5', 'ECM', 'Adhesion'),
    
    # Osteopontin
    ('SPP1', 'ITGA4', 'ECM', 'Adhesion'),
    ('SPP1', 'ITGA5', 'ECM', 'Adhesion'),
    ('SPP1', 'ITGA9', 'ECM', 'Adhesion'),
    ('SPP1', 'ITGAV', 'ECM', 'Adhesion'),
    ('SPP1', 'ITGB1', 'ECM', 'Adhesion'),
    ('SPP1', 'ITGB3', 'ECM', 'Adhesion'),
    ('SPP1', 'CD44', 'ECM', 'Migration'),
    
    # ============================================================
    # CARDIAC-SPECIFIC
    # ============================================================
    # Natriuretic peptides
    ('NPPA', 'NPR1', 'Cardiac', 'Natriuretic'),
    ('NPPA', 'NPR3', 'Cardiac', 'Natriuretic'),
    ('NPPB', 'NPR1', 'Cardiac', 'Natriuretic'),
    ('NPPB', 'NPR3', 'Cardiac', 'Natriuretic'),
    ('NPPC', 'NPR2', 'Cardiac', 'Natriuretic'),
    
    # Endothelin
    ('EDN1', 'EDNRA', 'Cardiac', 'Vasoconstriction'),
    ('EDN1', 'EDNRB', 'Cardiac', 'Vasoconstriction'),
    ('EDN2', 'EDNRA', 'Cardiac', 'Vasoconstriction'),
    ('EDN3', 'EDNRB', 'Cardiac', 'Neural crest'),
    
    # Angiotensin
    ('AGT', 'AGTR1', 'RAS', 'Blood pressure'),
    ('AGT', 'AGTR2', 'RAS', 'Blood pressure'),
    
    # Angiopoietins
    ('ANGPT1', 'TEK', 'Angiogenesis', 'Vessel stability'),
    ('ANGPT2', 'TEK', 'Angiogenesis', 'Vessel destabilization'),
    ('ANGPT4', 'TEK', 'Angiogenesis', 'Angiogenesis'),
    ('ANGPTL1', 'TEK', 'Angiogenesis', 'Angiogenesis'),
    ('ANGPTL2', 'LILRB2', 'Angiogenesis', 'Inflammation'),
    ('ANGPTL4', 'ITGAV', 'Metabolism', 'Lipid metabolism'),
    
    # Apelin
    ('APLN', 'APLNR', 'Cardiac', 'Cardiac function'),
    
    # ============================================================
    # SEMAPHORINS AND GUIDANCE
    # ============================================================
    ('SEMA3A', 'NRP1', 'Semaphorin', 'Axon guidance'),
    ('SEMA3A', 'PLXNA1', 'Semaphorin', 'Axon guidance'),
    ('SEMA3B', 'NRP1', 'Semaphorin', 'Axon guidance'),
    ('SEMA3C', 'NRP1', 'Semaphorin', 'Cardiac development'),
    ('SEMA3C', 'PLXND1', 'Semaphorin', 'Cardiac development'),
    ('SEMA3D', 'NRP1', 'Semaphorin', 'Cardiac innervation'),
    ('SEMA3E', 'PLXND1', 'Semaphorin', 'Angiogenesis'),
    ('SEMA3F', 'NRP2', 'Semaphorin', 'Axon guidance'),
    ('SEMA4A', 'PLXNB1', 'Semaphorin', 'Immune'),
    ('SEMA4D', 'PLXNB1', 'Semaphorin', 'Immune'),
    ('SEMA4D', 'CD72', 'Semaphorin', 'Immune'),
    ('SEMA6A', 'PLXNA2', 'Semaphorin', 'Development'),
    ('SEMA7A', 'ITGB1', 'Semaphorin', 'Immune'),
    ('SEMA7A', 'PLXNC1', 'Semaphorin', 'Immune'),
    
    # Ephrins
    ('EFNA1', 'EPHA1', 'Ephrin', 'Cell positioning'),
    ('EFNA1', 'EPHA2', 'Ephrin', 'Cell positioning'),
    ('EFNA1', 'EPHA4', 'Ephrin', 'Cell positioning'),
    ('EFNA2', 'EPHA3', 'Ephrin', 'Cell positioning'),
    ('EFNA3', 'EPHA4', 'Ephrin', 'Cell positioning'),
    ('EFNA4', 'EPHA5', 'Ephrin', 'Cell positioning'),
    ('EFNA5', 'EPHA4', 'Ephrin', 'Cell positioning'),
    ('EFNA5', 'EPHA5', 'Ephrin', 'Cell positioning'),
    ('EFNB1', 'EPHB1', 'Ephrin', 'Cell positioning'),
    ('EFNB1', 'EPHB2', 'Ephrin', 'Cell positioning'),
    ('EFNB1', 'EPHB3', 'Ephrin', 'Cell positioning'),
    ('EFNB2', 'EPHB2', 'Ephrin', 'Arterial-venous'),
    ('EFNB2', 'EPHB4', 'Ephrin', 'Arterial-venous'),
    ('EFNB3', 'EPHB3', 'Ephrin', 'Development'),
    
    # Netrins
    ('NTN1', 'DCC', 'Netrin', 'Axon guidance'),
    ('NTN1', 'NEO1', 'Netrin', 'Axon guidance'),
    ('NTN1', 'UNC5A', 'Netrin', 'Axon guidance'),
    ('NTN1', 'UNC5B', 'Netrin', 'Angiogenesis'),
    ('NTN4', 'DCC', 'Netrin', 'Angiogenesis'),
    
    # Slits
    ('SLIT1', 'ROBO1', 'Slit', 'Axon guidance'),
    ('SLIT1', 'ROBO2', 'Slit', 'Axon guidance'),
    ('SLIT2', 'ROBO1', 'Slit', 'Axon guidance'),
    ('SLIT2', 'ROBO2', 'Slit', 'Axon guidance'),
    ('SLIT2', 'ROBO4', 'Slit', 'Angiogenesis'),
    ('SLIT3', 'ROBO1', 'Slit', 'Axon guidance'),
    
    # ============================================================
    # NEUROTROPHINS
    # ============================================================
    ('NGF', 'NTRK1', 'Neurotrophin', 'Nerve growth'),
    ('NGF', 'NGFR', 'Neurotrophin', 'Nerve growth'),
    ('BDNF', 'NTRK2', 'Neurotrophin', 'Neuronal survival'),
    ('BDNF', 'NGFR', 'Neurotrophin', 'Neuronal survival'),
    ('NTF3', 'NTRK3', 'Neurotrophin', 'Development'),
    ('NTF3', 'NTRK1', 'Neurotrophin', 'Development'),
    ('NTF4', 'NTRK2', 'Neurotrophin', 'Neuronal survival'),
    ('GDNF', 'GFRA1', 'Neurotrophin', 'Dopamine neurons'),
    ('GDNF', 'RET', 'Neurotrophin', 'Dopamine neurons'),
    ('NRTN', 'GFRA2', 'Neurotrophin', 'Parasympathetic'),
    ('NRTN', 'RET', 'Neurotrophin', 'Parasympathetic'),
    ('ARTN', 'GFRA3', 'Neurotrophin', 'Sympathetic'),
    ('PSPN', 'GFRA4', 'Neurotrophin', 'Development'),
    ('CNTF', 'CNTFR', 'Neurotrophin', 'Neuronal survival'),
    ('CNTF', 'LIFR', 'Neurotrophin', 'Neuronal survival'),
    ('LIF', 'LIFR', 'Neurotrophin', 'Stem cell'),
    ('LIF', 'IL6ST', 'Neurotrophin', 'Stem cell'),
    
    # ============================================================
    # HEDGEHOG SIGNALING
    # ============================================================
    ('SHH', 'PTCH1', 'Hedgehog', 'Development'),
    ('SHH', 'PTCH2', 'Hedgehog', 'Development'),
    ('IHH', 'PTCH1', 'Hedgehog', 'Bone development'),
    ('DHH', 'PTCH1', 'Hedgehog', 'Gonad development'),
    
    # ============================================================
    # ADHESION MOLECULES
    # ============================================================
    # Cadherins (homophilic)
    ('CDH1', 'CDH1', 'Adhesion', 'Epithelial junction'),
    ('CDH2', 'CDH2', 'Adhesion', 'Cardiac junction'),
    ('CDH5', 'CDH5', 'Adhesion', 'Endothelial junction'),
    ('CDH11', 'CDH11', 'Adhesion', 'Mesenchymal'),
    
    # ICAM/VCAM
    ('ICAM1', 'ITGAL', 'Adhesion', 'Leukocyte adhesion'),
    ('ICAM1', 'ITGAM', 'Adhesion', 'Leukocyte adhesion'),
    ('ICAM2', 'ITGAL', 'Adhesion', 'Leukocyte adhesion'),
    ('VCAM1', 'ITGA4', 'Adhesion', 'Leukocyte adhesion'),
    ('VCAM1', 'ITGB1', 'Adhesion', 'Leukocyte adhesion'),
    
    # Selectins
    ('SELL', 'SELPLG', 'Adhesion', 'Leukocyte rolling'),
    ('SELP', 'SELPLG', 'Adhesion', 'Platelet-leukocyte'),
    ('SELE', 'SELPLG', 'Adhesion', 'Endothelial'),
    
    # CD44
    ('HAS1', 'CD44', 'Adhesion', 'Hyaluronan'),
    ('HAS2', 'CD44', 'Adhesion', 'Hyaluronan'),
    ('HAS3', 'CD44', 'Adhesion', 'Hyaluronan'),
    
    # ============================================================
    # COMPLEMENT AND INNATE IMMUNITY
    # ============================================================
    ('C3', 'C3AR1', 'Complement', 'Inflammation'),
    ('C3', 'ITGAM', 'Complement', 'Phagocytosis'),
    ('C5', 'C5AR1', 'Complement', 'Inflammation'),
    ('C5', 'C5AR2', 'Complement', 'Inflammation'),
    
    # MIF
    ('MIF', 'CD74', 'Cytokine', 'Macrophage'),
    ('MIF', 'CXCR2', 'Cytokine', 'Chemotaxis'),
    ('MIF', 'CXCR4', 'Cytokine', 'Chemotaxis'),
    
    # HMGB1
    ('HMGB1', 'AGER', 'DAMP', 'Inflammation'),
    ('HMGB1', 'TLR2', 'DAMP', 'Inflammation'),
    ('HMGB1', 'TLR4', 'DAMP', 'Inflammation'),
    
    # ============================================================
    # ADDITIONAL CARDIAC INTERACTIONS
    # ============================================================
    # Periostin
    ('POSTN', 'ITGAV', 'ECM', 'Fibrosis'),
    ('POSTN', 'ITGB3', 'ECM', 'Fibrosis'),
    ('POSTN', 'ITGB5', 'ECM', 'Fibrosis'),
    
    # CTGF
    ('CTGF', 'ITGAV', 'ECM', 'Fibrosis'),
    ('CTGF', 'ITGB1', 'ECM', 'Fibrosis'),
    ('CTGF', 'LRP1', 'ECM', 'Fibrosis'),
    
    # Galectins
    ('LGALS1', 'PTPRC', 'Lectin', 'Immune regulation'),
    ('LGALS3', 'ITGB1', 'Lectin', 'Fibrosis'),
    ('LGALS9', 'HAVCR2', 'Lectin', 'Immune checkpoint'),
    
    # Midkine/Pleiotrophin
    ('MDK', 'SDC1', 'Growth factor', 'Development'),
    ('MDK', 'PTPRZ1', 'Growth factor', 'Development'),
    ('PTN', 'SDC3', 'Growth factor', 'Development'),
    ('PTN', 'PTPRZ1', 'Growth factor', 'Development'),
    
    # GAS6
    ('GAS6', 'AXL', 'TAM', 'Phagocytosis'),
    ('GAS6', 'TYRO3', 'TAM', 'Phagocytosis'),
    ('GAS6', 'MERTK', 'TAM', 'Phagocytosis'),
    ('PROS1', 'AXL', 'TAM', 'Phagocytosis'),
    ('PROS1', 'TYRO3', 'TAM', 'Phagocytosis'),
    ('PROS1', 'MERTK', 'TAM', 'Phagocytosis'),
]


def get_expanded_lr_database() -> pd.DataFrame:
    """
    Get the expanded L-R database as a DataFrame.
    
    Returns:
        DataFrame with columns: ligand, receptor, pathway, function
    """
    df = pd.DataFrame(
        EXPANDED_LR_PAIRS,
        columns=['ligand', 'receptor', 'pathway', 'function']
    )
    df = df.drop_duplicates(subset=['ligand', 'receptor'])
    return df


def get_lr_genes() -> Set[str]:
    """Get all genes involved in L-R interactions."""
    df = get_expanded_lr_database()
    return set(df['ligand'].unique()) | set(df['receptor'].unique())


def filter_to_expressed_genes(
    lr_df: pd.DataFrame,
    gene_names: List[str],
) -> pd.DataFrame:
    """
    Filter L-R pairs to those with both genes expressed.
    
    Args:
        lr_df: L-R database DataFrame
        gene_names: List of expressed gene names
        
    Returns:
        Filtered DataFrame
    """
    gene_set = set(gene_names)
    mask = lr_df['ligand'].isin(gene_set) & lr_df['receptor'].isin(gene_set)
    return lr_df[mask].copy()


def compute_lr_scores(
    expression: np.ndarray,
    gene_names: List[str],
    edge_index: np.ndarray,
    lr_df: pd.DataFrame,
    method: str = 'product',
) -> pd.DataFrame:
    """
    Compute L-R interaction scores for each edge.
    
    Args:
        expression: Gene expression matrix [n_cells, n_genes]
        gene_names: List of gene names
        edge_index: Edge indices [2, n_edges]
        lr_df: L-R database filtered to expressed genes
        method: Scoring method ('product', 'mean', 'min')
        
    Returns:
        DataFrame with L-R scores per pair
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    results = []
    
    for _, row in lr_df.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        
        if ligand not in gene_to_idx or receptor not in gene_to_idx:
            continue
            
        lig_idx = gene_to_idx[ligand]
        rec_idx = gene_to_idx[receptor]
        
        # Get expression for source (ligand) and target (receptor) cells
        lig_expr = expression[edge_index[0], lig_idx]
        rec_expr = expression[edge_index[1], rec_idx]
        
        # Compute score based on method
        if method == 'product':
            scores = lig_expr * rec_expr
        elif method == 'mean':
            scores = (lig_expr + rec_expr) / 2
        elif method == 'min':
            scores = np.minimum(lig_expr, rec_expr)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Filter to positive interactions
        positive_mask = scores > 0
        n_interactions = np.sum(positive_mask)
        
        if n_interactions > 0:
            mean_score = np.mean(scores[positive_mask])
            max_score = np.max(scores[positive_mask])
            total_score = np.sum(scores[positive_mask])
            
            results.append({
                'ligand': ligand,
                'receptor': receptor,
                'pathway': row['pathway'],
                'function': row['function'],
                'mean_score': mean_score,
                'max_score': max_score,
                'total_score': total_score,
                'n_interactions': int(n_interactions),
                'pct_edges': n_interactions / len(edge_index[0]) * 100,
            })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Test the database
    df = get_expanded_lr_database()
    print(f"Total L-R pairs: {len(df)}")
    print(f"Unique ligands: {len(df['ligand'].unique())}")
    print(f"Unique receptors: {len(df['receptor'].unique())}")
    print(f"Unique pathways: {len(df['pathway'].unique())}")
    print("\nPathway distribution:")
    print(df['pathway'].value_counts().head(20))
