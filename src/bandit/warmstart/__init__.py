"""
Phase 8a: Warm-Start Bandit Pre-Training Module.

This module implements offline pre-training for the bandit router to address
the cold-start limitation discovered in Phase 7. The approach uses:

1. PCA dimensionality reduction (768 -> 64 dimensions)
2. Oracle label construction from Phase 6 results
3. Simulated log behavior cloning via VW
4. Warm-started online deployment

Components:
- PCATransformer: Fit and apply PCA to MiniLM embeddings
- OracleBuilder: Merge Phase 6b/6c results into oracle labels
- VWConverter: Convert oracle data to VW ADF format
- WarmTrainer: Offline VW training harness
"""

from .pca_transformer import PCATransformer
from .oracle_builder import OracleBuilder, OracleLabel
from .vw_converter import VWConverter
from .warm_trainer import WarmTrainer
from .warm_router import WarmStartRouter

__all__ = [
    "PCATransformer",
    "OracleBuilder",
    "OracleLabel",
    "VWConverter",
    "WarmTrainer",
    "WarmStartRouter",
]
