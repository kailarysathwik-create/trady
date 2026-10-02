"""TRADY Target and Label Engine.

Generates forward-looking outcome targets for supervised quantitative machine
learning experiments while guaranteeing strict separation between prediction-time
features and future targets.

Core Principles:
- Features (X): information available at or before prediction time t <= T.
- Targets (y): future outcome being predicted t > T.
- Target Leakage Prevention: Features and Targets are guaranteed to be disjoint.
- Unavailable Future: Trailing rows (the final H observations) are explicitly NaN.
"""

from trady.targets.builder import (
    compute_binary_classification_target,
    compute_forward_return,
    compute_forward_volatility,
)
from trady.targets.config import TargetConfig
from trady.targets.dataset import ModelingDataset, ModelingDatasetMetadata
from trady.targets.metadata import TargetMetadata, TargetType
from trady.targets.pipeline import TargetPipeline, TargetReport
from trady.targets.registry import TargetRegistry, target_registry

__all__ = [
    "ModelingDataset",
    "ModelingDatasetMetadata",
    "TargetConfig",
    "TargetMetadata",
    "TargetPipeline",
    "TargetRegistry",
    "TargetReport",
    "TargetType",
    "compute_binary_classification_target",
    "compute_forward_return",
    "compute_forward_volatility",
    "target_registry",
]
