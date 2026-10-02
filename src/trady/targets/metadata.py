"""Target metadata and specification models for TRADY."""

from dataclasses import dataclass
from typing import Literal

TargetType = Literal["continuous", "binary", "categorical"]


@dataclass(frozen=True)
class TargetMetadata:
    """Metadata describing a forward-looking target/label.

    Attributes:
        name: Canonical name of the target column (e.g. 'target_fwd_ret_5d').
        horizon: Number of future periods/bars H into the future (H >= 1).
        target_type: 'continuous', 'binary', or 'categorical'.
        threshold: Decision boundary threshold for binary/categorical classification.
        calculation_method: Mathematical identifier (e.g. 'forward_simple_return').
        description: Human-readable explanation of the future outcome.
        is_future_target: Flag explicitly identifying future outcome data (True).
    """

    name: str
    horizon: int
    target_type: TargetType
    calculation_method: str
    description: str
    threshold: float | None = None
    is_future_target: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Target name cannot be empty")

        if self.horizon < 1:
            raise ValueError(
                f"Target horizon must be at least 1 bar, got {self.horizon}"
            )

        if self.target_type not in ("continuous", "binary", "categorical"):
            msg = (
                f"Invalid target_type '{self.target_type}'. Must be "
                "continuous, binary, or categorical"
            )
            raise ValueError(msg)

        if self.target_type == "binary" and self.threshold is None:
            # Default threshold for binary classification is 0.0
            object.__setattr__(self, "threshold", 0.0)
