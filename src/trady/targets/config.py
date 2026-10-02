"""Configuration management for TRADY Target Engine."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for target computation and labeled dataset assembly.

    Attributes:
        selected_targets: Tuple of target names to compute.
        drop_unavailable_rows: Whether to discard trailing rows where future targets
            are unavailable (NaN) due to reaching the end of the historical sample.
        output_dir: Destination folder for serialized target datasets.
    """

    selected_targets: tuple[str, ...] = ("target_fwd_ret_5d", "target_binary_up_5d")
    drop_unavailable_rows: bool = False
    output_dir: str = "data/targets"

    def __post_init__(self) -> None:
        if not self.selected_targets:
            raise ValueError("selected_targets cannot be empty")

    def resolved_output_path(self, base_dir: Path | None = None) -> Path:
        """Resolve output path relative to base or project directory."""
        p = Path(self.output_dir)
        if p.is_absolute() or base_dir is None:
            return p
        return base_dir / p
