"""Configuration models and schema definitions for TRADY."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SystemConfig(BaseModel):
    """Core platform system settings."""

    model_config = ConfigDict(frozen=True)

    app_name: str = Field(default="TRADY", description="Platform identifier")
    version: str = Field(default="0.1.0", description="Semantic version")
    environment: str = Field(
        default="development",
        description="Environment: development, testing, or production",
    )
    random_seed: int = Field(
        default=42, description="Global random seed for reproducibility"
    )


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level verbosity"
    )
    format: Literal["text", "json"] = Field(
        default="text", description="Log output formatting"
    )
    show_timestamp: bool = Field(
        default=True, description="Whether to include ISO timestamp"
    )
    show_caller: bool = Field(
        default=False, description="Whether to include module/caller info"
    )


class PathsConfig(BaseModel):
    """Data and directory path definitions."""

    model_config = ConfigDict(frozen=True)

    data_raw: Path = Field(default=Path("data/raw"))
    data_interim: Path = Field(default=Path("data/interim"))
    data_processed: Path = Field(default=Path("data/processed"))
    notebooks: Path = Field(default=Path("notebooks"))
    reports: Path = Field(default=Path("reports"))
    configs: Path = Field(default=Path("configs"))

    @field_validator(
        "data_raw",
        "data_interim",
        "data_processed",
        "notebooks",
        "reports",
        "configs",
        mode="before",
    )
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        return Path(v) if isinstance(v, str) else v


class HardwareConfig(BaseModel):
    """Target compute constraints tailored for Lenovo LOQ workstation."""

    model_config = ConfigDict(frozen=True)

    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Primary computation device target"
    )
    max_workers: int = Field(
        default=4, ge=1, le=16, description="Parallel compute thread/process count"
    )
    batch_size: int = Field(
        default=64, ge=1, description="Default batch size for memory-constrained models"
    )
    memory_limit_mb: int = Field(
        default=12288,
        ge=1024,
        description="Soft memory limit in MB (leaves headroom on 16GB RAM)",
    )
    cuda_deterministic: bool = Field(
        default=True, description="Enforce deterministic CUDA algorithms"
    )


class SafetyConfig(BaseModel):
    """Safety and regulatory firewalls.

    CRITICAL: Real-money trading is explicitly forbidden.
    """

    model_config = ConfigDict(frozen=True)

    live_trading_enabled: bool = Field(
        default=False,
        description="Must always be False. Real-money trading is strictly forbidden.",
    )
    paper_trading_only: bool = Field(
        default=True, description="Platform strictly restricted to paper simulation."
    )
    disclaimer_acknowledged: bool = Field(
        default=True,
        description=(
            "Affirmation of educational research scope and no profit guarantee."
        ),
    )
    max_position_size_pct: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum simulated position size as a fraction of portfolio",
    )
    max_drawdown_limit_pct: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum tolerated drawdown limit for simulated risk checks",
    )

    @model_validator(mode="after")
    def verify_safety_invariants(self) -> "SafetyConfig":
        if self.live_trading_enabled:
            raise ValueError(
                "SAFETY VIOLATION: TRADY is an educational research and paper-trading "
                "platform. Real-money live trading execution is strictly prohibited by "
                "system architecture."
            )
        if not self.paper_trading_only:
            raise ValueError(
                "SAFETY VIOLATION: TRADY must operate exclusively in paper-trading "
                "simulation mode."
            )
        return self


class Settings(BaseModel):
    """Root configuration model for TRADY."""

    model_config = ConfigDict(frozen=True)

    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
