from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    n_train_sizes: list[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 5000]
    )
    n_test: int = 1000
    data_seeds: list[int] = field(default_factory=lambda: [11, 22, 33])


@dataclass
class ResolutionConfig:
    train_resolutions: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    eval_resolutions: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )


@dataclass
class ModelConfig:
    activation: str = "gelu"
    hidden_widths: list[int] = field(default_factory=lambda: [64, 64])


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 5000
    early_stopping_patience: int = 200
    batch_size: int = 64
    train_seeds: list[int] = field(default_factory=lambda: [101, 202, 303])


@dataclass
class TaskConfig:
    """Task-specific parameters (overridden per PDE family)."""
    name: str = "burgers_operator"
    T: float = 1.0
    domain: tuple[float, float] = (0.0, 1.0)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    out_dir: str = "runs"
