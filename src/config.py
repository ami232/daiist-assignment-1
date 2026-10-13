from dataclasses import dataclass, field
from datetime import date
from typing import Literal

PlotSize = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class PlotConfig:
    line_colors: tuple[str, ...] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    )
    carpet_colorscale: str = "RdYlBu_r"
    calendar_colorscale: str = "Viridis"
    heatmap_colorscale: str = "RdBu_r"
    table_cmap: str = "RdYlGn_r"
    # (width, height) in px
    sizes: dict[PlotSize, tuple[int, int]] = field(
        default_factory=lambda: {
            "small": (700, 350),
            "medium": (1000, 500),
            "large": (1400, 700),
        },
    )


@dataclass(frozen=True)
class ModelConfig:
    # spreads before 2021 almost never clear the cycle cost: a different market
    start: date = date(2021, 1, 1)
    # EUR/MWh battery wear: the spread after losses must beat it to cycle
    cycle_cost: float = 20.0
    test_frac: float = 0.2
    n_folds: int = 5
    n_trials: int = 20
    epochs: int = 300
    seed: int = 0


@dataclass(frozen=True)
class Config:
    plot: PlotConfig = field(default_factory=PlotConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


CONFIG = Config()
