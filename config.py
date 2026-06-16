"""Configuration management for Nifty 50 predictor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    data_dir: Path = Path(__file__).parent / "data"
    csv_file: str = "nifty_50_1y.csv"
    date_column: str = "Date"
    close_column: str = "Close"


@dataclass
class ARIMAConfig:
    """ARIMA model configuration."""
    
    order: tuple = (5, 1, 0)  # (p, d, q)
    seasonal_order: Optional[tuple] = None


@dataclass
class GARCHConfig:
    """GARCH model configuration."""
    
    p: int = 1
    q: int = 1
    vol_model: str = "Garch"
    rescale_factor: float = 1000.0


@dataclass
class MLConfig:
    """Machine Learning model configuration."""
    
    enabled: bool = True
    n_estimators: int = 100
    random_state: int = 42
    lags: int = 5


@dataclass
class SupportResistanceConfig:
    """Support and Resistance calculation configuration."""
    
    enabled: bool = True
    window_size: int = 10


@dataclass
class PredictionConfig:
    """Prediction configuration."""
    
    train_test_split: float = 0.8
    prediction_mode: str = "test"  # "test" or "future"
    future_days: int = 5
    train_end_date: Optional[str] = None  # e.g., "2024-11-26"
    future_start_date: Optional[str] = None  # e.g., "2024-11-27"
    future_end_date: Optional[str] = None  # e.g., "2024-11-29"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    
    figsize: tuple = (14, 7)
    dpi: int = 100
    style: str = "seaborn-v0_8-darkgrid"
    save_plot: bool = False
    plot_dir: Path = Path(__file__).parent / "outputs"


@dataclass
class Config:
    """Main configuration class."""
    
    data: DataConfig = DataConfig()
    arima: ARIMAConfig = ARIMAConfig()
    garch: GARCHConfig = GARCHConfig()
    ml: MLConfig = MLConfig()
    support_resistance: SupportResistanceConfig = SupportResistanceConfig()
    prediction: PredictionConfig = PredictionConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.visualization.plot_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
