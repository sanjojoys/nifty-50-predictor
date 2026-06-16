"""Data handling and preprocessing module."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataHandler:
    """Handle data loading and preprocessing."""
    
    def __init__(self, data_dir: Path, csv_file: str):
        """Initialize DataHandler."""
        self.data_dir = Path(data_dir)
        self.csv_file = csv_file
        self.data: pd.DataFrame = None
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data from file."""
        file_path = self.data_dir / self.csv_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path)
        logger.info(f"Data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data
    
    def preprocess(self, date_column: str = "Date", close_column: str = "Close") -> pd.Series:
        """Preprocess data and return close prices."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Preprocessing data...")
        
        # Parse dates
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Sort by date
        self.data = self.data.sort_values(date_column)
        
        # Set date as index
        self.data.set_index(date_column, inplace=True)
        
        # Set business day frequency
        try:
            self.data = self.data.asfreq('B')
        except Exception as e:
            logger.warning(f"Could not set business day frequency: {e}")
        
        # Extract and validate close prices
        close_prices = self.data[close_column].dropna()
        
        logger.info(f"Preprocessing complete: {len(close_prices)} valid close prices")
        return close_prices
    
    def split_data(
        self,
        close_prices: pd.Series,
        train_test_split: float = 0.8,
        train_end_date: str = None
    ) -> Tuple[pd.Series, pd.Series]:
        """Split data into train and test sets."""
        if train_end_date:
            logger.info(f"Splitting data at date: {train_end_date}")
            train = close_prices[:train_end_date]
            test = close_prices[train_end_date:]
        else:
            split_idx = int(len(close_prices) * train_test_split)
            train = close_prices.iloc[:split_idx]
            test = close_prices.iloc[split_idx:]
            logger.info(f"Splitting data at index: {split_idx}")
        
        logger.info(f"Train set: {len(train)} samples, Test set: {len(test)} samples")
        return train, test
    
    def calculate_returns(self) -> pd.Series:
        """Calculate percentage returns from close prices."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        returns = self.data['Close'].pct_change()
        logger.info(f"Calculated returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
        return returns
    
    def get_lagged_features(self, data: pd.Series, lags: int = 5) -> Tuple[np.ndarray, pd.Series]:
        """Create lagged features for machine learning."""
        lagged = pd.concat([data.shift(i) for i in range(1, lags + 1)], axis=1).dropna()
        target = data.iloc[lags:]
        
        logger.info(f"Created lagged features: {lagged.shape[0]} samples, {lagged.shape[1]} features")
        return lagged.values, target
