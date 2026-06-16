"""Visualization utilities for predictions."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class Plotter:
    """Handle visualization of predictions."""
    
    def __init__(self, figsize: tuple = (14, 7), dpi: int = 100, style: str = "seaborn-v0_8-darkgrid"):
        """Initialize plotter."""
        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Could not apply style {style}: {e}")
    
    def plot_ensemble_predictions(
        self,
        actual_data: pd.Series,
        predictions: pd.Series,
        split_date: Optional[pd.Timestamp] = None,
        title: str = "Nifty 50 Index Prediction",
        save_path: Optional[Path] = None
    ) -> None:
        """Plot actual vs predicted prices."""
        logger.info("Creating ensemble prediction plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot actual prices
        ax.plot(actual_data.index, actual_data.values, label='Actual Price', color='blue', linewidth=2)
        
        # Plot predictions
        ax.plot(predictions.index, predictions.values, label='Predicted Price', color='red', 
                linewidth=2, linestyle='--')
        
        # Add split line
        if split_date is not None:
            ax.axvline(x=split_date, color='green', linestyle='--', linewidth=2, label='Prediction Start')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Close Price', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_with_support_resistance(
        self,
        close_prices: pd.Series,
        predictions: pd.Series,
        support: pd.Series,
        resistance: pd.Series,
        title: str = "Nifty 50 with Support/Resistance",
        save_path: Optional[Path] = None
    ) -> None:
        """Plot prices with support and resistance levels."""
        logger.info("Creating support/resistance plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot actual prices
        ax.plot(close_prices.index, close_prices.values, label='Actual Price', color='blue', linewidth=2)
        
        # Plot predictions
        ax.plot(predictions.index, predictions.values, label='Predicted Price', color='red',
                linewidth=2, linestyle='--')
        
        # Fill support/resistance region
        ax.fill_between(support.index, support.values, resistance.values, alpha=0.2, color='gray',
                       label='Support/Resistance Zone')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Close Price', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_components(
        self,
        ensemble_pred: pd.Series,
        components: dict,
        title: str = "Prediction Components",
        save_path: Optional[Path] = None
    ) -> None:
        """Plot ensemble components."""
        logger.info("Creating components plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # ARIMA
        axes[0, 0].plot(ensemble_pred.index, components['arima'], label='ARIMA', color='blue')
        axes[0, 0].set_title('ARIMA Forecast')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # GARCH
        axes[0, 1].plot(ensemble_pred.index, components['garch_volatility'], label='GARCH Volatility', color='green')
        axes[0, 1].set_title('GARCH Volatility')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # ML Correction
        axes[1, 0].plot(ensemble_pred.index, components['ml_correction'], label='ML Correction', color='orange')
        axes[1, 0].set_title('ML Residual Correction')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Ensemble
        axes[1, 1].plot(ensemble_pred.index, ensemble_pred.values, label='Ensemble', color='red', linewidth=2)
        axes[1, 1].set_title('Final Ensemble Prediction')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Components plot saved to {save_path}")
        
        plt.show()
