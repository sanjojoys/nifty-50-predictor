"""Prediction models for Nifty 50 forecasting."""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class PredictionModel:
    """Base class for prediction models."""
    
    def __init__(self):
        """Initialize model."""
        self.model = None
        self.predictions = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit model to data."""
        raise NotImplementedError
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions."""
        raise NotImplementedError
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate predictions against actual values."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        logger.info(f"Model Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        
        return {"rmse": rmse, "mae": mae, "mape": mape}


class ARIMAModel(PredictionModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """Initialize ARIMA model."""
        super().__init__()
        self.order = order
    
    def fit(self, data: pd.Series) -> None:
        """Fit ARIMA model."""
        logger.info(f"Fitting ARIMA{self.order}...")
        self.model = ARIMA(data, order=self.order).fit()
        logger.info("ARIMA model fitted successfully")
    
    def predict(self, steps: int) -> np.ndarray:
        """Forecast using ARIMA."""
        self.predictions = self.model.forecast(steps=steps)
        return self.predictions.values
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.resid


class GARCHModel(PredictionModel):
    """GARCH model for volatility forecasting."""
    
    def __init__(self, p: int = 1, q: int = 1, rescale_factor: float = 1000.0):
        """Initialize GARCH model."""
        super().__init__()
        self.p = p
        self.q = q
        self.rescale_factor = rescale_factor
    
    def fit(self, residuals: pd.Series) -> None:
        """Fit GARCH model on residuals."""
        logger.info(f"Fitting GARCH({self.p},{self.q})...")
        
        # Scale residuals for numerical stability
        scaled_residuals = residuals / self.rescale_factor
        
        self.model = arch_model(
            scaled_residuals,
            vol='Garch',
            p=self.p,
            q=self.q
        ).fit(disp="off")
        
        logger.info("GARCH model fitted successfully")
    
    def predict_volatility(self, horizon: int) -> np.ndarray:
        """Forecast volatility."""
        forecast = self.model.forecast(horizon=horizon)
        variance = forecast.variance.iloc[-1].values
        volatility = np.sqrt(variance) * self.rescale_factor
        return volatility


class RandomForestModel(PredictionModel):
    """Random Forest model for residual correction."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """Initialize Random Forest model."""
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest model."""
        logger.info(f"Fitting Random Forest with {self.n_estimators} estimators...")
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        logger.info("Random Forest model fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.predictions = self.model.predict(X)
        return self.predictions


class EnsemblePredictor:
    """Ensemble predictor combining ARIMA, GARCH, and ML."""
    
    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (5, 1, 0),
        garch_p: int = 1,
        garch_q: int = 1,
        use_ml: bool = True,
        n_lags: int = 5
    ):
        """Initialize ensemble predictor."""
        self.arima = ARIMAModel(order=arima_order)
        self.garch = GARCHModel(p=garch_p, q=garch_q)
        self.rf = RandomForestModel() if use_ml else None
        self.n_lags = n_lags
    
    def fit(
        self,
        train_data: pd.Series,
        ml_features: Optional[np.ndarray] = None,
        ml_target: Optional[np.ndarray] = None
    ) -> None:
        """Fit all components of ensemble."""
        logger.info("Fitting ensemble model...")
        
        # Fit ARIMA
        self.arima.fit(train_data)
        
        # Fit GARCH on ARIMA residuals
        residuals = self.arima.get_residuals()
        self.garch.fit(residuals)
        
        # Fit ML model if provided
        if self.rf and ml_features is not None and ml_target is not None:
            self.rf.fit(ml_features, ml_target)
    
    def predict(
        self,
        steps: int,
        test_features: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble predictions."""
        logger.info(f"Generating predictions for {steps} steps...")
        
        # ARIMA forecast
        arima_pred = self.arima.predict(steps)
        
        # GARCH volatility
        garch_vol = self.garch.predict_volatility(steps)
        
        # ML residual correction
        ml_correction = np.zeros(steps)
        if self.rf and test_features is not None and len(test_features) > 0:
            ml_correction = self.rf.predict(test_features[:steps])
        
        # Combine predictions
        ensemble_pred = arima_pred + garch_vol + ml_correction
        
        logger.info(f"Predictions generated: mean={ensemble_pred.mean():.2f}, std={ensemble_pred.std():.2f}")
        
        return ensemble_pred, {
            "arima": arima_pred,
            "garch_volatility": garch_vol,
            "ml_correction": ml_correction
        }


def calculate_support_resistance(
    data: pd.DataFrame,
    close_column: str = "Close",
    window_size: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """Calculate support and resistance levels."""
    logger.info(f"Calculating support/resistance with window={window_size}...")
    
    support = data[close_column].rolling(window=window_size).min()
    resistance = data[close_column].rolling(window=window_size).max()
    
    return support, resistance
