# Nifty 50 Stock Index Predictor

A sophisticated time series forecasting system for the Nifty 50 index using ensemble methods combining ARIMA, GARCH, and Machine Learning models.

## Features

✨ **Ensemble Prediction Model**
- **ARIMA**: Captures trend and seasonality in time series data
- **GARCH**: Forecasts volatility (conditional heteroskedasticity)
- **Random Forest**: Learns complex patterns in residuals
- Combines all three for robust predictions

📊 **Advanced Analysis**
- Support and Resistance level calculation
- Technical indicator computation
- Component-wise prediction visualization
- Comprehensive evaluation metrics (RMSE, MAE, MAPE)

🎯 **Flexible Modes**
- Test set prediction (80-20 split)
- Future date range prediction
- Configurable parameters via `config.py`

📈 **Professional Visualizations**
- Actual vs. predicted prices overlay
- Support/resistance zones
- Component analysis (ARIMA, GARCH, ML)
- Publication-ready plots with customizable styling

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nifty-50-predictor.git
   cd nifty-50-predictor
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

Run the predictor with default configuration:

```bash
python main.py
```

### Configure Prediction Mode

Edit `config.py` to choose prediction mode and parameters:

```python
# For test set prediction (80-20 split)
prediction: PredictionConfig = PredictionConfig(
    prediction_mode="test",
    train_test_split=0.8
)

# For future prediction (e.g., next 5 business days)
prediction: PredictionConfig = PredictionConfig(
    prediction_mode="future",
    train_end_date="2024-11-26",
    future_start_date="2024-11-27",
    future_end_date="2024-12-10"
)
```

### Customize Model Parameters

Adjust model hyperparameters in `config.py`:

```python
# ARIMA order
arima: ARIMAConfig = ARIMAConfig(order=(5, 1, 0))

# Machine Learning settings
ml: MLConfig = MLConfig(
    enabled=True,
    n_estimators=100,
    lags=5
)

# Support/Resistance window
support_resistance: SupportResistanceConfig = SupportResistanceConfig(
    enabled=True,
    window_size=10
)
```

## Project Structure

```
nifty-50-predictor/
├── data/                              # Historical data
│   ├── nifty_50_1y.csv               # 1-year historical data
│   └── Nifty 50 Historical Data.csv   # Additional historical data
├── outputs/                           # Generated plots and results
├── config.py                          # Configuration management
├── data_handler.py                    # Data loading and preprocessing
├── models.py                          # Prediction models (ARIMA, GARCH, RF, Ensemble)
├── visualize.py                       # Visualization utilities
├── main.py                            # Main entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── .gitignore                         # Git ignore rules
```

## Modules Overview

### `config.py`
Centralized configuration management with dataclasses:
- `DataConfig`: Data source settings
- `ARIMAConfig`: ARIMA model parameters
- `GARCHConfig`: GARCH model parameters
- `MLConfig`: Machine Learning settings
- `PredictionConfig`: Prediction mode and date range
- `VisualizationConfig`: Plot styling and output

### `data_handler.py`
Handles all data operations:
- **DataHandler class**: Load, preprocess, split data
- Automatic date parsing and frequency setting
- Lagged feature generation for ML
- Returns indicator calculation

### `models.py`
Prediction models with clean API:
- **ARIMAModel**: Time series forecasting
- **GARCHModel**: Volatility estimation
- **RandomForestModel**: Residual learning
- **EnsemblePredictor**: Combines all three models
- **calculate_support_resistance()**: Technical analysis

### `visualize.py`
Professional visualization tools:
- **Plotter class**: Create publication-ready plots
- Ensemble predictions with split markers
- Support/resistance zones
- Component analysis charts

### `main.py`
Orchestrates the full pipeline:
- Mode selection (test or future)
- Data loading and preprocessing
- Model training and prediction
- Evaluation and visualization

## Model Architecture

### Ensemble Strategy

```
1. ARIMA Forecast (Trend Component)
         ↓
2. GARCH Volatility (Volatility Component)
         ↓
3. Random Forest (Residual Correction)
         ↓
4. Ensemble Prediction = ARIMA + GARCH + RF
```

### Why This Approach?

- **ARIMA**: Excellent for trending, stationary/non-stationary patterns
- **GARCH**: Captures changing volatility (common in financial data)
- **Random Forest**: Learns non-linear relationships in residuals
- **Ensemble**: Combines strengths, reduces individual model biases

## Results & Evaluation

The model provides:

- **RMSE** (Root Mean Squared Error): Penalty for large errors
- **MAE** (Mean Absolute Error): Average absolute deviation
- **MAPE** (Mean Absolute Percentage Error): Percentage error

Lower values indicate better predictions.

## Example Output

```
Loading data from data/nifty_50_1y.csv
Data loaded: 252 rows, 6 columns
Preprocessing data...
Preprocessing complete: 252 valid close prices
Splitting data at index: 201
Train set: 201 samples, Test set: 51 samples

Fitting ARIMA(5, 1, 0)...
ARIMA model fitted successfully
Fitting GARCH(1,1)...
GARCH model fitted successfully
Fitting Random Forest with 100 estimators...
Random Forest model fitted successfully

Generating predictions for 51 steps...
Predictions generated: mean=23450.50, std=120.30

Model Evaluation - RMSE: 125.45, MAE: 98.67, MAPE: 0.42%
```

## Performance Tips

1. **Feature Engineering**: Add more lags or technical indicators in `data_handler.py`
2. **Hyperparameter Tuning**: Modify `config.py` values based on backtesting
3. **Data Quality**: Ensure CSV has columns: Date, Close (and others if needed)
4. **Frequency**: Set `asfreq('B')` for business days, adjust if needed
5. **ML Lags**: Increase `n_lags` for complex patterns (trades computational cost)

## Limitations & Disclaimers

⚠️ **Important**:
- Past performance does not guarantee future results
- Financial markets are influenced by unpredictable events
- Use predictions as one input among multiple factors
- Always validate with domain experts
- **Not financial advice** - consult professionals for investment decisions

## Contributing

Contributions welcome! Areas for enhancement:

- [ ] Add SARIMA for seasonal patterns
- [ ] Implement attention mechanisms for better feature weighting
- [ ] Add cross-validation framework
- [ ] Support multiple indices (Bank Nifty, Sensex, etc.)
- [ ] Web interface for easy access
- [ ] Real-time prediction pipeline

## License

MIT License - See LICENSE file for details

## Author

Created as a financial machine learning project for Nifty 50 index forecasting.

## Resources

- [ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [GARCH Models](https://arch.readthedocs.io/)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Time Series Forecasting](https://otexts.com/fpp2/)

---

**Last Updated**: June 2026
**Status**: Active Development
