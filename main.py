"""Main script for Nifty 50 prediction."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import get_config
from data_handler import DataHandler
from models import EnsemblePredictor, calculate_support_resistance
from visualize import Plotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_test_set(config) -> dict:
    """Predict on test set with 80-20 split."""
    logger.info("="*60)
    logger.info("MODE: Test Set Prediction (80-20 Split)")
    logger.info("="*60)
    
    # Load and preprocess data
    data_handler = DataHandler(config.data.data_dir, config.data.csv_file)
    data = data_handler.load_data()
    close_prices = data_handler.preprocess(config.data.date_column, config.data.close_column)
    
    # Split data
    train, test = data_handler.split_data(
        close_prices,
        train_test_split=config.prediction.train_test_split
    )
    
    # Prepare ML features
    ml_features, ml_target = data_handler.get_lagged_features(train, lags=config.ml.lags)
    
    # Fit ensemble model
    predictor = EnsemblePredictor(
        arima_order=config.arima.order,
        garch_p=config.garch.p,
        garch_q=config.garch.q,
        use_ml=config.ml.enabled,
        n_lags=config.ml.lags
    )
    
    predictor.fit(train, ml_features, ml_target)
    
    # Predict on test set
    test_ml_features, _ = data_handler.get_lagged_features(test, lags=config.ml.lags)
    ensemble_pred, components = predictor.predict(len(test), test_ml_features)
    
    # Evaluate
    metrics = predictor.arima.evaluate(test.iloc[config.ml.lags:].values, ensemble_pred[:len(test)-config.ml.lags])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': test.index[config.ml.lags:],
        'Actual': test.iloc[config.ml.lags:].values,
        'Predicted': ensemble_pred[:len(test)-config.ml.lags]
    })
    
    return {
        'results': results,
        'metrics': metrics,
        'components': components,
        'train_end_date': train.index[-1],
        'ensemble_pred': ensemble_pred,
        'test_set': test
    }


def predict_future(config) -> dict:
    """Predict future prices."""
    logger.info("="*60)
    logger.info("MODE: Future Prediction")
    logger.info("="*60)
    
    # Load and preprocess data
    data_handler = DataHandler(config.data.data_dir, config.data.csv_file)
    data = data_handler.load_data()
    close_prices = data_handler.preprocess(config.data.date_column, config.data.close_column)
    
    # Split data at specified date
    train, _ = data_handler.split_data(
        close_prices,
        train_end_date=config.prediction.train_end_date
    )
    
    logger.info(f"Training data until {config.prediction.train_end_date}")
    
    # Prepare ML features
    ml_features, ml_target = data_handler.get_lagged_features(train, lags=config.ml.lags)
    
    # Fit ensemble model
    predictor = EnsemblePredictor(
        arima_order=config.arima.order,
        garch_p=config.garch.p,
        garch_q=config.garch.q,
        use_ml=config.ml.enabled,
        n_lags=config.ml.lags
    )
    
    predictor.fit(train, ml_features, ml_target)
    
    # Generate future dates
    future_start = pd.to_datetime(config.prediction.future_start_date)
    future_end = pd.to_datetime(config.prediction.future_end_date)
    future_dates = pd.date_range(start=future_start, end=future_end, freq='B')
    
    # Predict
    ensemble_pred, components = predictor.predict(len(future_dates), None)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': future_dates,
        'Predicted': ensemble_pred
    })
    
    return {
        'results': results,
        'components': components,
        'ensemble_pred': ensemble_pred,
        'future_dates': future_dates,
        'close_prices': close_prices
    }


def main():
    """Main execution function."""
    logger.info("Starting Nifty 50 Predictor")
    
    # Get configuration
    config = get_config()
    
    try:
        # Choose prediction mode
        if config.prediction.prediction_mode == "test":
            output = predict_test_set(config)
            
            # Print results
            logger.info("\n" + "="*60)
            logger.info("PREDICTION RESULTS")
            logger.info("="*60)
            print(output['results'].head(10))
            print(f"\nMetrics: {output['metrics']}")
            
            # Visualize
            plotter = Plotter(
                figsize=config.visualization.figsize,
                dpi=config.visualization.dpi
            )
            
            split_date = output['train_end_date']
            combined_data = pd.concat([
                output['test_set'],
                pd.Series(output['ensemble_pred'][:len(output['test_set'])], index=output['test_set'].index)
            ])
            
            plotter.plot_ensemble_predictions(
                actual_data=output['test_set'],
                predictions=pd.Series(output['ensemble_pred'][:len(output['test_set'])], index=output['test_set'].index),
                split_date=split_date,
                title="Nifty 50 Test Set Prediction",
                save_path=config.visualization.plot_dir / "test_prediction.png" if config.visualization.save_plot else None
            )
        
        elif config.prediction.prediction_mode == "future":
            output = predict_future(config)
            
            # Print results
            logger.info("\n" + "="*60)
            logger.info("FUTURE PREDICTIONS")
            logger.info("="*60)
            print(output['results'])
            
            # Visualize
            plotter = Plotter(
                figsize=config.visualization.figsize,
                dpi=config.visualization.dpi
            )
            
            plotter.plot_ensemble_predictions(
                actual_data=output['close_prices'].tail(50),
                predictions=output['results'].set_index('Date')['Predicted'],
                split_date=output['close_prices'].index[-1],
                title="Nifty 50 Future Prediction",
                save_path=config.visualization.plot_dir / "future_prediction.png" if config.visualization.save_plot else None
            )
        
        logger.info("✅ Prediction completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
