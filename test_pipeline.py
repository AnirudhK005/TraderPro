"""
Test the full TraderPro pipeline:
Data Collection → Feature Engineering → Target Engineering → Backtest
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.collector import DataCollector
from src.features.engineering import FeatureEngineer
from src.features.target import TargetEngineer
from src.backtest.walkforward import WalkForwardBacktester, BacktestConfig


def load_and_prepare_data(tickers: list, data_dir: str = "data") -> pd.DataFrame:
    """Load data for multiple tickers and combine into a single DataFrame."""
    all_data = []
    
    for ticker in tickers:
        filepath = Path(data_dir) / f"{ticker}.csv"
        if not filepath.exists():
            print(f"  Skipping {ticker} - no data file")
            continue
        
        # Load raw data
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df = df.reset_index()
        df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['ticker'] = ticker
        
        # Apply feature engineering
        fe = FeatureEngineer(lag=1)
        df_features = fe.compute_features(df.set_index('date')).reset_index()
        df_features.rename(columns={'index': 'date'}, inplace=True)
        df_features['ticker'] = ticker
        
        # Apply target engineering
        te = TargetEngineer(horizon=5)
        df_final = te.add_target_to_features(df_features)
        
        all_data.append(df_final)
        print(f"  Processed {ticker}: {len(df_final)} rows")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    return combined


def main():
    print("=" * 60)
    print("TRADERPRO PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Check if we have data, if not collect it
    data_dir = Path("data")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ"]
    
    existing_files = list(data_dir.glob("*.csv"))
    if len(existing_files) < 5:
        print("\n[1/4] Collecting data...")
        collector = DataCollector()
        collector.tickers = test_tickers
        collector.collect(period="5y")
    else:
        print(f"\n[1/4] Using existing data ({len(existing_files)} files)")
    
    # Step 2: Load and prepare data
    print("\n[2/4] Processing data (features + targets)...")
    df = load_and_prepare_data(test_tickers)
    print(f"\nCombined dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Tickers: {df['ticker'].nunique()}")
    
    # Step 3: Verify data quality
    print("\n[3/4] Data quality check...")
    print(f"  Rows with NaN target: {df['target'].isna().sum()}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Check feature availability
    feature_cols = [c for c in df.columns if c not in 
                    ['date', 'ticker', 'target', 'forward_return', 
                     'Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"  Feature columns: {len(feature_cols)}")
    
    # Step 4: Run backtest
    print("\n[4/4] Running walk-forward backtest...")
    config = BacktestConfig(
        initial_capital=100000,
        min_probability=0.52,  # Lower threshold for testing
        n_folds=3,  # Fewer folds for faster testing
        min_training_days=200,  # Slightly less for 5-year data
    )
    
    backtester = WalkForwardBacktester(config)
    results = backtester.run(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()

