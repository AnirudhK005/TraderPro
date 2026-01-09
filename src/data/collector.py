"""
Data collector for S&P500 equities.
Fetches historical OHLCV data from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional


class DataCollector:
    """Collects historical price data for stocks."""
    
    def __init__(self, config_path: str = "configs/tickers.yaml", data_dir: str = "data"):
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load ticker configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def fetch_ticker(self, ticker: str, period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Stock symbol
            period: Time period (default from config)
        
        Returns:
            DataFrame with OHLCV data
        """
        period = period or self.config.get('period', '5y')
        print(f"Fetching {ticker}...")
        
        data = yf.download(ticker, period=period, progress=False)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data
    
    def save_ticker(self, ticker: str, data: pd.DataFrame) -> Path:
        """
        Save ticker data to CSV.
        
        Args:
            ticker: Stock symbol
            data: DataFrame to save
        
        Returns:
            Path to saved file
        """
        filepath = self.data_dir / f"{ticker}.csv"
        data.to_csv(filepath)
        print(f"  Saved -> {filepath}")
        return filepath
    
    def collect(self, tickers: Optional[List[str]] = None, period: Optional[str] = None) -> dict:
        """
        Fetch and save data for multiple tickers.
        
        Args:
            tickers: List of tickers (default from config)
            period: Time period (default from config)
        
        Returns:
            Dict with success/failure counts
        """
        tickers = tickers or self.config.get('tickers', [])
        results = {'success': 0, 'failed': 0, 'errors': []}
        
        print(f"Collecting data for {len(tickers)} tickers...")
        print("-" * 40)
        
        for ticker in tickers:
            try:
                data = self.fetch_ticker(ticker, period)
                if not data.empty:
                    self.save_ticker(ticker, data)
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{ticker}: Empty data")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{ticker}: {str(e)}")
                print(f"  Error: {e}")
        
        print("-" * 40)
        print(f"Complete: {results['success']} success, {results['failed']} failed")
        
        return results


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect()

