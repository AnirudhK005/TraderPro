"""
Feature engineering for trading signals.
All features are lagged by 1 day to prevent look-ahead bias.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class FeatureEngineer:
    """Computes technical indicators and features for trading."""
    
    def __init__(self, lag: int = 1):
        """
        Args:
            lag: Number of days to lag features (default 1 to prevent look-ahead)
        """
        self.lag = lag
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a price DataFrame.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with original data plus features
        """
        df = df.copy()
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Compute all features
        df = self._add_sma(df)
        df = self._add_ema(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger(df)
        df = self._add_atr(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume_features(df)
        
        # Apply lag to all computed features (not OHLCV)
        df = self._apply_lag(df, required)
        
        return df
    
    def _add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple Moving Averages."""
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            # Price relative to SMA
            df[f'close_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        return df
    
    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Averages."""
        for period in [10, 20]:
            df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD indicator."""
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def _add_bollinger(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=period).mean()
        df['atr_ratio'] = df['atr_14'] / df['Close']
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price momentum."""
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling volatility."""
        returns = df['Close'].pct_change()
        for period in [10, 20]:
            df[f'volatility_{period}'] = returns.rolling(window=period).std()
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        return df
    
    def _apply_lag(self, df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
        """
        Apply lag to all feature columns to prevent look-ahead bias.
        
        Args:
            df: DataFrame with features
            exclude_cols: Columns to not lag (original OHLCV)
        """
        if self.lag == 0:
            return df
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        for col in feature_cols:
            df[col] = df[col].shift(self.lag)
        
        return df
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process a single CSV file.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to save output (optional)
        
        Returns:
            DataFrame with features
        """
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        df = self.compute_features(df)
        
        if output_path:
            df.to_csv(output_path)
            print(f"Saved features -> {output_path}")
        
        return df
    
    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        """
        Process all CSV files in a directory.
        
        Args:
            input_dir: Directory with raw price CSVs
            output_dir: Directory to save feature CSVs
        
        Returns:
            Dict with success/failure counts
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {'success': 0, 'failed': 0, 'errors': []}
        csv_files = list(input_path.glob("*.csv"))
        
        print(f"Processing {len(csv_files)} files...")
        print("-" * 40)
        
        for csv_file in csv_files:
            ticker = csv_file.stem
            try:
                output_file = output_path / f"{ticker}_features.csv"
                self.process_file(str(csv_file), str(output_file))
                results['success'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{ticker}: {str(e)}")
                print(f"Error processing {ticker}: {e}")
        
        print("-" * 40)
        print(f"Complete: {results['success']} success, {results['failed']} failed")
        
        return results


if __name__ == "__main__":
    engineer = FeatureEngineer(lag=1)
    engineer.process_directory("data", "data/features")

