"""
Target engineering for trading signals.
Creates prediction targets from price data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class TargetEngineer:
    """Creates prediction targets for model training."""
    
    def __init__(self, horizon: int = 5, threshold: float = 0.0):
        """
        Args:
            horizon: Number of days forward to predict (default 5)
            threshold: Minimum return to be considered positive (default 0.0)
        """
        self.horizon = horizon
        self.threshold = threshold
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target: 1 if forward return > threshold, else 0.
        
        Args:
            df: DataFrame with Close prices
        
        Returns:
            DataFrame with target column added
        """
        df = df.copy()
        
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")
        
        # Calculate forward return (future data - this is intentional for target)
        future_price = df['Close'].shift(-self.horizon)
        forward_return = (future_price - df['Close']) / df['Close']
        
        # Binary target
        df['target'] = (forward_return > self.threshold).astype(int)
        
        # Also store the actual return for analysis
        df['forward_return'] = forward_return
        
        return df
    
    def add_target_to_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target to a DataFrame that already has features.
        Drops rows with NaN targets (last 'horizon' rows).
        
        Args:
            df: DataFrame with features and Close prices
        
        Returns:
            DataFrame with target, NaN rows dropped
        """
        df = self.create_target(df)
        
        # Drop rows where target is NaN (last 'horizon' rows)
        df = df.dropna(subset=['target'])
        
        return df
    
    def get_target_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about the target distribution.
        
        Args:
            df: DataFrame with target column
        
        Returns:
            Dict with target statistics
        """
        if 'target' not in df.columns:
            raise ValueError("DataFrame must have 'target' column")
        
        total = len(df)
        positive = df['target'].sum()
        negative = total - positive
        
        return {
            'total_samples': total,
            'positive': int(positive),
            'negative': int(negative),
            'positive_rate': positive / total if total > 0 else 0,
            'negative_rate': negative / total if total > 0 else 0,
        }
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Add target to a CSV file.
        
        Args:
            input_path: Path to input CSV (with features)
            output_path: Path to save output (optional)
        
        Returns:
            DataFrame with target
        """
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        df = self.add_target_to_features(df)
        
        if output_path:
            df.to_csv(output_path)
            print(f"Saved with target -> {output_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    from src.features.engineering import FeatureEngineer
    
    # Load raw data
    df = pd.read_csv("data/AAPL.csv", index_col=0, parse_dates=True)
    
    # Add features
    feature_eng = FeatureEngineer(lag=1)
    df = feature_eng.compute_features(df)
    
    # Add target
    target_eng = TargetEngineer(horizon=5)
    df = target_eng.add_target_to_features(df)
    
    # Show stats
    stats = target_eng.get_target_stats(df)
    print(f"Target stats: {stats}")

