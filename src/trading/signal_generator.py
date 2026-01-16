"""
Signal generator for TraderPro.
Uses the trained ensemble model to generate trading signals.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from src.features.engineering import FeatureEngineer
from src.features.target import TargetEngineer


class SignalGenerator:
    """Generates trading signals from price data using ML model."""
    
    def __init__(
        self,
        min_probability: float = 0.52,
        model_path: Optional[str] = None
    ):
        """
        Initialize signal generator.
        
        Args:
            min_probability: Minimum prediction probability to generate signal
            model_path: Path to saved model (optional, will train if not provided)
        """
        self.min_probability = min_probability
        self.feature_engineer = FeatureEngineer(lag=1)
        self.target_engineer = TargetEngineer(horizon=5, threshold=0.01)
        self.model = None
        self.feature_cols = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train_model(self, df: pd.DataFrame) -> None:
        """
        Train the ensemble model on historical data.
        
        Args:
            df: DataFrame with features and target column
        """
        # Identify feature columns
        exclude_cols = ['date', 'ticker', 'target', 'forward_return',
                        'Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Prepare training data
        train_df = df.dropna(subset=self.feature_cols + ['target'])
        X = train_df[self.feature_cols]
        y = train_df['target']
        
        print(f"Training on {len(X)} samples with {len(self.feature_cols)} features...")
        
        # Create ensemble model
        self.model = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ],
            voting='soft'
        )
        
        self.model.fit(X, y)
        print("Model trained successfully!")
    
    def save_model(self, path: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train first!")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_cols = data['feature_cols']
        print(f"Model loaded from {path}")
    
    def prepare_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare features for multiple tickers.
        
        Args:
            price_data: Dict mapping ticker -> OHLCV DataFrame
        
        Returns:
            Combined DataFrame with features
        """
        all_data = []
        
        for ticker, df in price_data.items():
            if df.empty:
                continue
            
            # Reset index to get date column
            df = df.reset_index()
            df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['ticker'] = ticker
            
            # Compute features
            df_features = self.feature_engineer.compute_features(
                df.set_index('date')
            ).reset_index()
            df_features.rename(columns={'index': 'date'}, inplace=True)
            df_features['ticker'] = ticker
            
            all_data.append(df_features)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    def generate_signals(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Generate trading signals for multiple tickers.
        
        Args:
            price_data: Dict mapping ticker -> OHLCV DataFrame
        
        Returns:
            List of signal dicts with ticker, signal, probability, price
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first!")
        
        # Prepare features
        df = self.prepare_features(price_data)
        
        if df.empty:
            return []
        
        # Get the most recent data for each ticker
        latest = df.groupby('ticker').last().reset_index()
        
        # Check we have all required features
        missing_features = [c for c in self.feature_cols if c not in latest.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features[:5]}...")
            return []
        
        # Filter out rows with NaN features
        latest_clean = latest.dropna(subset=self.feature_cols)
        
        if latest_clean.empty:
            return []
        
        # Generate predictions
        X = latest_clean[self.feature_cols]
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (up)
        
        # Build signals
        signals = []
        for i, row in latest_clean.iterrows():
            ticker = row['ticker']
            prob = probabilities[latest_clean.index.get_loc(i)]
            price = row['Close']
            
            # Determine signal
            if prob >= self.min_probability:
                signal = 'BUY'
            elif prob <= (1 - self.min_probability):
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            signals.append({
                'ticker': ticker,
                'signal': signal,
                'probability': round(prob, 4),
                'price': round(price, 2),
                'date': row['date']
            })
        
        # Sort by probability (strongest signals first)
        signals.sort(key=lambda x: abs(x['probability'] - 0.5), reverse=True)
        
        return signals
    
    def get_top_signals(
        self,
        signals: List[Dict],
        n: int = 10,
        signal_type: str = 'BUY'
    ) -> List[Dict]:
        """
        Get top N signals of a specific type.
        
        Args:
            signals: List of signal dicts
            n: Number of top signals to return
            signal_type: 'BUY' or 'SELL'
        
        Returns:
            Top N signals of the specified type
        """
        filtered = [s for s in signals if s['signal'] == signal_type]
        
        if signal_type == 'BUY':
            filtered.sort(key=lambda x: x['probability'], reverse=True)
        else:
            filtered.sort(key=lambda x: x['probability'])
        
        return filtered[:n]


if __name__ == "__main__":
    # Test signal generator
    generator = SignalGenerator(min_probability=0.52)
    print("SignalGenerator initialized successfully")
    print(f"Min probability threshold: {generator.min_probability}")

