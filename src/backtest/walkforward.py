"""
Walk-forward backtester with integrated training.
Trains model within each fold to prevent look-ahead bias.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb


@dataclass
class BacktestConfig:
    """Backtesting configuration with realistic defaults."""
    # Portfolio
    initial_capital: float = 100000.0
    
    # Costs (realistic)
    commission_per_share: float = 0.01
    slippage_bps: float = 10.0  # 10 basis points
    
    # Risk Management
    risk_per_trade: float = 0.01  # 1% of portfolio per trade
    max_open_positions: int = 10
    
    # Trade Management
    stop_loss_atr: float = 1.5  # Stop at 1.5x ATR below entry
    take_profit_atr: float = 3.0  # Target at 3x ATR above entry
    max_holding_days: int = 10
    
    # Signal Threshold
    min_probability: float = 0.55
    
    # Walk-forward Settings
    n_folds: int = 4
    min_training_days: int = 252  # 1 year minimum training
    embargo_days: int = 5  # Gap between train and validation


@dataclass
class Trade:
    """Record of a single trade."""
    ticker: str
    fold: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    stop_price: float = 0.0
    target_price: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    probability: float = 0.0


class WalkForwardBacktester:
    """
    Walk-forward backtester that trains models within each fold.
    This ensures no look-ahead bias in model training.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.model = None
        self.feature_cols = None
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
    
    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run walk-forward backtest.
        
        Args:
            df: DataFrame with features, target, and OHLCV data
                Must have columns: date, ticker, target, Close, atr_14
        
        Returns:
            Dict with results
        """
        print("=" * 60)
        print("WALK-FORWARD BACKTEST")
        print("=" * 60)
        
        # Validate data
        required_cols = ['date', 'ticker', 'target', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'ticker'])
        
        # Create folds
        folds = self._create_folds(df)
        print(f"Created {len(folds)} folds")
        
        # Initialize portfolio
        cash = self.config.initial_capital
        positions = {}  # ticker -> Trade
        
        # Run each fold
        for fold in folds:
            print(f"\n--- {fold['name']} ---")
            
            # Get training data (before validation, with embargo)
            train_df = df[
                (df['date'] >= fold['train_start']) &
                (df['date'] <= fold['train_end'])
            ]
            
            # Get validation data
            val_df = df[
                (df['date'] >= fold['val_start']) &
                (df['date'] <= fold['val_end'])
            ]
            
            print(f"Train: {fold['train_start'].date()} to {fold['train_end'].date()} ({len(train_df)} rows)")
            print(f"Val: {fold['val_start'].date()} to {fold['val_end'].date()} ({len(val_df)} rows)")
            
            if len(train_df) < 100:
                print("Skipping - insufficient training data")
                continue
            
            # Train model on this fold's training data
            self._train_model(train_df)
            
            # Run backtest on validation data
            cash, positions = self._run_fold(val_df, fold['name'], cash, positions)
        
        # Close any remaining positions
        self._close_all_positions(positions, df['date'].max())
        
        # Calculate results
        results = self._calculate_results()
        self._print_results(results)
        
        return results
    
    def _create_folds(self, df: pd.DataFrame) -> List[Dict]:
        """Create walk-forward folds."""
        dates = sorted(df['date'].unique())
        
        # Need minimum training data before first fold
        first_val_idx = self.config.min_training_days
        if first_val_idx >= len(dates):
            raise ValueError("Not enough data for walk-forward validation")
        
        # Divide remaining dates into folds
        remaining_dates = dates[first_val_idx:]
        dates_per_fold = len(remaining_dates) // self.config.n_folds
        
        folds = []
        for i in range(self.config.n_folds):
            start_idx = i * dates_per_fold
            end_idx = (i + 1) * dates_per_fold if i < self.config.n_folds - 1 else len(remaining_dates)
            
            val_start = remaining_dates[start_idx]
            val_end = remaining_dates[end_idx - 1]
            
            # Training ends with embargo before validation
            train_end_idx = first_val_idx + start_idx - self.config.embargo_days
            train_end = dates[max(0, train_end_idx)]
            train_start = dates[0]
            
            folds.append({
                'name': f'fold_{i}',
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
            })
        
        return folds
    
    def _train_model(self, train_df: pd.DataFrame) -> None:
        """Train ensemble model on training data."""
        # Identify feature columns
        exclude = ['date', 'ticker', 'target', 'forward_return', 
                   'Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_cols = [c for c in train_df.columns if c not in exclude]
        
        X = train_df[self.feature_cols].fillna(0)
        y = train_df['target'].astype(int)
        
        # Create ensemble
        xgb_model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('rf', rf_model)],
            voting='soft'
        )
        
        self.model.fit(X, y)
        
        # Quick CV check
        cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='accuracy')
        print(f"  Model CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    def _run_fold(self, val_df: pd.DataFrame, fold_name: str,
                  cash: float, positions: Dict) -> Tuple[float, Dict]:
        """Run backtest on a single fold."""
        
        dates = sorted(val_df['date'].unique())
        
        for date in dates:
            day_data = val_df[val_df['date'] == date]
            
            # Update portfolio value
            portfolio_value = cash
            for ticker, trade in positions.items():
                current_price = self._get_price(day_data, ticker, trade.entry_price)
                portfolio_value += trade.shares * current_price
            
            self.portfolio_history.append({
                'date': date,
                'cash': cash,
                'portfolio_value': portfolio_value,
                'n_positions': len(positions)
            })
            
            # Check existing positions for exit
            tickers_to_close = []
            for ticker, trade in positions.items():
                ticker_data = day_data[day_data['ticker'] == ticker]
                if len(ticker_data) == 0:
                    continue
                
                row = ticker_data.iloc[0]
                current_price = row['Close']
                holding_days = (date - trade.entry_date).days
                
                # Check stop loss
                if current_price <= trade.stop_price:
                    trade.exit_date = date
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = 'stop_loss'
                    tickers_to_close.append(ticker)
                
                # Check take profit
                elif current_price >= trade.target_price:
                    trade.exit_date = date
                    trade.exit_price = trade.target_price
                    trade.exit_reason = 'take_profit'
                    tickers_to_close.append(ticker)
                
                # Check max holding
                elif holding_days >= self.config.max_holding_days:
                    trade.exit_date = date
                    trade.exit_price = current_price
                    trade.exit_reason = 'max_holding'
                    tickers_to_close.append(ticker)
            
            # Close positions
            for ticker in tickers_to_close:
                trade = positions.pop(ticker)
                cash = self._close_trade(trade, cash)
            
            # Look for new positions
            if len(positions) < self.config.max_open_positions:
                for _, row in day_data.iterrows():
                    if len(positions) >= self.config.max_open_positions:
                        break
                    
                    ticker = row['ticker']
                    if ticker in positions:
                        continue
                    
                    # Get prediction
                    prob = self._get_prediction(row)
                    
                    if prob >= self.config.min_probability:
                        # Calculate position size
                        trade = self._open_trade(row, fold_name, prob, cash, portfolio_value)
                        if trade:
                            positions[ticker] = trade
                            cash -= (trade.shares * trade.entry_price + trade.commission + trade.slippage)
        
        return cash, positions
    
    def _get_prediction(self, row: pd.Series) -> float:
        """Get model prediction probability."""
        if self.model is None or self.feature_cols is None:
            return 0.5
        
        features = row[self.feature_cols].fillna(0).values.reshape(1, -1)
        try:
            prob = self.model.predict_proba(features)[0][1]
            return prob
        except:
            return 0.5
    
    def _open_trade(self, row: pd.Series, fold: str, prob: float,
                    cash: float, portfolio_value: float) -> Optional[Trade]:
        """Open a new trade."""
        ticker = row['ticker']
        price = row['Close']
        atr = row.get('atr_14', price * 0.02)  # Default 2% if no ATR
        
        # Calculate position size based on risk
        stop_distance = atr * self.config.stop_loss_atr
        risk_amount = portfolio_value * self.config.risk_per_trade
        shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        
        if shares <= 0:
            return None
        
        # Calculate costs
        entry_price = price * (1 + self.config.slippage_bps / 10000)
        commission = shares * self.config.commission_per_share
        slippage = shares * price * (self.config.slippage_bps / 10000)
        total_cost = shares * entry_price + commission + slippage
        
        if total_cost > cash:
            # Reduce position to fit
            shares = int((cash - commission) / (entry_price * 1.001))
            if shares <= 0:
                return None
            commission = shares * self.config.commission_per_share
            slippage = shares * price * (self.config.slippage_bps / 10000)
        
        trade = Trade(
            ticker=ticker,
            fold=fold,
            entry_date=row['date'],
            entry_price=entry_price,
            shares=shares,
            stop_price=entry_price - stop_distance,
            target_price=entry_price + (atr * self.config.take_profit_atr),
            commission=commission,
            slippage=slippage,
            probability=prob
        )
        
        self.trades.append(trade)
        return trade
    
    def _close_trade(self, trade: Trade, cash: float) -> float:
        """Close a trade and return updated cash."""
        # Calculate P&L
        gross_pnl = (trade.exit_price - trade.entry_price) * trade.shares
        exit_commission = trade.shares * self.config.commission_per_share
        exit_slippage = trade.shares * trade.exit_price * (self.config.slippage_bps / 10000)
        
        trade.commission += exit_commission
        trade.slippage += exit_slippage
        trade.pnl = gross_pnl - trade.commission - trade.slippage
        
        return cash + (trade.shares * trade.exit_price) - exit_commission - exit_slippage
    
    def _close_all_positions(self, positions: Dict, final_date: datetime) -> None:
        """Close all remaining positions at the end."""
        for ticker, trade in positions.items():
            trade.exit_date = final_date
            trade.exit_price = trade.entry_price  # Assume flat if no data
            trade.exit_reason = 'end_of_backtest'
            trade.pnl = -trade.commission - trade.slippage
    
    def _get_price(self, day_data: pd.DataFrame, ticker: str, default: float) -> float:
        """Get current price for a ticker."""
        ticker_data = day_data[day_data['ticker'] == ticker]
        if len(ticker_data) > 0:
            return ticker_data.iloc[0]['Close']
        return default
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Basic metrics
        initial = self.config.initial_capital
        final = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final - initial) / initial
        
        # Returns for Sharpe
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        returns = portfolio_df['returns'].dropna()
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        
        # Drawdown
        peak = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade stats
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        n_trades = len(completed_trades)
        
        if n_trades > 0:
            wins = [t for t in completed_trades if t.pnl > 0]
            losses = [t for t in completed_trades if t.pnl <= 0]
            win_rate = len(wins) / n_trades
            
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
            profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if losses and avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_value': final,
            'portfolio_df': portfolio_df,
            'trades': self.trades
        }
    
    def _print_results(self, results: Dict) -> None:
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['n_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")

