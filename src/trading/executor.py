"""
Trade executor for TraderPro.
Manages order execution and position management.
"""
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

from src.trading.alpaca_client import AlpacaClient


class TradeExecutor:
    """Executes trades based on signals."""
    
    def __init__(
        self,
        client: AlpacaClient,
        max_positions: int = 10,
        log_dir: str = "logs"
    ):
        """
        Initialize trade executor.
        
        Args:
            client: Alpaca client instance
            max_positions: Maximum number of positions to hold
            log_dir: Directory to save trade logs
        """
        self.client = client
        self.max_positions = max_positions
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Trade log for the day
        self.daily_trades = []
    
    def execute_signals(
        self,
        signals: List[Dict],
        dry_run: bool = False
    ) -> Dict:
        """
        Execute trading signals.
        
        Args:
            signals: List of signal dicts from SignalGenerator
            dry_run: If True, don't actually execute trades
        
        Returns:
            Summary of execution results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'buys': [],
            'sells': [],
            'holds': [],
            'errors': []
        }
        
        # Get current positions
        positions = self.client.get_positions()
        current_symbols = set(positions.keys())
        account = self.client.get_account()
        
        print(f"\n{'='*50}")
        print(f"EXECUTING SIGNALS - {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"{'='*50}")
        print(f"Account Equity: ${account['equity']:,.2f}")
        print(f"Cash Available: ${account['cash']:,.2f}")
        print(f"Current Positions: {len(current_symbols)}/{self.max_positions}")
        print(f"Signals to Process: {len(signals)}")
        print("-" * 50)
        
        # Process SELL signals first (free up capital)
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        for signal in sell_signals:
            ticker = signal['ticker']
            
            if ticker not in current_symbols:
                continue  # Don't short, only close existing positions
            
            print(f"SELL {ticker} @ ${signal['price']:.2f} (prob: {signal['probability']:.2%})")
            
            if not dry_run:
                try:
                    result = self.client.close_position(ticker)
                    results['sells'].append({
                        'ticker': ticker,
                        'price': signal['price'],
                        'probability': signal['probability'],
                        'order': result
                    })
                except Exception as e:
                    results['errors'].append(f"SELL {ticker}: {str(e)}")
            else:
                results['sells'].append({
                    'ticker': ticker,
                    'price': signal['price'],
                    'probability': signal['probability'],
                    'order': 'DRY_RUN'
                })
        
        # Process BUY signals
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        
        # Only buy if we have room for more positions
        available_slots = self.max_positions - len(current_symbols) + len(results['sells'])
        buy_signals = buy_signals[:available_slots]  # Limit to available slots
        
        for signal in buy_signals:
            ticker = signal['ticker']
            
            if ticker in current_symbols:
                results['holds'].append(signal)
                continue  # Already have position
            
            # Calculate position size
            shares = self.client.calculate_position_size(signal['price'])
            
            print(f"BUY {ticker} x{shares} @ ${signal['price']:.2f} (prob: {signal['probability']:.2%})")
            
            if not dry_run:
                try:
                    result = self.client.submit_market_order(
                        symbol=ticker,
                        qty=shares,
                        side='buy'
                    )
                    results['buys'].append({
                        'ticker': ticker,
                        'shares': shares,
                        'price': signal['price'],
                        'probability': signal['probability'],
                        'order': result
                    })
                except Exception as e:
                    results['errors'].append(f"BUY {ticker}: {str(e)}")
            else:
                results['buys'].append({
                    'ticker': ticker,
                    'shares': shares,
                    'price': signal['price'],
                    'probability': signal['probability'],
                    'order': 'DRY_RUN'
                })
        
        # Summary
        print("-" * 50)
        print(f"Buys Executed: {len(results['buys'])}")
        print(f"Sells Executed: {len(results['sells'])}")
        print(f"Positions Held: {len(results['holds'])}")
        print(f"Errors: {len(results['errors'])}")
        
        # Log results
        self._log_trades(results)
        
        return results
    
    def close_losing_positions(
        self,
        loss_threshold: float = -0.05,
        dry_run: bool = False
    ) -> List[Dict]:
        """
        Close positions that have exceeded loss threshold.
        
        Args:
            loss_threshold: Maximum acceptable loss (e.g., -0.05 = -5%)
            dry_run: If True, don't actually close positions
        
        Returns:
            List of closed positions
        """
        positions = self.client.get_positions()
        closed = []
        
        for symbol, pos in positions.items():
            if pos['unrealized_plpc'] < loss_threshold:
                print(f"STOP LOSS: {symbol} at {pos['unrealized_plpc']:.2%}")
                
                if not dry_run:
                    result = self.client.close_position(symbol)
                    if result:
                        closed.append({
                            'symbol': symbol,
                            'loss_pct': pos['unrealized_plpc'],
                            'order': result
                        })
                else:
                    closed.append({
                        'symbol': symbol,
                        'loss_pct': pos['unrealized_plpc'],
                        'order': 'DRY_RUN'
                    })
        
        return closed
    
    def take_profits(
        self,
        profit_threshold: float = 0.10,
        dry_run: bool = False
    ) -> List[Dict]:
        """
        Close positions that have exceeded profit threshold.
        
        Args:
            profit_threshold: Target profit (e.g., 0.10 = 10%)
            dry_run: If True, don't actually close positions
        
        Returns:
            List of closed positions
        """
        positions = self.client.get_positions()
        closed = []
        
        for symbol, pos in positions.items():
            if pos['unrealized_plpc'] > profit_threshold:
                print(f"TAKE PROFIT: {symbol} at {pos['unrealized_plpc']:.2%}")
                
                if not dry_run:
                    result = self.client.close_position(symbol)
                    if result:
                        closed.append({
                            'symbol': symbol,
                            'profit_pct': pos['unrealized_plpc'],
                            'order': result
                        })
                else:
                    closed.append({
                        'symbol': symbol,
                        'profit_pct': pos['unrealized_plpc'],
                        'order': 'DRY_RUN'
                    })
        
        return closed
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of current portfolio."""
        account = self.client.get_account()
        positions = self.client.get_positions()
        
        total_pl = sum(p['unrealized_pl'] for p in positions.values())
        
        return {
            'equity': account['equity'],
            'cash': account['cash'],
            'positions_count': len(positions),
            'positions': positions,
            'unrealized_pl': total_pl,
            'unrealized_pl_pct': total_pl / account['equity'] if account['equity'] > 0 else 0
        }
    
    def _log_trades(self, results: Dict) -> None:
        """Save trade results to log file."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"trades_{date_str}.json"
        
        # Load existing log if present
        if log_file.exists():
            with open(log_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = []
        
        # Append new results
        existing.append(results)
        
        # Save
        with open(log_file, 'w') as f:
            json.dump(existing, f, indent=2, default=str)


if __name__ == "__main__":
    print("TradeExecutor module loaded successfully")

