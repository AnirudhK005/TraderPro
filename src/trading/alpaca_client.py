"""
Alpaca API client wrapper for TraderPro.
Handles connection, orders, and position management.
"""
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yaml

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed. Run: pip install alpaca-trade-api")


class AlpacaClient:
    """Wrapper for Alpaca Trading API."""
    
    def __init__(self, config_path: str = "configs/alpaca_config.yaml"):
        """
        Initialize Alpaca client from config file.
        
        Args:
            config_path: Path to YAML config with API credentials
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api not installed")
        
        self.config = self._load_config(config_path)
        
        # Initialize REST client
        self.api = tradeapi.REST(
            key_id=self.config['api_key'],
            secret_key=self.config['secret_key'],
            base_url=self.config['base_url']
        )
        
        # Trading settings
        self.max_positions = self.config.get('max_positions', 10)
        self.position_size_pct = self.config.get('position_size_pct', 0.10)
        self.min_probability = self.config.get('min_probability', 0.52)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Copy configs/alpaca_config.yaml.example to {config_path} "
                "and fill in your API keys."
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_account(self) -> dict:
        """Get account information."""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'status': account.status
        }
    
    def get_positions(self) -> Dict[str, dict]:
        """Get all current positions."""
        positions = self.api.list_positions()
        return {
            p.symbol: {
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'avg_entry': float(p.avg_entry_price),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            }
            for p in positions
        }
    
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        return positions.get(symbol)
    
    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str  # 'buy' or 'sell'
    ) -> dict:
        """
        Submit a market order.
        
        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'
        
        Returns:
            Order details dict
        """
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        
        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side,
            'status': order.status,
            'submitted_at': str(order.submitted_at)
        }
    
    def close_position(self, symbol: str) -> Optional[dict]:
        """Close entire position for a symbol."""
        try:
            order = self.api.close_position(symbol)
            return {
                'id': str(order.id),
                'symbol': order.symbol,
                'status': 'closing'
            }
        except Exception as e:
            print(f"Error closing {symbol}: {e}")
            return None
    
    def close_all_positions(self) -> List[dict]:
        """Close all open positions."""
        results = []
        positions = self.get_positions()
        
        for symbol in positions.keys():
            result = self.close_position(symbol)
            if result:
                results.append(result)
        
        return results
    
    def get_latest_bars(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Get recent OHLCV bars for multiple symbols.
        Uses IEX feed (free) instead of SIP (paid).
        
        Args:
            symbols: List of tickers
            days: Number of days of history
        
        Returns:
            Dict mapping symbol -> DataFrame with OHLCV
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        
        result = {}
        
        # Fetch in batches to avoid API limits
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            try:
                bars = self.api.get_bars(
                    batch,
                    tradeapi.TimeFrame.Day,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    feed='iex'  # Use free IEX feed instead of SIP
                ).df
                
                if bars.empty:
                    continue
                
                # Reset index to get timestamp as column
                bars = bars.reset_index()
                
                # Process each symbol
                for symbol in batch:
                    symbol_data = bars[bars['symbol'] == symbol].copy()
                    if symbol_data.empty:
                        continue
                    
                    # Rename columns to match our format
                    df = symbol_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                    df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    result[symbol] = df
            
            except Exception as e:
                print(f"Error fetching bars for batch: {e}")
                continue
        
        return result
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.api.get_clock()
        return clock.is_open
    
    def get_next_open(self) -> datetime:
        """Get the next market open time."""
        clock = self.api.get_clock()
        return clock.next_open
    
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate number of shares based on portfolio allocation.
        
        Args:
            price: Current stock price
        
        Returns:
            Number of shares to buy
        """
        account = self.get_account()
        allocation = account['equity'] * self.position_size_pct
        shares = int(allocation / price)
        return max(1, shares)  # At least 1 share


if __name__ == "__main__":
    # Test connection
    try:
        client = AlpacaClient()
        account = client.get_account()
        print("Connected to Alpaca!")
        print(f"Account Status: {account['status']}")
        print(f"Equity: ${account['equity']:,.2f}")
        print(f"Cash: ${account['cash']:,.2f}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error: {e}")
