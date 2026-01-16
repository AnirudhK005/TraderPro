"""
TraderPro Paper Trading Bot
Runs daily to generate signals and execute paper trades.

Usage:
    # Run once (manual)
    python paper_trader.py --run-once
    
    # Run in scheduled mode (every day at market open)
    python paper_trader.py --scheduled
    
    # Dry run (show signals without trading)
    python paper_trader.py --dry-run
    
    # Train model first
    python paper_trader.py --train --run-once
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
from datetime import datetime, time
from pathlib import Path
import yaml
import schedule
import time as time_module

from src.trading.alpaca_client import AlpacaClient
from src.trading.signal_generator import SignalGenerator
from src.trading.executor import TradeExecutor


def load_tickers(config_path: str = "configs/tickers.yaml") -> list:
    """Load ticker list from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('tickers', [])


def train_model(generator: SignalGenerator, tickers: list) -> None:
    """Train model on historical data."""
    from src.data.collector import DataCollector
    from src.features.target import TargetEngineer
    import pandas as pd
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Collect data if needed
    data_dir = Path("data")
    collector = DataCollector()
    
    existing_files = list(data_dir.glob("*.csv"))
    if len(existing_files) < len(tickers) // 2:
        print("Collecting historical data...")
        collector.collect(tickers=tickers, period="2y")
    
    # Load and prepare data
    print("Preparing training data...")
    all_data = []
    te = TargetEngineer(horizon=5, threshold=0.01)
    
    for ticker in tickers:
        filepath = data_dir / f"{ticker}.csv"
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df = df.reset_index()
        df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['ticker'] = ticker
        
        # Compute features
        df_features = generator.feature_engineer.compute_features(
            df.set_index('date')
        ).reset_index()
        df_features.rename(columns={'index': 'date'}, inplace=True)
        df_features['ticker'] = ticker
        
        # Add target
        df_final = te.add_target_to_features(df_features)
        all_data.append(df_final)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"Training data: {len(combined)} rows, {combined['ticker'].nunique()} tickers")
    
    # Train model
    generator.train_model(combined)
    
    # Save model
    model_path = "models/trading_model.pkl"
    Path("models").mkdir(exist_ok=True)
    generator.save_model(model_path)


def run_trading_cycle(
    client: AlpacaClient,
    generator: SignalGenerator,
    executor: TradeExecutor,
    tickers: list,
    dry_run: bool = False
) -> None:
    """Run a single trading cycle."""
    print("\n" + "="*60)
    print(f"TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check market status
    if not client.is_market_open():
        next_open = client.get_next_open()
        print(f"Market is closed. Next open: {next_open}")
        if not dry_run:
            return
        print("Running in dry-run mode anyway...")
    
    # Get latest price data (need 100+ days for 50-day SMA)
    print("\nFetching latest market data...")
    price_data = client.get_latest_bars(tickers, days=120)
    print(f"Got data for {len(price_data)} tickers")
    
    if not price_data:
        print("No price data available!")
        return
    
    # Generate signals
    print("\nGenerating signals...")
    signals = generator.generate_signals(price_data)
    print(f"Generated {len(signals)} signals")
    
    # Show top signals
    buy_signals = generator.get_top_signals(signals, n=10, signal_type='BUY')
    sell_signals = generator.get_top_signals(signals, n=5, signal_type='SELL')
    
    print("\nTop BUY Signals:")
    print("-" * 40)
    for s in buy_signals[:5]:
        print(f"  {s['ticker']:6} | Prob: {s['probability']:.2%} | ${s['price']:.2f}")
    
    if sell_signals:
        print("\nTop SELL Signals:")
        print("-" * 40)
        for s in sell_signals[:5]:
            print(f"  {s['ticker']:6} | Prob: {s['probability']:.2%} | ${s['price']:.2f}")
    
    # Execute signals
    results = executor.execute_signals(buy_signals + sell_signals, dry_run=dry_run)
    
    # Check stop-loss and take-profit
    print("\nChecking risk management...")
    closed_losses = executor.close_losing_positions(loss_threshold=-0.05, dry_run=dry_run)
    closed_profits = executor.take_profits(profit_threshold=0.10, dry_run=dry_run)
    
    if closed_losses:
        print(f"Stop-loss triggered: {len(closed_losses)} positions")
    if closed_profits:
        print(f"Take-profit triggered: {len(closed_profits)} positions")
    
    # Portfolio summary
    print("\n" + "="*60)
    print("PORTFOLIO SUMMARY")
    print("="*60)
    summary = executor.get_portfolio_summary()
    print(f"Equity:        ${summary['equity']:,.2f}")
    print(f"Cash:          ${summary['cash']:,.2f}")
    print(f"Positions:     {summary['positions_count']}")
    print(f"Unrealized P/L: ${summary['unrealized_pl']:,.2f} ({summary['unrealized_pl_pct']:.2%})")
    
    if summary['positions']:
        print("\nCurrent Positions:")
        print("-" * 50)
        for symbol, pos in summary['positions'].items():
            pl_color = "+" if pos['unrealized_pl'] >= 0 else ""
            print(f"  {symbol:6} | {pos['qty']:>5} shares | "
                  f"${pos['market_value']:>10,.2f} | "
                  f"{pl_color}{pos['unrealized_plpc']:.2%}")


def main():
    parser = argparse.ArgumentParser(description='TraderPro Paper Trading Bot')
    parser.add_argument('--run-once', action='store_true', help='Run a single trading cycle')
    parser.add_argument('--scheduled', action='store_true', help='Run on schedule (daily at market open)')
    parser.add_argument('--dry-run', action='store_true', help='Show signals without executing trades')
    parser.add_argument('--train', action='store_true', help='Train model before running')
    parser.add_argument('--config', default='configs/alpaca_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TRADERPRO PAPER TRADING BOT")
    print("="*60)
    
    # Load tickers
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers from config")
    
    # Initialize components
    print("\nInitializing...")
    
    try:
        client = AlpacaClient(config_path=args.config)
        account = client.get_account()
        print(f"Connected to Alpaca (Paper Trading)")
        print(f"Account Status: {account['status']}")
        print(f"Starting Equity: ${account['equity']:,.2f}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo set up Alpaca:")
        print("1. Create account at https://alpaca.markets")
        print("2. Get API keys from the Paper Trading dashboard")
        print("3. Copy configs/alpaca_config.yaml.example to configs/alpaca_config.yaml")
        print("4. Fill in your API keys")
        return
    except Exception as e:
        print(f"\nERROR connecting to Alpaca: {e}")
        return
    
    # Load or train model
    model_path = "models/trading_model.pkl"
    generator = SignalGenerator(min_probability=client.min_probability)
    
    if args.train or not Path(model_path).exists():
        train_model(generator, tickers)
    else:
        generator.load_model(model_path)
    
    # Initialize executor
    executor = TradeExecutor(
        client=client,
        max_positions=client.max_positions
    )
    
    # Run mode
    if args.run_once or args.dry_run:
        run_trading_cycle(client, generator, executor, tickers, dry_run=args.dry_run)
    
    elif args.scheduled:
        print("\nScheduled mode - Will run at 9:35 AM ET daily")
        print("Press Ctrl+C to stop")
        
        def scheduled_run():
            run_trading_cycle(client, generator, executor, tickers, dry_run=False)
        
        # Schedule for 9:35 AM ET (5 minutes after market open)
        schedule.every().monday.at("09:35").do(scheduled_run)
        schedule.every().tuesday.at("09:35").do(scheduled_run)
        schedule.every().wednesday.at("09:35").do(scheduled_run)
        schedule.every().thursday.at("09:35").do(scheduled_run)
        schedule.every().friday.at("09:35").do(scheduled_run)
        
        while True:
            schedule.run_pending()
            time_module.sleep(60)
    
    else:
        print("\nNo run mode specified. Use --run-once, --dry-run, or --scheduled")
        print("Run with --help for more options")


if __name__ == "__main__":
    main()

