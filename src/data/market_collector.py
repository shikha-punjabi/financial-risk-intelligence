import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
from config.settings import Config

logger = logging.getLogger(__name__)

class MarketDataCollector:
    def __init__(self):
        self.major_indices = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF', 
            'DIA': 'Dow Jones ETF',
            'IWM': 'Russell 2000 ETF',
            'VTI': 'Total Stock Market ETF'
        }
        
        self.volatility_indicators = ['^VIX', '^VXN', '^RVX']
        self.sector_etfs = {
            'XLF': 'Financial',
            'XLK': 'Technology', 
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
    
    def get_market_data(self, symbols: List[str], period: str = "1mo", retries: int = 3) -> Dict:
        """Get market data with error handling and retries"""
        market_data = {}
        for symbol in symbols:
            attempt = 0
            while attempt < retries:
                try:
                    logger.info(f"Fetching data for {symbol}")
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if hist.empty:
                        logger.warning(f"No historical data for {symbol}")
                        print(f"[MarketDataCollector] No historical data for {symbol}")
                        break
                    try:
                        info = ticker.info
                    except Exception as e:
                        logger.warning(f"Error fetching info for {symbol}: {e}", exc_info=True)
                        info = {}
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                    market_data[symbol] = {
                        'price_data': hist,
                        'current_price': hist['Close'].iloc[-1] if not hist.empty else None,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', None),
                        'beta': info.get('beta', 1.0),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'volatility': volatility,
                        'returns': returns,
                        'volume_avg': hist['Volume'].mean() if not hist.empty else 0,
                        'last_updated': datetime.now().isoformat()
                    }
                    if not hist.empty:
                        market_data[symbol].update(self._calculate_technical_indicators(hist))
                    time.sleep(0.1)
                    break  # Success, break retry loop
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol} (attempt {attempt+1}): {e}", exc_info=True)
                    print(f"[MarketDataCollector] Error fetching data for {symbol}: {e}")
                    time.sleep(2 ** attempt)
                    attempt += 1
        print(f"[MarketDataCollector] Fetched data for {len(market_data)} symbols")
        if market_data:
            sample_symbol = list(market_data.keys())[0]
            print(f"[MarketDataCollector] Sample data for {sample_symbol}:")
            print(market_data[sample_symbol])
        return market_data
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate basic technical indicators"""
        close = price_data['Close']
        
        indicators = {}
        
        try:
            # Moving averages
            indicators['sma_20'] = close.rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = close.rolling(window=50).mean().iloc[-1]
            
            # RSI (simplified)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Bollinger Bands
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            
            # Price relative to moving averages
            current_price = close.iloc[-1]
            indicators['price_vs_sma20'] = (current_price / indicators['sma_20'] - 1) * 100
            indicators['price_vs_sma50'] = (current_price / indicators['sma_50'] - 1) * 100
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def get_comprehensive_market_snapshot(self) -> Dict:
        """Get comprehensive market data including indices, sectors, and volatility"""
        all_symbols = (list(self.major_indices.keys()) + 
                      self.volatility_indicators + 
                      list(self.sector_etfs.keys()))
        
        logger.info("Collecting comprehensive market snapshot")
        market_data = self.get_market_data(all_symbols)
        
        # Add market summary
        market_summary = self._calculate_market_summary(market_data)
        
        return {
            'market_data': market_data,
            'market_summary': market_summary,
            'collection_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_market_summary(self, market_data: Dict) -> Dict:
        """Calculate overall market summary metrics"""
        summary = {}
        
        try:
            # Market performance
            if 'SPY' in market_data and market_data['SPY']['price_data'] is not None:
                spy_returns = market_data['SPY']['returns']
                summary['spy_daily_return'] = spy_returns.iloc[-1] * 100 if len(spy_returns) > 0 else 0
                summary['spy_volatility'] = market_data['SPY']['volatility'] * 100
            
            # VIX level
            if '^VIX' in market_data and market_data['^VIX']['current_price']:
                summary['vix_level'] = market_data['^VIX']['current_price']
                summary['market_fear_level'] = self._interpret_vix(summary['vix_level'])
            
            # Sector performance
            sector_performance = {}
            for symbol, sector in self.sector_etfs.items():
                if symbol in market_data and market_data[symbol]['returns'] is not None:
                    returns = market_data[symbol]['returns']
                    if len(returns) > 0:
                        sector_performance[sector] = returns.iloc[-1] * 100
            
            summary['sector_performance'] = sector_performance
            summary['best_sector'] = max(sector_performance.items(), key=lambda x: x[1]) if sector_performance else None
            summary['worst_sector'] = min(sector_performance.items(), key=lambda x: x[1]) if sector_performance else None
            
        except Exception as e:
            logger.error(f"Error calculating market summary: {e}")
        
        return summary
    
    def _interpret_vix(self, vix_level: float) -> str:
        """Interpret VIX level"""
        if vix_level < 12:
            return "Very Low Volatility"
        elif vix_level < 20:
            return "Low Volatility"
        elif vix_level < 30:
            return "Normal Volatility"
        elif vix_level < 40:
            return "High Volatility"
        else:
            return "Very High Volatility"
    
    def save_market_data(self, market_data: Dict, filename: str = None) -> str:
        """Save market data to file"""
        if filename is None:
            filename = f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = f"{Config.RAW_DATA_DIR}/{filename}"
        
        # Convert numpy types to native Python types for JSON serialization
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj
        
        # Clean data for JSON serialization
        clean_data = json.loads(json.dumps(market_data, default=convert_numpy))
        
        with open(filepath, 'w') as f:
            json.dump(clean_data, f, indent=2)
        
        logger.info(f"Saved market data to {filepath}")
        return filepath
