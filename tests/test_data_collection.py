<<<<<<< HEAD
=======
# tests/test_data_collection.py
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
import pytest
import pandas as pd
from src.data.news_collector import NewsCollector
from src.data.market_collector import MarketDataCollector
from src.data.economic_collector import EconomicDataCollector
from config.settings import Config
import os

def test_data_pipeline():
    """Test the complete data collection pipeline"""
<<<<<<< HEAD
=======
    
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
    # Test configuration
    try:
        Config.validate_keys()
        print("✅ API keys validated")
    except ValueError as e:
        print(f"❌ Missing API keys: {e}")
        return False
<<<<<<< HEAD
=======
    
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
    # Test news collection
    try:
        news_collector = NewsCollector(Config.NEWS_API_KEY)
        news_df = news_collector.get_financial_news(hours_back=6)  # Small test
<<<<<<< HEAD
=======
        
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
        if not news_df.empty:
            print(f"✅ News collection successful: {len(news_df)} articles")
            print(f"   Sample headline: {news_df.iloc[0]['title']}")
        else:
            print("⚠️  No news articles collected (may be API limit)")
<<<<<<< HEAD
    except Exception as e:
        print(f"❌ News collection failed: {e}")
        return False
=======
        
    except Exception as e:
        print(f"❌ News collection failed: {e}")
        return False
    
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
    # Test market data collection
    try:
        market_collector = MarketDataCollector()
        market_data = market_collector.get_market_data(['SPY', 'QQQ'])
<<<<<<< HEAD
=======
        
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
        if market_data:
            print(f"✅ Market data collection successful: {len(market_data)} symbols")
            spy_price = market_data.get('SPY', {}).get('current_price')
            if spy_price:
                print(f"   SPY current price: ${spy_price:.2f}")
        else:
            print("❌ No market data collected")
            return False
<<<<<<< HEAD
    except Exception as e:
        print(f"❌ Market data collection failed: {e}")
        return False
    # Test economic data collection
    try:
        econ_collector = EconomicDataCollector(Config.FRED_API_KEY)
        # Test single indicator first
        gdp_data = econ_collector.get_indicator('GDP')
        if gdp_data is not None:
            print(f"✅ Economic data collection successful: GDP data points: {len(gdp_data)}")
            print(f"   Latest GDP: {gdp_data.iloc[-1]:.2f}")
        else:
            print("❌ No economic data collected")
            return False
    except Exception as e:
        print(f"❌ Economic data collection failed: {e}")
        return False
    print("✅ All data collection tests passed!")
    assert True

if __name__ == "__main__":
    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    test_data_pipeline()
=======
        
    except Exception as e:
        print(f"❌ Market data collection failed: {e}")
        return False
    
    # Test economic data collection
    try:
        econ_collector = EconomicDataCollector(Config.FRED_API_KEY)
        
        # Test single indicator first
        gdp_data = econ_collector.get_indicator('GDP')

        if gdp_data is not None:
           print(f"✅ Economic data collection successful: GDP data points: {len(gdp_data)}")
           print(f"   Latest GDP: {gdp_data.iloc[-1]:.2f}")
        else:
           print("❌ No economic data collected")
        return False
    
    except Exception as e:
       print(f"❌ Economic data collection failed: {e}")
       return False
    print("✅ All data collection tests passed!")
    return True

if __name__ == "__main__":
   # Create data directories
   os.makedirs("data/raw", exist_ok=True)
   os.makedirs("data/processed", exist_ok=True)
   
   test_data_pipeline()
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
