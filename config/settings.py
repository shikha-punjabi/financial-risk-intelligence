import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Free API Keys
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    FRED_API_KEY = os.getenv('FRED_API_KEY', '')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
    
    # HuggingFace settings (free)
    HUGGINGFACE_MODEL = "ProsusAI/finbert"  # Free financial sentiment model
    
    # Data settings
    DATA_DIR = "data"
    RAW_DATA_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
    
    # Rate limiting (respect free API limits)
    NEWS_API_RATE_LIMIT = 1000  # per day
    FRED_API_RATE_LIMIT = 120   # per minute
    ALPHA_VANTAGE_RATE_LIMIT = 5  # per minute
    
    # Cache settings
    CACHE_DURATION_HOURS = 1
    
    # Monitoring thresholds and intervals (user-editable)
    NEWS_CHECK_INTERVAL_MIN = int(os.getenv('NEWS_CHECK_INTERVAL_MIN', 30))
    MARKET_CHECK_INTERVAL_MIN = int(os.getenv('MARKET_CHECK_INTERVAL_MIN', 5))
    ECONOMIC_CHECK_INTERVAL_MIN = int(os.getenv('ECONOMIC_CHECK_INTERVAL_MIN', 60))
    SENTIMENT_THRESHOLD_HIGH = float(os.getenv('SENTIMENT_THRESHOLD_HIGH', -0.3))
    SENTIMENT_THRESHOLD_CRITICAL = float(os.getenv('SENTIMENT_THRESHOLD_CRITICAL', -0.5))
    VOLATILITY_THRESHOLD_HIGH = float(os.getenv('VOLATILITY_THRESHOLD_HIGH', 0.3))
    VOLATILITY_THRESHOLD_CRITICAL = float(os.getenv('VOLATILITY_THRESHOLD_CRITICAL', 0.5))
    RISK_ARTICLE_THRESHOLD = float(os.getenv('RISK_ARTICLE_THRESHOLD', 0.2))
    MAX_ALERTS_PER_HOUR = int(os.getenv('MAX_ALERTS_PER_HOUR', 10))
    
    @classmethod
    def validate_keys(cls):
        missing_keys = []
        if not cls.NEWS_API_KEY:
            missing_keys.append('NEWS_API_KEY')
        if not cls.FRED_API_KEY:
            missing_keys.append('FRED_API_KEY')
        
        if missing_keys:
            raise ValueError(f"Missing API keys: {missing_keys}")
        
        return True
