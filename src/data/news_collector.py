import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.request_count = 0
        self.daily_limit = Config.NEWS_API_RATE_LIMIT
        
    def _make_request(self, endpoint: str, params: dict, retries: int = 3) -> Optional[dict]:
        """Make API request with error handling, rate limiting, and retries"""
        if self.request_count >= self.daily_limit:
            logger.warning("Daily API limit reached")
            return None
            
        params['apiKey'] = self.api_key
        
        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning("Rate limit hit, waiting...")
                    time.sleep(60)
                    attempt += 1
                    continue
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return None
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}): {e}", exc_info=True)
                time.sleep(2 ** attempt)  # Exponential backoff
                attempt += 1
        
        logger.error(f"Failed to fetch data from News API after {retries} attempts.")
        return None
    
    def get_financial_news(self, query: str = "financial markets", hours_back: int = 24) -> pd.DataFrame:
        """Get financial news with comprehensive error handling"""
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%dT%H:%M:%S')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 100
            }
            
            data = self._make_request('everything', params)
            
            if not data or 'articles' not in data:
                logger.warning("No articles returned from API")
                print("[NewsCollector] No articles returned from API")
                return pd.DataFrame()
            
            articles = []
            for article in data['articles']:
                articles.append({
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', 'No Description'),
                    'content': article.get('content', 'No Content'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', datetime.now().isoformat()),
                    'url': article.get('url', ''),
                    'collected_at': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(articles)
            
            df = df.dropna(subset=['title', 'description'])
            df = df[df['title'] != '[Removed]']
            
            logger.info(f"Collected {len(df)} valid articles")
            print(f"[NewsCollector] Collected {len(df)} valid articles")
            print(df.head())
            return df
            
        except Exception as e:
            logger.error(f"Error collecting news: {e}", exc_info=True)
            print(f"[NewsCollector] Error collecting news: {e}")
            return pd.DataFrame()
    
    def get_company_specific_news(self, companies: List[str]) -> pd.DataFrame:
        """Get news for specific companies"""
        all_news = []
        
        for company in companies:
            logger.info(f"Collecting news for {company}")
            company_news = self.get_financial_news(query=company, hours_back=48)
            if not company_news.empty:
                company_news['target_company'] = company
                all_news.append(company_news)
            
            time.sleep(1)
        
        return pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
    
    def save_news_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save news data with timestamp"""
        if filename is None:
            filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = f"{Config.RAW_DATA_DIR}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} articles to {filepath}")
        return filepath
