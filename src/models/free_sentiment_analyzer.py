import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FreeSentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._initialize_models()
        # Financial keywords for enhanced analysis
        self.financial_positive = [
            'profit', 'growth', 'earnings', 'revenue', 'beat', 'outperform', 
            'strong', 'robust', 'bullish', 'upgrade', 'buy', 'momentum',
            'gain', 'surge', 'rally', 'optimistic', 'confidence', 'expansion'
        ]
        self.financial_negative = [
            'loss', 'decline', 'miss', 'underperform', 'weak', 'bearish',
            'downgrade', 'sell', 'risk', 'concern', 'volatility', 'uncertainty',
            'drop', 'crash', 'plunge', 'pessimistic', 'recession', 'crisis'
        ]
        self.risk_keywords = [
            'bankruptcy', 'default', 'liquidation', 'restructuring', 'lawsuit',
            'investigation', 'fraud', 'scandal', 'warning', 'alert', 'emergency',
            'critical', 'severe', 'major', 'significant', 'substantial'
        ]
    def _initialize_models(self):
        """Initialize free sentiment analysis models"""
        try:
            # FinBERT - Free financial sentiment model
            logger.info("Loading FinBERT model...")
            self.models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1  # CPU only for free version
            )
            logger.info("✅ FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            self.models['finbert'] = None
        try:
            # General sentiment model as backup
            logger.info("Loading general sentiment model...")
            self.models['general'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            logger.info("✅ General sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load general model: {e}")
            self.models['general'] = None
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Comprehensive sentiment analysis using multiple free models"""
        if not text or pd.isna(text):
            return self._empty_sentiment()
        results = {}
        # VADER Sentiment (Rule-based, very fast)
        try:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results['vader'] = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu']
            }
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            results['vader'] = self._empty_sentiment()['vader']
        # TextBlob Sentiment (Simple but effective)
        try:
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            results['textblob'] = {'polarity': 0, 'subjectivity': 0}
        # FinBERT (Financial-specific)
        if self.models['finbert']:
            try:
                # Truncate text if too long
                text_truncated = text[:512] if len(text) > 512 else text
                finbert_result = self.models['finbert'](text_truncated)
                # Convert to numerical score
                label = finbert_result[0]['label'].lower()
                score = finbert_result[0]['score']
                if label == 'positive':
                    finbert_score = score
                elif label == 'negative':
                    finbert_score = -score
                else:  # neutral
                    finbert_score = 0
                results['finbert'] = {
                    'score': finbert_score,
                    'label': label,
                    'confidence': score
                }
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {e}")
                results['finbert'] = {'score': 0, 'label': 'neutral', 'confidence': 0}
        # Financial keyword analysis
        results['financial_keywords'] = self._analyze_financial_keywords(text)
        # Risk keyword analysis
        results['risk_keywords'] = self._analyze_risk_keywords(text)
        # Combined sentiment score
        results['combined_sentiment'] = self._calculate_combined_sentiment(results)
        return results
    def _analyze_financial_keywords(self, text: str) -> Dict:
        """Analyze financial-specific keywords"""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.financial_positive if word in text_lower)
        negative_count = sum(1 for word in self.financial_negative if word in text_lower)
        total_words = positive_count + negative_count
        if total_words > 0:
            sentiment_score = (positive_count - negative_count) / total_words
        else:
            sentiment_score = 0
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_score': sentiment_score,
            'keyword_intensity': total_words
        }
    def _analyze_risk_keywords(self, text: str) -> Dict:
        """Analyze risk-related keywords"""
        text_lower = text.lower()
        risk_count = sum(1 for word in self.risk_keywords if word in text_lower)
        risk_words_found = [word for word in self.risk_keywords if word in text_lower]
        # Risk level based on keyword count and severity
        if risk_count >= 3:
            risk_level = "HIGH"
        elif risk_count >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        return {
            'risk_word_count': risk_count,
            'risk_words_found': risk_words_found,
            'risk_level': risk_level,
            'risk_score': min(risk_count / 3, 1.0)  # Normalize to 0-1
        }
    def _calculate_combined_sentiment(self, results: Dict) -> Dict:
        """Calculate weighted combined sentiment score"""
        scores = []
        weights = []
        # VADER (weight: 0.2)
        if 'vader' in results:
            scores.append(results['vader']['compound'])
            weights.append(0.2)
        # TextBlob (weight: 0.2)
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
            weights.append(0.2)
        # FinBERT (weight: 0.4 - highest weight for financial text)
        if 'finbert' in results and results['finbert']['score'] != 0:
            scores.append(results['finbert']['score'])
            weights.append(0.4)

        if scores:
            # Weighted average
            combined_score = np.average(scores, weights=weights[:len(scores)])
        
        if scores:
            # Weighted average
            risk_adjustment = -risk_score * 0.3  # Reduce sentiment by risk
            final_score = combined_score + risk_adjustment
            # Confidence based on agreement between models
            confidence = 1 - (np.std(scores) if len(scores) > 1 else 0)
            return {
                'sentiment_score': final_score,
                'confidence': confidence,
                'risk_adjusted': risk_adjustment != 0,
                'model_agreement': len(scores)
            }
        return {'sentiment_score': 0, 'confidence': 0, 'risk_adjusted': False, 'model_agreement': 0}
    def _empty_sentiment(self) -> Dict:
        """Return empty sentiment structure"""
        return {
            'vader': {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1},
            'textblob': {'polarity': 0, 'subjectivity': 0},
            'finbert': {'score': 0, 'label': 'neutral', 'confidence': 0},
            'financial_keywords': {'positive_words': 0, 'negative_words': 0, 'sentiment_score': 0, 'keyword_intensity': 0},
            'risk_keywords': {'risk_word_count': 0, 'risk_words_found': [], 'risk_level': 'LOW', 'risk_score': 0},
            'combined_sentiment': {'sentiment_score': 0, 'confidence': 0, 'risk_adjusted': False, 'model_agreement': 0}
        }
    def analyze_news_batch(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for a batch of news articles"""
        logger.info(f"Analyzing sentiment for {len(news_df)} articles")
        if news_df.empty:
            return news_df
        # Combine title and description for analysis
        news_df['combined_text'] = (
            news_df['title'].fillna('') + ' ' + 
            news_df['description'].fillna('')
        ).str.strip()
        # Analyze each article
        sentiment_results = []
        for idx, row in news_df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Processing article {idx + 1}/{len(news_df)}")
            sentiment = self.analyze_text_sentiment(row['combined_text'])
            sentiment_results.append(sentiment)
        # Add sentiment columns to dataframe
        news_df['sentiment_score'] = [r['combined_sentiment']['sentiment_score'] for r in sentiment_results]
        news_df['sentiment_confidence'] = [r['combined_sentiment']['confidence'] for r in sentiment_results]
        news_df['risk_level'] = [r['risk_keywords']['risk_level'] for r in sentiment_results]
        news_df['risk_score'] = [r['risk_keywords']['risk_score'] for r in sentiment_results]
        news_df['financial_keyword_count'] = [r['financial_keywords']['keyword_intensity'] for r in sentiment_results]
        # Store detailed results
        news_df['sentiment_details'] = sentiment_results
        logger.info("✅ Sentiment analysis completed")
        return news_df
    def get_market_sentiment_summary(self, analyzed_news: pd.DataFrame) -> Dict:
        """Get overall market sentiment summary"""
        if analyzed_news.empty:
            return {'error': 'No news data available'}
        summary = {
            'total_articles': len(analyzed_news),
            'avg_sentiment': analyzed_news['sentiment_score'].mean(),
            'sentiment_std': analyzed_news['sentiment_score'].std(),
            'positive_articles': (analyzed_news['sentiment_score'] > 0.1).sum(),
            'negative_articles': (analyzed_news['sentiment_score'] < -0.1).sum(),
            'neutral_articles': (abs(analyzed_news['sentiment_score']) <= 0.1).sum(),
            'high_risk_articles': (analyzed_news['risk_level'] == 'HIGH').sum(),
            'medium_risk_articles': (analyzed_news['risk_level'] == 'MEDIUM').sum(),
            'low_risk_articles': (analyzed_news['risk_level'] == 'LOW').sum(),
            'avg_confidence': analyzed_news['sentiment_confidence'].mean(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        # Market sentiment interpretation
        avg_sentiment = summary['avg_sentiment']
        if avg_sentiment > 0.2:
            summary['market_mood'] = "Bullish"
        elif avg_sentiment > 0.05:
            summary['market_mood'] = "Cautiously Optimistic"
        elif avg_sentiment > -0.05:
            summary['market_mood'] = "Neutral"
        elif avg_sentiment > -0.2:
            summary['market_mood'] = "Cautiously Pessimistic"
        else:
            summary['market_mood'] = "Bearish"
        # Risk assessment
        risk_ratio = summary['high_risk_articles'] / summary['total_articles']
        if risk_ratio > 0.2:
            summary['risk_assessment'] = "High Risk Environment"
        elif risk_ratio > 0.1:
            summary['risk_assessment'] = "Elevated Risk"
        else:
            summary['risk_assessment'] = "Normal Risk"
        return summary
