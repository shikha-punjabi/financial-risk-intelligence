from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class FreeLLMAnalyzer:
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        # Initialize free models
        self._initialize_models()
        # Financial analysis templates
        self.analysis_templates = self._load_analysis_templates()
    def _initialize_models(self):
        """Initialize free LLM models from Hugging Face"""
        try:
            # Financial text classification model (free)
            logger.info("Loading financial classification model...")
            self.models['financial_classifier'] = pipeline(
                "text-classification",
                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Financial classifier loaded")
        except Exception as e:
            logger.warning(f"Failed to load financial classifier: {e}")
            self.models['financial_classifier'] = None
        try:
            # General text generation model (small, free)
            logger.info("Loading text generation model...")
            self.models['text_generator'] = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if self.device == "cuda" else -1,
                max_length=200,
                do_sample=True,
                temperature=0.7
            )
            logger.info("✅ Text generator loaded")
        except Exception as e:
            logger.warning(f"Failed to load text generator: {e}")
            self.models['text_generator'] = None
        try:
            # Question answering model (free)
            logger.info("Loading question answering model...")
            self.models['qa_model'] = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ QA model loaded")
        except Exception as e:
            logger.warning(f"Failed to load QA model: {e}")
            self.models['qa_model'] = None
    def _load_analysis_templates(self) -> Dict:
        """Load analysis templates for different scenarios"""
        return {
            'risk_analysis': {
                'prompt': "Analyze the financial risk in this news: {text}",
                'questions': [
                    "What are the main risk factors mentioned?",
                    "Which sectors could be affected?",
                    "What is the potential market impact?",
                    "How urgent is this risk?"
                ]
            },
            'sector_analysis': {
                'prompt': "Identify affected sectors in this financial news: {text}",
                'questions': [
                    "Which industries are mentioned?",
                    "Are the impacts positive or negative?",
                    "Which companies might be affected?"
                ]
            },
            'market_impact': {
                'prompt': "Assess market impact of this news: {text}",
                'questions': [
                    "Will this affect stock prices?",
                    "What is the expected direction of market movement?",
                    "How significant is this news?"
                ]
            }
        }
    def analyze_text_with_qa(self, text: str, analysis_type: str = 'risk_analysis') -> Dict:
        """Analyze text using question-answering approach"""
        if not self.models['qa_model'] or not text:
            return {'error': 'QA model not available or empty text'}
        template = self.analysis_templates.get(analysis_type, self.analysis_templates['risk_analysis'])
        results = {
            'analysis_type': analysis_type,
            'text_length': len(text),
            'answers': {},
            'timestamp': datetime.now().isoformat()
        }
        # Truncate text if too long
        context = text[:1000] if len(text) > 1000 else text
        try:
            for question in template['questions']:
                answer = self.models['qa_model'](
                    question=question,
                    context=context
                )
                results['answers'][question] = {
                    'answer': answer['answer'],
                    'confidence': answer['score'],
                    'start': answer['start'],
                    'end': answer['end']
                }
        except Exception as e:
            logger.error(f"QA analysis failed: {e}")
            results['error'] = str(e)
        return results
    def classify_financial_content(self, text: str) -> Dict:
        """Classify financial content using free models"""
        results = {}
        if self.models['financial_classifier'] and text:
            try:
                # Truncate text
                text_truncated = text[:512] if len(text) > 512 else text
                classification = self.models['financial_classifier'](text_truncated)
                results['financial_classification'] = {
                    'label': classification[0]['label'],
                    'score': classification[0]['score'],
                    'model': 'financial_classifier'
                }
            except Exception as e:
                logger.error(f"Financial classification failed: {e}")
                results['financial_classification'] = {'error': str(e)}
        return results
    def extract_key_entities(self, text: str) -> Dict:
        """Extract financial entities using rule-based approach"""
        entities = {
            'companies': [],
            'currencies': [],
            'numbers': [],
            'dates': [],
            'financial_terms': []
        }
        try:
            # Company patterns (basic)
            company_patterns = [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co)\b',
                r'\b[A-Z]{2,5}\b',  # Stock tickers
            ]
            for pattern in company_patterns:
                matches = re.findall(pattern, text)
                entities['companies'].extend(matches)
            # Currency patterns
            currency_pattern = r'\$[\d,]+(?:\.\d{2})?[BMK]?'
            entities['currencies'] = re.findall(currency_pattern, text)
            # Percentage patterns
            percentage_pattern = r'\d+(?:\.\d+)?%'
            entities['numbers'] = re.findall(percentage_pattern, text)
            # Financial terms
            financial_terms = [
                'earnings', 'revenue', 'profit', 'loss', 'dividend', 'stock',
                'bond', 'interest rate', 'inflation', 'GDP', 'unemployment',
                'merger', 'acquisition', 'IPO', 'bankruptcy', 'default'
            ]
            text_lower = text.lower()
            for term in financial_terms:
                if term in text_lower:
                    entities['financial_terms'].append(term)
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            entities['error'] = str(e)
        return entities
    def generate_risk_summary(self, news_articles: List[Dict]) -> Dict:
        """Generate risk summary using rule-based approach and free models"""
        summary = {
            'total_articles': len(news_articles),
            'risk_categories': {},
            'affected_sectors': {},
            'key_themes': [],
            'urgency_assessment': 'LOW',
            'generation_method': 'rule_based_with_ml',
            'timestamp': datetime.now().isoformat()
        }
        if not news_articles:
            return summary
        try:
            # Aggregate sentiment and risk scores
            sentiment_scores = []
            risk_levels = []
            all_text = []
            for article in news_articles:
                if 'sentiment_score' in article:
                    sentiment_scores.append(article['sentiment_score'])
                if 'risk_level' in article:
                    risk_levels.append(article['risk_level'])
                # Combine text for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                all_text.append(text)
            # Risk category analysis
            high_risk_count = risk_levels.count('HIGH')
            medium_risk_count = risk_levels.count('MEDIUM')
            low_risk_count = risk_levels.count('LOW')
            summary['risk_categories'] = {
                'HIGH': high_risk_count,
                'MEDIUM': medium_risk_count,
                'LOW': low_risk_count
            }
            # Overall urgency assessment
            risk_ratio = high_risk_count / len(news_articles) if news_articles else 0
            if risk_ratio > 0.3:
                summary['urgency_assessment'] = 'HIGH'
            elif risk_ratio > 0.15:
                summary['urgency_assessment'] = 'MEDIUM'
            else:
                summary['urgency_assessment'] = 'LOW'
            # Extract common themes using entity extraction
            all_entities = []
            for text in all_text[:10]:  # Limit to first 10 articles for performance
                entities = self.extract_key_entities(text)
                all_entities.extend(entities.get('financial_terms', []))
            # Count theme frequency
            theme_counts = pd.Series(all_entities).value_counts()
            summary['key_themes'] = theme_counts.head(5).to_dict()
            # Sentiment analysis
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                summary['average_sentiment'] = avg_sentiment
                summary['sentiment_volatility'] = np.std(sentiment_scores)
                if avg_sentiment < -0.3:
                    summary['market_mood'] = 'Very Negative'
                elif avg_sentiment < -0.1:
                    summary['market_mood'] = 'Negative'
                elif avg_sentiment < 0.1:
                    summary['market_mood'] = 'Neutral'
                elif avg_sentiment < 0.3:
                    summary['market_mood'] = 'Positive'
                else:
                    summary['market_mood'] = 'Very Positive'
            # Generate recommendations based on analysis
            summary['recommendations'] = self._generate_recommendations(summary)
        except Exception as e:
            logger.error(f"Risk summary generation failed: {e}")
            summary['error'] = str(e)
        return summary
    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        try:
            urgency = summary.get('urgency_assessment', 'LOW')
            mood = summary.get('market_mood', 'Neutral')
            if urgency == 'HIGH':
                recommendations.extend([
                    "Monitor portfolio risk exposure closely",
                    "Consider reducing position sizes in volatile sectors",
                    "Review hedging strategies"
                ])
            
            if mood in ['Very Negative', 'Negative']:
                recommendations.extend([
                    "Increase cash reserves",
                    "Look for defensive investment opportunities",
                    "Monitor stop-loss levels"
                ])
            elif mood in ['Very Positive', 'Positive']:
                recommendations.extend([
                    "Consider increasing exposure to trending sectors",
                    "Monitor for overbought conditions",
                    "Prepare for potential corrections"
                ])
            
            # Theme-based recommendations
            themes = summary.get('key_themes', {})
            if 'inflation' in themes:
                recommendations.append("Monitor inflation-sensitive assets")
            if 'interest rate' in themes:
                recommendations.append("Review interest rate exposure")
            if 'earnings' in themes:
                recommendations.append("Focus on earnings quality analysis")
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations[:5]  # Limit to top 5 recommendations

def analyze_news_batch(self, news_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze a batch of news articles using free LLM models"""
    logger.info(f"Analyzing {len(news_df)} articles with free LLM models")
    
    if news_df.empty:
        return news_df
        # Add LLM analysis columns
        news_df['llm_risk_analysis'] = None
        news_df['llm_entities'] = None
        news_df['llm_classification'] = None
        for idx, row in news_df.iterrows():
            if idx % 5 == 0:
                logger.info(f"Processing article {idx + 1}/{len(news_df)}")
            text = f"{row.get('title', '')} {row.get('description', '')}"
            # Risk analysis using QA
            risk_analysis = self.analyze_text_with_qa(text, 'risk_analysis')
            news_df.at[idx, 'llm_risk_analysis'] = risk_analysis
            # Entity extraction
            entities = self.extract_key_entities(text)
            news_df.at[idx, 'llm_entities'] = entities
            # Financial classification
            classification = self.classify_financial_content(text)
            news_df.at[idx, 'llm_classification'] = classification
        logger.info("✅ LLM analysis completed")
        return news_df
