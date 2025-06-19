import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import schedule
import time
import threading
from typing import Dict, List, Optional, Callable
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
import os

from src.data.news_collector import NewsCollector
from src.data.market_collector import MarketDataCollector
from src.data.economic_collector import EconomicDataCollector
from src.models.free_sentiment_analyzer import FreeSentimentAnalyzer
from src.models.free_llm_analyzer import FreeLLMAnalyzer
from config.settings import Config
from src.agents.alert_utils import create_alert

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    id: str
    timestamp: datetime
    severity: AlertSeverity
    category: str
    title: str
    description: str
    affected_assets: List[str]
    recommended_actions: List[str]
    confidence_score: float
    data_source: str
    expiry_time: Optional[datetime] = None

class PortfolioMonitoringAgent:
    def __init__(self):
        self.is_running = False
        self.monitoring_thread = None
        
        # Initialize data collectors
        self.news_collector = NewsCollector(Config.NEWS_API_KEY)
        self.market_collector = MarketDataCollector()
        self.econ_collector = EconomicDataCollector(Config.FRED_API_KEY)
        
        # Initialize AI models
        self.sentiment_analyzer = FreeSentimentAnalyzer()
        self.llm_analyzer = FreeLLMAnalyzer()
        
        # Alert management
        self.active_alerts = []
        self.alert_history = []
        self.alert_callbacks = []
        self.muted_alert_ids = set()

        # Monitoring configuration
        self.config = {
            'news_check_interval': Config.NEWS_CHECK_INTERVAL_MIN,  # minutes
            'market_check_interval': Config.MARKET_CHECK_INTERVAL_MIN,   # minutes
            'economic_check_interval': Config.ECONOMIC_CHECK_INTERVAL_MIN,  # minutes
            'sentiment_threshold_high': Config.SENTIMENT_THRESHOLD_HIGH,
            'sentiment_threshold_critical': Config.SENTIMENT_THRESHOLD_CRITICAL,
            'volatility_threshold_high': Config.VOLATILITY_THRESHOLD_HIGH,
            'volatility_threshold_critical': Config.VOLATILITY_THRESHOLD_CRITICAL,
            'risk_article_threshold': Config.RISK_ARTICLE_THRESHOLD,  # fraction
            'max_alerts_per_hour': Config.MAX_ALERTS_PER_HOUR
        }
        
        # Portfolio context (would be loaded from user data)
        self.portfolio_context = {
            'sectors': ['Technology', 'Financial', 'Healthcare', 'Energy'],
            'major_holdings': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'risk_tolerance': 'MEDIUM',
            'investment_horizon': 'LONG_TERM'
        }
        
        # Last check timestamps
        self.last_checks = {
            'news': datetime.min,
            'market': datetime.min,
            'economic': datetime.min
        }
        
        logger.info("Portfolio Monitoring Agent initialized")
    
    def start_monitoring(self):
        """Start the monitoring agent"""
        if self.is_running:
            logger.warning("Monitoring agent already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Portfolio Monitoring Agent")
        
        # Schedule monitoring tasks
        schedule.every(self.config['news_check_interval']).minutes.do(self._check_news_risks)
        schedule.every(self.config['market_check_interval']).minutes.do(self._check_market_risks)
        schedule.every(self.config['economic_check_interval']).minutes.do(self._check_economic_risks)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("✅ Portfolio monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring agent"""
        self.is_running = False
        schedule.clear()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Portfolio monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
                # Clean up expired alerts
                self._cleanup_expired_alerts()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    def _check_news_risks(self):
        """Check for news-based risks"""
        try:
            logger.info("🔍 Checking news risks...")
            if self.last_checks['news'] == datetime.min:
                hours_back = 24
            else:
                hours_since_last = (datetime.now() - self.last_checks['news']).total_seconds() / 3600
                hours_back = int(max(1, min(hours_since_last + 1, 24)))
            news_df = self.news_collector.get_financial_news(hours_back=hours_back)
            # Ensure raw data directory exists
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            if not news_df.empty:
                filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                news_df.to_csv(f"{Config.RAW_DATA_DIR}/{filename}", index=False)
                logger.info(f"Saved news batch to {Config.RAW_DATA_DIR}/{filename}")
            if news_df.empty:
                logger.info("No new articles found")
                return
            analyzed_news = self.sentiment_analyzer.analyze_news_batch(news_df)
            self._analyze_news_risks(analyzed_news)
            self.last_checks['news'] = datetime.now()
        except Exception as e:
            logger.error(f"Error checking news risks: {e}", exc_info=True)

    def _analyze_news_risks(self, news_df: pd.DataFrame):
        required_cols = {'sentiment_score', 'risk_level'}
        if not required_cols.issubset(news_df.columns):
            logger.error(f"Missing columns in news_df: {required_cols - set(news_df.columns)}")
            return
        
        # High negative sentiment alert
        very_negative_articles = news_df[news_df['sentiment_score'] < Config.RISK_THRESHOLDS["sentiment_critical"]]
        if len(very_negative_articles) > 0:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                category="SENTIMENT",
                title=f"Critical Negative Sentiment Detected",
                description=f"{len(very_negative_articles)} articles with very negative sentiment found",
                affected_assets=self._extract_affected_assets(very_negative_articles),
                recommended_actions=[
                    "Review portfolio exposure to mentioned sectors",
                    "Consider defensive positioning",
                    "Monitor for further developments"
                ],
                confidence_score=0.9,
                data_source="news_sentiment"
            )
        
        # High risk article concentration
        high_risk_articles = news_df[news_df['risk_level'] == 'HIGH']
        risk_ratio = len(high_risk_articles) / len(news_df)
        
        if risk_ratio > Config.RISK_THRESHOLDS["risk_article_threshold"]:
            self._create_alert(
                severity=AlertSeverity.HIGH,
                category="RISK_CONCENTRATION",
                title=f"High Risk Article Concentration: {risk_ratio:.1%}",
                description=f"{len(high_risk_articles)} high-risk articles out of {len(news_df)} total",
                affected_assets=self._extract_affected_assets(high_risk_articles),
                recommended_actions=[
                    "Investigate specific risk factors",
                    "Review position sizing",
                    "Increase monitoring frequency"
                ],
                confidence_score=0.8,
                data_source="news_risk_analysis"
            )
        
        # Sector-specific risks
        self._check_sector_risks(news_df)
    
    def _check_market_risks(self):
        """Check for market-based risks"""
        try:
            logger.info("📈 Checking market risks...")
            market_data = self.market_collector.get_comprehensive_market_snapshot()
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            if market_data and 'market_data' in market_data:
                filename = f"market_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(f"{Config.RAW_DATA_DIR}/{filename}", "w") as f:
                    json.dump(market_data, f, default=str, indent=2)
                logger.info(f"Saved market batch to {Config.RAW_DATA_DIR}/{filename}")
            if not market_data['market_data']:
                logger.warning("No market data available")
                return
            self._analyze_market_risks(market_data)
            self.last_checks['market'] = datetime.now()
        except Exception as e:
            logger.error(f"Error checking market risks: {e}", exc_info=True)
    
    def _analyze_market_risks(self, market_data: Dict):
        """Analyze market data for risk patterns"""
        market_summary = market_data.get('market_summary', {})
        
        # VIX spike detection
        vix_level = market_summary.get('vix_level', 20)
        if vix_level > 30:
            severity = AlertSeverity.CRITICAL if vix_level > 40 else AlertSeverity.HIGH
            
            self._create_alert(
                severity=severity,
                category="VOLATILITY",
                title=f"VIX Spike Detected: {vix_level:.1f}",
                description=f"Volatility index at {vix_level:.1f}, indicating {market_summary.get('market_fear_level', 'high volatility')}",
                affected_assets=["SPY", "QQQ", "Portfolio-wide"],
                recommended_actions=[
                    "Consider hedging strategies",
                    "Review stop-loss levels",
                    "Reduce position sizes if needed"
                ],
                confidence_score=0.95,
                data_source="market_volatility"
            )
        
        # Market performance alerts
        spy_return = market_summary.get('spy_daily_return', 0)
        if spy_return < -3:  # 3% daily drop
            self._create_alert(
                severity=AlertSeverity.HIGH,
                category="MARKET_DECLINE",
                title=f"Significant Market Decline: {spy_return:.1f}%",
                description=f"S&P 500 down {spy_return:.1f}% today",
                affected_assets=["SPY", "Broad Market"],
                recommended_actions=[
                    "Assess portfolio beta exposure",
                    "Look for buying opportunities",
                    "Monitor for trend continuation"
                ],
                confidence_score=0.9,
                data_source="market_performance"
            )
        
        # Sector rotation alerts
        sector_performance = market_summary.get('sector_performance', {})
        if sector_performance:
            worst_sector = market_summary.get('worst_sector')
            if worst_sector and worst_sector[1] < -5:  # 5% sector decline
                self._create_alert(
                    severity=AlertSeverity.MEDIUM,
                    category="SECTOR_DECLINE",
                    title=f"Sector Decline: {worst_sector[0]}",
                    description=f"{worst_sector[0]} sector down {worst_sector[1]:.1f}%",
                    affected_assets=[worst_sector[0]],
                    recommended_actions=[
                        f"Review {worst_sector[0]} sector exposure",
                        "Investigate sector-specific news",
                        "Consider sector rebalancing"
                    ],
                    confidence_score=0.8,
                    data_source="sector_performance"
                )
    
    def _check_economic_risks(self):
        """Check for economic indicator risks"""
        try:
            logger.info("🏛️ Checking economic risks...")
            econ_summary = self.econ_collector.get_economic_summary()
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            if econ_summary and 'data_date' in econ_summary:
                filename = f"economic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(f"{Config.RAW_DATA_DIR}/{filename}", "w") as f:
                    json.dump(econ_summary, f, default=str, indent=2)
                logger.info(f"Saved economic batch to {Config.RAW_DATA_DIR}/{filename}")
            if 'error' in econ_summary:
                logger.warning(f"Economic data error: {econ_summary['error']}")
                return
            self._analyze_economic_risks(econ_summary)
            self.last_checks['economic'] = datetime.now()
        except Exception as e:
            logger.error(f"Error checking economic risks: {e}", exc_info=True)
    
    def _analyze_economic_risks(self, econ_summary: Dict):
        """Analyze economic indicators for risks"""
        
        # Yield curve inversion
        yield_curve = econ_summary.get('yield_curve_2_10')
        if yield_curve is not None and yield_curve < 0:
            self._create_alert(
                severity=AlertSeverity.HIGH,
                category="YIELD_CURVE",
                title="Yield Curve Inversion Detected",
                description=f"2-10 year yield curve inverted by {yield_curve:.2f} basis points",
                affected_assets=["Bonds", "Financial Sector", "Overall Market"],
                recommended_actions=[
                    "Prepare for potential recession",
                    "Review duration risk",
                    "Consider defensive sectors"
                ],
                confidence_score=0.85,
                data_source="economic_indicators"
            )
        
        # Inflation concerns
        inflation_rate = econ_summary.get('inflation_rate')
        if inflation_rate is not None and inflation_rate > 4:
            self._create_alert(
                severity=AlertSeverity.MEDIUM,
                category="INFLATION",
                title=f"High Inflation: {inflation_rate:.1f}%",
                description=f"CPI inflation at {inflation_rate:.1f}%, above Fed target",
                affected_assets=["TIPS", "Real Assets", "Growth Stocks"],
                recommended_actions=[
                    "Consider inflation-protected securities",
                    "Review real asset allocation",
                    "Monitor Fed policy response"
                ],
                confidence_score=0.8,
                data_source="economic_indicators"
            )
        
        # Economic health assessment
        economic_health = econ_summary.get('economic_health', 'Unknown')
        if economic_health == 'High Risk':
            self._create_alert(
                severity=AlertSeverity.HIGH,
                category="ECONOMIC_HEALTH",
                title="Economic Health Deterioration",
                description="Multiple economic indicators showing stress",
                affected_assets=["Broad Market", "Credit Markets"],
                recommended_actions=[
                    "Increase cash allocation",
                    "Focus on quality investments",
                    "Monitor leading indicators"
                ],
                confidence_score=0.75,
                data_source="economic_health_assessment"
            )
    
    def _check_sector_risks(self, news_df: pd.DataFrame):
        """Check for sector-specific risks"""
        try:
            # Group by detected sectors/entities
            sector_news = {}
            
            for idx, row in news_df.iterrows():
                # Extract entities from LLM analysis if available
                entities = row.get('llm_entities', {})
                financial_terms = entities.get('financial_terms', [])
                
                # Simple sector mapping
                text = f"{row['title']} {row['description']}".lower()
                
                for sector in self.portfolio_context['sectors']:
                    keywords = Config.SECTOR_KEYWORDS.get(sector, [])
                    if any(keyword in text for keyword in keywords):
                        if sector not in sector_news:
                            sector_news[sector] = []
                        sector_news[sector].append(row)
            
            # Analyze risks for each sector
            for sector, articles in sector_news.items():
                if len(articles) < 2:  # Need multiple articles for pattern
                    continue
                
                sector_df = pd.DataFrame(articles)
                avg_sentiment = sector_df['sentiment_score'].mean()
                high_risk_count = (sector_df['risk_level'] == 'HIGH').sum()
                
                if avg_sentiment < -0.3 or high_risk_count > len(articles) * 0.5:
                    self._create_alert(
                        severity=AlertSeverity.MEDIUM,
                        category="SECTOR_RISK",
                        title=f"Sector Risk Alert: {sector}",
                        description=f"{len(articles)} negative articles about {sector} sector",
                        affected_assets=[sector],
                        recommended_actions=[
                            f"Review {sector} sector exposure",
                            f"Analyze {sector} sector fundamentals",
                            f"Consider {sector} sector hedging"
                        ],
                        confidence_score=0.7,
                        data_source="sector_news_analysis"
                    )
                    
        except Exception as e:
            logger.error(f"Error in sector risk analysis: {e}", exc_info=True)
    
    def _extract_affected_assets(self, news_df: pd.DataFrame) -> List[str]:
        """Extract affected assets from news articles"""
        assets = set()
        
        try:
            for _, row in news_df.iterrows():
                entities = row.get('llm_entities', {})
                companies = entities.get('companies', [])
                assets.update(companies)
                
                # Add sector mappings
                text = f"{row['title']} {row['description']}".lower()
                for sector in self.portfolio_context['sectors']:
                    if sector.lower() in text:
                        assets.add(sector)
        
        except Exception as e:
            logger.error(f"Error extracting affected assets: {e}")
        
        return list(assets)[:5]  # Limit to 5 assets
    
    def _create_alert(self, severity: AlertSeverity, category: str, title: str, 
                     description: str, affected_assets: List[str], 
                     recommended_actions: List[str], confidence_score: float, 
                     data_source: str):
        """Centralized alert creation using alert_utils.create_alert"""
        create_alert(self, severity, category, title, description, affected_assets, recommended_actions, confidence_score, data_source)
    
    def _is_duplicate_alert(self, new_alert: Alert) -> bool:
        """Check if alert is duplicate of recent alert"""
        recent_threshold = datetime.now() - timedelta(hours=2)
        
        for existing_alert in self.active_alerts:
            if (existing_alert.timestamp > recent_threshold and
                existing_alert.category == new_alert.category and
                existing_alert.title == new_alert.title):
                return True
        
        return False
    
    def _cleanup_expired_alerts(self):
        """Remove expired alerts"""
        now = datetime.now()
        self.active_alerts = [alert for alert in self.active_alerts 
                             if alert.expiry_time is None or alert.expiry_time > now]
    
    def _trigger_alert_callbacks(self, alert: Alert):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def mute_alert(self, alert_id: str):
        """Mute/acknowledge an alert by ID"""
        self.muted_alert_ids.add(alert_id)

    def unmute_alert(self, alert_id: str):
        """Unmute an alert by ID"""
        self.muted_alert_ids.discard(alert_id)

    def get_active_alerts(self) -> List[Alert]:
        """Get current active alerts (excluding muted)"""
        self._cleanup_expired_alerts()
        return sorted([a for a in self.active_alerts if a.id not in self.muted_alert_ids], key=lambda x: x.timestamp, reverse=True)

    def get_muted_alerts(self) -> List[Alert]:
        """Get currently muted alerts"""
        return [a for a in self.active_alerts if a.id in self.muted_alert_ids]

    def get_alert_summary(self) -> Dict:
        """Get summary of current alert status"""
        active = self.get_active_alerts()
        
        return {
            'total_active_alerts': len(active),
            'critical_alerts': len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            'high_alerts': len([a for a in active if a.severity == AlertSeverity.HIGH]),
            'medium_alerts': len([a for a in active if a.severity == AlertSeverity.MEDIUM]),
            'low_alerts': len([a for a in active if a.severity == AlertSeverity.LOW]),
            'latest_alert': active[0].timestamp.isoformat() if active else None,
            'monitoring_status': 'ACTIVE' if self.is_running else 'STOPPED',
            'last_checks': {k: v.isoformat() for k, v in self.last_checks.items()}
        }
    
    def save_alerts_to_file(self, filename: str = None):
        """Save alerts to file for persistence"""
        if filename is None:
            filename = f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        alerts_data = []
        for alert in self.alert_history:
            alerts_data.append({
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'category': alert.category,
                'title': alert.title,
                'description': alert.description,
                'affected_assets': alert.affected_assets,
                'recommended_actions': alert.recommended_actions,
                'confidence_score': alert.confidence_score,
                'data_source': alert.data_source,
                'expiry_time': alert.expiry_time.isoformat() if alert.expiry_time else None
            })
        
        filepath = f"{Config.PROCESSED_DATA_DIR}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        logger.info(f"Saved {len(alerts_data)} alerts to {filepath}")
        return filepath

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = PortfolioMonitoringAgent()
    print("Running immediate data collection for news, market, and economic data...")
    agent._check_news_risks()
    agent._check_market_risks()
    agent._check_economic_risks()
    print("Done. Check data/raw/ for output files.")
