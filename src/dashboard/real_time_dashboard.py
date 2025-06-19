import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List

# Import our modules
from src.agents.portfolio_monitor import PortfolioMonitoringAgent, AlertSeverity
from src.data.news_collector import NewsCollector
from src.data.market_collector import MarketDataCollector
from src.models.free_sentiment_analyzer import FreeSentimentAnalyzer
from config.settings import Config

# Page config
st.set_page_config(
    page_title="Financial Risk Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'monitoring_agent' not in st.session_state:
    st.session_state.monitoring_agent = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

def initialize_monitoring_agent():
    """Initialize the monitoring agent"""
    if st.session_state.monitoring_agent is None:
        try:
            Config.validate_keys()
            st.session_state.monitoring_agent = PortfolioMonitoringAgent()
            
            # Add callback for real-time alerts
            def alert_callback(alert):
                if 'new_alerts' not in st.session_state:
                    st.session_state.new_alerts = []
                st.session_state.new_alerts.append(alert)
            
            st.session_state.monitoring_agent.add_alert_callback(alert_callback)
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize monitoring agent: {e}")
            return False
    return True

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_market_data():
    """Load current market data"""
    try:
        market_collector = MarketDataCollector()
        return market_collector.get_comprehensive_market_snapshot()
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return {}

@st.cache_data(ttl=600)  # Cache for 10 minutes  
def load_recent_news():
    """Load recent financial news"""
    try:
        news_collector = NewsCollector(Config.NEWS_API_KEY)
        sentiment_analyzer = FreeSentimentAnalyzer()
        
        news_df = news_collector.get_financial_news(hours_back=6)
        if not news_df.empty:
            analyzed_news = sentiment_analyzer.analyze_news_batch(news_df)
            return analyzed_news
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading news: {e}")
        return pd.DataFrame()

def render_alert_sidebar():
    """Render alerts in sidebar"""
    st.sidebar.header("🚨 Real-time Alerts")
    agent = st.session_state.monitoring_agent
    if agent:
        show_muted = st.sidebar.checkbox("Show Muted Alerts", value=False)
        if show_muted:
            alerts = agent.get_muted_alerts()
            st.sidebar.info("Showing muted/acknowledged alerts")
        else:
            alerts = agent.get_active_alerts()
        if alerts:
            for alert in alerts[:5]:  # Show top 5 alerts
                severity_colors = {
                    AlertSeverity.CRITICAL: "🔴",
                    AlertSeverity.HIGH: "🟠", 
                    AlertSeverity.MEDIUM: "🟡",
                    AlertSeverity.LOW: "🟢"
                }
                color = severity_colors.get(alert.severity, "⚪")
                with st.sidebar.expander(f"{color} {alert.title}", expanded=alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]):
                    st.write(f"**Time:** {alert.timestamp.strftime('%H:%M:%S')}")
                    st.write(f"**Category:** {alert.category}")
                    st.write(f"**Description:** {alert.description}")
                    st.write(f"**Confidence:** {alert.confidence_score:.1%}")
                    if alert.affected_assets:
                        st.write(f"**Affected:** {', '.join(alert.affected_assets)}")
                    if alert.recommended_actions:
                        st.write("**Actions:**")
                        for action in alert.recommended_actions:
                            st.write(f"• {action}")
                    # Mute/unmute button
                    if alert.id in agent.muted_alert_ids:
                        if st.button(f"Unmute {alert.id}", key=f"unmute_{alert.id}"):
                            agent.unmute_alert(alert.id)
                            st.experimental_rerun()
                    else:
                        if st.button(f"Mute {alert.id}", key=f"mute_{alert.id}"):
                            agent.mute_alert(alert.id)
                            st.experimental_rerun()
        else:
            st.sidebar.info("No active alerts" if not show_muted else "No muted alerts")
    else:
        st.sidebar.warning("Monitoring agent not initialized")

def render_monitoring_controls():
    """Render monitoring controls"""
    st.sidebar.header("⚙️ Monitoring Controls")
    
    if initialize_monitoring_agent():
        agent = st.session_state.monitoring_agent
        
        # Start/Stop monitoring
        if agent.is_running:
            if st.sidebar.button("🛑 Stop Monitoring"):
                agent.stop_monitoring()
                st.sidebar.success("Monitoring stopped")
        else:
            if st.sidebar.button("🚀 Start Monitoring"):
                agent.start_monitoring()
                st.sidebar.success("Monitoring started")
        
        # Monitoring status
        status = "🟢 ACTIVE" if agent.is_running else "🔴 STOPPED"
        st.sidebar.write(f"**Status:** {status}")
        
        # Alert summary
        alert_summary = agent.get_alert_summary()
        st.sidebar.write(f"**Active Alerts:** {alert_summary['total_active_alerts']}")
        
        if alert_summary['total_active_alerts'] > 0:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Critical", alert_summary['critical_alerts'])
                st.metric("Medium", alert_summary['medium_alerts'])
            with col2:
                st.metric("High", alert_summary['high_alerts'])
                st.metric("Low", alert_summary['low_alerts'])
    
    # Auto-refresh control
    st.sidebar.header("🔄 Auto-refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh every 30 seconds
        time.sleep(30)
        st.rerun()

def render_main_dashboard():
    """Render main dashboard content"""
    st.title("🔍 Financial Risk Intelligence Platform")
    st.markdown("*Real-time AI-powered financial risk monitoring*")
    
    # Load data
    with st.spinner("Loading real-time data..."):
        market_data = load_market_data()
        news_df = load_recent_news()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if market_data and 'market_summary' in market_data:
            spy_return = market_data['market_summary'].get('spy_daily_return', 0)
            st.metric("S&P 500 Daily", f"{spy_return:.2f}%", 
                     delta=f"{spy_return:.2f}%")
        else:
            st.metric("S&P 500 Daily", "N/A")
    
    with col2:
        if market_data and 'market_summary' in market_data:
            vix_level = market_data['market_summary'].get('vix_level', 20)
            st.metric("VIX Level", f"{vix_level:.1f}", 
                     delta="High" if vix_level > 25 else "Normal")
        else:
            st.metric("VIX Level", "N/A")
    
    with col3:
        if not news_df.empty:
            avg_sentiment = news_df['sentiment_score'].mean()
            st.metric("News Sentiment", f"{avg_sentiment:.3f}",
                     delta="Positive" if avg_sentiment > 0 else "Negative")
        else:
            st.metric("News Sentiment", "N/A")
    
    with col4:
        if not news_df.empty:
            high_risk_count = (news_df['risk_level'] == 'HIGH').sum()
            st.metric("High Risk Articles", f"{high_risk_count}")
        else:
            st.metric("High Risk Articles", "0")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📰 News Analysis", "🚨 Risk Alerts", "📈 Portfolio Impact"])
    
    with tab1:
        render_market_overview(market_data)
    
    with tab2:
        render_news_analysis(news_df)
    
    with tab3:
        render_risk_alerts()
    
    with tab4:
        render_portfolio_impact(market_data, news_df)

# ...existing code for render_market_overview, render_news_analysis, render_risk_alerts, render_portfolio_impact, main ...
