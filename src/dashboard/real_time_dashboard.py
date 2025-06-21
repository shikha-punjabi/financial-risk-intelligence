<<<<<<< HEAD
=======
# src/dashboard/real_time_dashboard.py
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
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
<<<<<<< HEAD
    agent = st.session_state.monitoring_agent
    if agent:
        show_muted = st.sidebar.checkbox("Show Muted Alerts", value=False)
        if show_muted:
            alerts = agent.get_muted_alerts()
            st.sidebar.info("Showing muted/acknowledged alerts")
        else:
            alerts = agent.get_active_alerts()
=======
    
    if st.session_state.monitoring_agent:
        alerts = st.session_state.monitoring_agent.get_active_alerts()
        
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
        if alerts:
            for alert in alerts[:5]:  # Show top 5 alerts
                severity_colors = {
                    AlertSeverity.CRITICAL: "🔴",
                    AlertSeverity.HIGH: "🟠", 
                    AlertSeverity.MEDIUM: "🟡",
                    AlertSeverity.LOW: "🟢"
                }
<<<<<<< HEAD
                color = severity_colors.get(alert.severity, "⚪")
=======
                
                color = severity_colors.get(alert.severity, "⚪")
                
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
                with st.sidebar.expander(f"{color} {alert.title}", expanded=alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]):
                    st.write(f"**Time:** {alert.timestamp.strftime('%H:%M:%S')}")
                    st.write(f"**Category:** {alert.category}")
                    st.write(f"**Description:** {alert.description}")
                    st.write(f"**Confidence:** {alert.confidence_score:.1%}")
<<<<<<< HEAD
                    if alert.affected_assets:
                        st.write(f"**Affected:** {', '.join(alert.affected_assets)}")
=======
                    
                    if alert.affected_assets:
                        st.write(f"**Affected:** {', '.join(alert.affected_assets)}")
                    
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
                    if alert.recommended_actions:
                        st.write("**Actions:**")
                        for action in alert.recommended_actions:
                            st.write(f"• {action}")
<<<<<<< HEAD
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
=======
        else:
            st.sidebar.info("No active alerts")
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
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
<<<<<<< HEAD
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
=======
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

def render_market_overview(market_data):
   """Render market overview tab"""
   if not market_data:
       st.warning("No market data available")
       return
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.subheader("Market Performance")
       
       # Sector performance chart
       sector_performance = market_data.get('market_summary', {}).get('sector_performance', {})
       if sector_performance:
           sector_df = pd.DataFrame(list(sector_performance.items()), 
                                  columns=['Sector', 'Performance'])
           
           fig = px.bar(sector_df, x='Sector', y='Performance', 
                       title="Sector Performance Today (%)",
                       color='Performance',
                       color_continuous_scale='RdYlGn')
           fig.update_layout(xaxis_tickangle=-45)
           st.plotly_chart(fig, use_container_width=True)
       
       # Market indices
       st.subheader("Major Indices")
       indices_data = market_data.get('market_data', {})
       
       for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
           if symbol in indices_data:
               data = indices_data[symbol]
               price = data.get('current_price', 0)
               returns = data.get('returns')
               
               if returns is not None and len(returns) > 0:
                   daily_return = returns.iloc[-1] * 100
                   st.metric(symbol, f"${price:.2f}", f"{daily_return:.2f}%")
   
   with col2:
       st.subheader("Risk Indicators")
       
       # VIX chart
       vix_data = indices_data.get('^VIX', {})
       if 'price_data' in vix_data:
           vix_hist = vix_data['price_data']
           
           fig = go.Figure()
           fig.add_trace(go.Scatter(x=vix_hist.index, y=vix_hist['Close'],
                                  mode='lines', name='VIX'))
           fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                        annotation_text="Normal Level")
           fig.add_hline(y=30, line_dash="dash", line_color="red",
                        annotation_text="High Volatility")
           fig.update_layout(title="VIX - Market Fear Index", 
                           yaxis_title="VIX Level")
           st.plotly_chart(fig, use_container_width=True)
       
       # Market breadth indicators
       market_summary = market_data.get('market_summary', {})
       
       st.write("**Market Health Indicators:**")
       
       fear_level = market_summary.get('market_fear_level', 'Unknown')
       st.write(f"• Fear Level: {fear_level}")
       
       if 'best_sector' in market_summary and market_summary['best_sector']:
           best = market_summary['best_sector']
           st.write(f"• Best Sector: {best[0]} (+{best[1]:.1f}%)")
       
       if 'worst_sector' in market_summary and market_summary['worst_sector']:
           worst = market_summary['worst_sector']
           st.write(f"• Worst Sector: {worst[0]} ({worst[1]:.1f}%)")

def render_news_analysis(news_df):
   """Render news analysis tab"""
   if news_df.empty:
       st.warning("No recent news available")
       return
   
   col1, col2 = st.columns([2, 1])
   
   with col1:
       st.subheader("Sentiment Timeline")
       
       # Sentiment over time
       news_df['published_dt'] = pd.to_datetime(news_df['published_at'])
       news_df_sorted = news_df.sort_values('published_dt')
       
       fig = go.Figure()
       
       # Sentiment line
       fig.add_trace(go.Scatter(
           x=news_df_sorted['published_dt'],
           y=news_df_sorted['sentiment_score'],
           mode='lines+markers',
           name='Sentiment Score',
           line=dict(color='blue')
       ))
       
       # Add risk level colors
       risk_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
       for risk_level, color in risk_colors.items():
           risk_articles = news_df_sorted[news_df_sorted['risk_level'] == risk_level]
           if not risk_articles.empty:
               fig.add_trace(go.Scatter(
                   x=risk_articles['published_dt'],
                   y=risk_articles['sentiment_score'],
                   mode='markers',
                   name=f'{risk_level} Risk',
                   marker=dict(color=color, size=8),
                   showlegend=True
               ))
       
       fig.add_hline(y=0, line_dash="dash", line_color="gray")
       fig.update_layout(
           title="News Sentiment Timeline",
           xaxis_title="Time",
           yaxis_title="Sentiment Score",
           hovermode='x unified'
       )
       st.plotly_chart(fig, use_container_width=True)
       
       # Recent articles table
       st.subheader("Recent Articles")
       
       display_cols = ['title', 'source', 'sentiment_score', 'risk_level', 'published_at']
       display_df = news_df[display_cols].head(10).copy()
       display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
       display_df['published_at'] = pd.to_datetime(display_df['published_at']).dt.strftime('%H:%M')
       
       st.dataframe(
           display_df,
           column_config={
               "title": st.column_config.TextColumn("Title", width="large"),
               "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
               "risk_level": st.column_config.SelectboxColumn("Risk"),
           },
           use_container_width=True
       )
   
   with col2:
       st.subheader("News Summary")
       
       # Sentiment distribution
       sentiment_ranges = pd.cut(news_df['sentiment_score'], 
                               bins=[-1, -0.3, -0.1, 0.1, 0.3, 1],
                               labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
       
       sentiment_counts = sentiment_ranges.value_counts()
       
       fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                   title="Sentiment Distribution")
       st.plotly_chart(fig, use_container_width=True)
       
       # Risk level distribution
       risk_counts = news_df['risk_level'].value_counts()
       
       fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                   title="Risk Level Distribution",
                   color=risk_counts.index,
                   color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'})
       st.plotly_chart(fig, use_container_width=True)
       
       # Top sources
       st.write("**Top News Sources:**")
       source_counts = news_df['source'].value_counts().head(5)
       for source, count in source_counts.items():
           st.write(f"• {source}: {count} articles")
       
       # Key statistics
       st.write("**Statistics:**")
       st.write(f"• Total Articles: {len(news_df)}")
       st.write(f"• Average Sentiment: {news_df['sentiment_score'].mean():.3f}")
       st.write(f"• High Risk Articles: {(news_df['risk_level'] == 'HIGH').sum()}")
       st.write(f"• Time Range: {news_df['published_dt'].min().strftime('%H:%M')} - {news_df['published_dt'].max().strftime('%H:%M')}")

def render_risk_alerts():
   """Render risk alerts tab"""
   if not st.session_state.monitoring_agent:
       st.warning("Monitoring agent not initialized")
       return
   
   agent = st.session_state.monitoring_agent
   alerts = agent.get_active_alerts()
   
   if not alerts:
       st.info("🟢 No active risk alerts - All systems normal")
       return
   
   # Alert summary metrics
   col1, col2, col3, col4 = st.columns(4)
   
   alert_counts = {}
   for severity in AlertSeverity:
       alert_counts[severity] = len([a for a in alerts if a.severity == severity])
   
   with col1:
       st.metric("Critical", alert_counts[AlertSeverity.CRITICAL])
   with col2:
       st.metric("High", alert_counts[AlertSeverity.HIGH])
   with col3:
       st.metric("Medium", alert_counts[AlertSeverity.MEDIUM])
   with col4:
       st.metric("Low", alert_counts[AlertSeverity.LOW])
   
   # Alert timeline
   st.subheader("Alert Timeline")
   
   if alerts:
       alert_timeline_data = []
       for alert in alerts:
           alert_timeline_data.append({
               'time': alert.timestamp,
               'severity': alert.severity.value,
               'category': alert.category,
               'title': alert.title
           })
       
       timeline_df = pd.DataFrame(alert_timeline_data)
       
       fig = px.scatter(timeline_df, x='time', y='category', 
                       color='severity', size_max=15,
                       title="Alert Timeline",
                       color_discrete_map={
                           'CRITICAL': 'red',
                           'HIGH': 'orange', 
                           'MEDIUM': 'yellow',
                           'LOW': 'green'
                       })
       st.plotly_chart(fig, use_container_width=True)
   
   # Detailed alerts
   st.subheader("Active Alerts")
   
   for alert in alerts:
       severity_colors = {
           AlertSeverity.CRITICAL: "🔴",
           AlertSeverity.HIGH: "🟠",
           AlertSeverity.MEDIUM: "🟡", 
           AlertSeverity.LOW: "🟢"
       }
       
       color = severity_colors.get(alert.severity, "⚪")
       
       with st.expander(f"{color} {alert.title} - {alert.timestamp.strftime('%H:%M:%S')}", 
                       expanded=alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]):
           
           col1, col2 = st.columns([2, 1])
           
           with col1:
               st.write(f"**Description:** {alert.description}")
               st.write(f"**Category:** {alert.category}")
               st.write(f"**Data Source:** {alert.data_source}")
               
               if alert.affected_assets:
                   st.write(f"**Affected Assets:** {', '.join(alert.affected_assets)}")
               
               if alert.recommended_actions:
                   st.write("**Recommended Actions:**")
                   for action in alert.recommended_actions:
                       st.write(f"• {action}")
           
           with col2:
               st.metric("Confidence", f"{alert.confidence_score:.1%}")
               st.write(f"**Severity:** {alert.severity.value}")
               st.write(f"**Created:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
               
               if alert.expiry_time:
                   time_left = alert.expiry_time - datetime.now()
                   hours_left = time_left.total_seconds() / 3600
                   st.write(f"**Expires in:** {hours_left:.1f} hours")

def render_portfolio_impact(market_data, news_df):
   """Render portfolio impact analysis tab"""
   st.subheader("Portfolio Impact Analysis")
   
   # Simulated portfolio (in real app, this would come from user data)
   portfolio = {
       'AAPL': {'shares': 100, 'sector': 'Technology'},
       'MSFT': {'shares': 75, 'sector': 'Technology'},
       'JPM': {'shares': 50, 'sector': 'Financial'},
       'JNJ': {'shares': 60, 'sector': 'Healthcare'},
       'XOM': {'shares': 40, 'sector': 'Energy'}
   }
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.write("**Current Portfolio Exposure:**")
       
       # Portfolio composition
       sectors = {}
       for symbol, data in portfolio.items():
           sector = data['sector']
           if sector not in sectors:
               sectors[sector] = 0
           sectors[sector] += data['shares']
       
       sector_df = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Exposure'])
       
       fig = px.pie(sector_df, values='Exposure', names='Sector',
                   title="Portfolio Sector Allocation")
       st.plotly_chart(fig, use_container_width=True)
       
       # Risk assessment by sector
       st.write("**Sector Risk Assessment:**")
       
       if not news_df.empty:
           for sector in sectors.keys():
               sector_news = news_df[news_df['combined_text'].str.contains(sector.lower(), case=False, na=False)]
               
               if not sector_news.empty:
                   avg_sentiment = sector_news['sentiment_score'].mean()
                   risk_articles = (sector_news['risk_level'] == 'HIGH').sum()
                   
                   risk_level = "🔴 High" if avg_sentiment < -0.2 or risk_articles > 2 else \
                               "🟡 Medium" if avg_sentiment < 0 or risk_articles > 0 else "🟢 Low"
                   
                   st.write(f"• **{sector}:** {risk_level}")
                   st.write(f"  - Sentiment: {avg_sentiment:.3f}")
                   st.write(f"  - High-risk articles: {risk_articles}")
               else:
                   st.write(f"• **{sector}:** 🔵 No recent news")
   
   with col2:
       st.write("**Impact Recommendations:**")
       
       recommendations = []
       
       # Generate recommendations based on alerts and analysis
       if st.session_state.monitoring_agent:
           alerts = st.session_state.monitoring_agent.get_active_alerts()
           
           critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
           if critical_alerts:
               recommendations.append("🚨 **IMMEDIATE ACTION REQUIRED**")
               recommendations.extend([f"• {action}" for alert in critical_alerts for action in alert.recommended_actions])
           
           high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
           if high_alerts:
               recommendations.append("⚠️ **HIGH PRIORITY ACTIONS**")
               recommendations.extend([f"• {action}" for alert in high_alerts for action in alert.recommended_actions])
       
       # Market-based recommendations
       if market_data and 'market_summary' in market_data:
           vix_level = market_data['market_summary'].get('vix_level', 20)
           spy_return = market_data['market_summary'].get('spy_daily_return', 0)
           
           if vix_level > 30:
               recommendations.append("📉 **High Volatility Detected**")
               recommendations.append("• Consider reducing position sizes")
               recommendations.append("• Review stop-loss orders")
           
           if spy_return < -2:
               recommendations.append("📊 **Market Decline Alert**")
               recommendations.append("• Monitor for buying opportunities")
               recommendations.append("• Check portfolio correlation")
       
       # Sentiment-based recommendations
       if not news_df.empty:
           avg_sentiment = news_df['sentiment_score'].mean()
           if avg_sentiment < -0.3:
               recommendations.append("📰 **Negative News Sentiment**")
               recommendations.append("• Increase cash allocation")
               recommendations.append("• Focus on defensive sectors")
       
       if recommendations:
           for rec in recommendations:
               st.write(rec)
       else:
           st.info("✅ No immediate action required - Portfolio appears stable")
       
       # Performance tracking
       st.write("**Recent Portfolio Metrics:**")
       
       # Simulated metrics
       st.metric("Estimated Daily P&L", "$-1,234", "-0.8%")
       st.metric("Portfolio Beta", "1.15", "+0.05")
       st.metric("Risk Score", "Medium", "↑")

def main():
   """Main application function"""
   try:
       # Render sidebar controls first
       render_monitoring_controls()
       render_alert_sidebar()
       
       # Render main dashboard
       render_main_dashboard()
       
       # Update last refresh time
       st.session_state.last_refresh = datetime.now()
       
       # Show last update time
       st.sidebar.write(f"**Last Updated:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")
       
   except Exception as e:
       st.error(f"Application error: {e}")
       st.write("Please refresh the page and try again.")

if __name__ == "__main__":
   main()
>>>>>>> c3a7bb2 (Initial commit: AI-powered financial risk intelligence platform)
