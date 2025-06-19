import time
import threading
from datetime import datetime
from src.agents.portfolio_monitor import PortfolioMonitoringAgent
from config.settings import Config

def test_complete_system():
    """Test the complete monitoring system"""
    
    print("🧪 Testing Complete Financial Risk Intelligence System")
    print("=" * 60)
    
    try:
        # Validate configuration
        Config.validate_keys()
        print("✅ Configuration validated")
        
        # Initialize monitoring agent
        agent = PortfolioMonitoringAgent()
        print("✅ Monitoring agent initialized")
        
        # Add test callback for alerts
        alerts_received = []
        def test_callback(alert):
            alerts_received.append(alert)
            print(f"🚨 Alert received: {alert.title} [{alert.severity.value}]")
        
        agent.add_alert_callback(test_callback)
        
        # Start monitoring
        agent.start_monitoring()
        print("✅ Monitoring started")

        # Force immediate data collection
        print("Forcing immediate data collection...")
        agent._check_news_risks()
        agent._check_market_risks()
        agent._check_economic_risks()

        # Wait for some monitoring cycles
        print("⏳ Running monitoring for 2 minutes...")
        time.sleep(120)  # 2 minutes
        
        # Check results
        alert_summary = agent.get_alert_summary()
        print("\n📊 Test Results:")
        print(f"• Monitoring Status: {alert_summary['monitoring_status']}")
        print(f"• Total Active Alerts: {alert_summary['total_active_alerts']}")
        print(f"• Critical Alerts: {alert_summary['critical_alerts']}")
        print(f"• High Alerts: {alert_summary['high_alerts']}")
        print(f"• Medium Alerts: {alert_summary['medium_alerts']}")
        print(f"• Low Alerts: {alert_summary['low_alerts']}")
        
        # Test data collection
        print(f"\n🔍 Last Data Collection:")
        for source, timestamp in alert_summary['last_checks'].items():
            print(f"• {source}: {timestamp}")
        
        # Save test results
        agent.save_alerts_to_file("test_alerts.json")
        print("✅ Test results saved")
        
        # Stop monitoring
        agent.stop_monitoring()
        print("✅ Monitoring stopped")
        
        print("\n🎉 Complete system test passed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

if __name__ == "__main__":
    test_complete_system()
