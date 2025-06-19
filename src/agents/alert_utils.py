import logging
logger = logging.getLogger(__name__)

def create_alert(self, severity, category, title, description, affected_assets,
                 recommended_actions, confidence_score, data_source):
    """
    Centralized alert logic: always create and append Alert objects, never dicts.
    """
    from datetime import datetime, timedelta
    # Rate limiting
    recent_alerts = [a for a in getattr(self, 'alert_history', [])
                     if hasattr(a, 'timestamp') and a.timestamp > datetime.now() - timedelta(hours=1)]
    if hasattr(self, 'config') and len(recent_alerts) >= self.config.get('max_alerts_per_hour', 10):
        logger.warning("Alert rate limit reached, skipping alert creation")
        return
    # Always use Alert class
    Alert = getattr(self, 'Alert', None)
    alert_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Alert:
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            description=description,
            affected_assets=affected_assets,
            recommended_actions=recommended_actions,
            confidence_score=confidence_score,
            data_source=data_source,
            expiry_time=datetime.now() + timedelta(hours=24)
        )
        # Duplicate check
        recent_threshold = datetime.now() - timedelta(hours=2)
        for existing_alert in getattr(self, 'active_alerts', []):
            if (existing_alert.timestamp > recent_threshold and
                existing_alert.category == alert.category and
                existing_alert.title == alert.title):
                logger.info("Duplicate alert detected, skipping.")
                return
        # Append to lists
        if hasattr(self, 'active_alerts'):
            self.active_alerts.append(alert)
        if hasattr(self, 'alert_history'):
            self.alert_history.append(alert)
        logger.info(f"🚨 {severity} Alert Created: {title}")
        # Trigger callbacks if present
        for callback in getattr(self, 'alert_callbacks', []):
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}", exc_info=True)
    else:
        raise RuntimeError("Alert class not found on agent. Cannot create alert.")
