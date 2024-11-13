# File: config/alert_config.py

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AlertConfig:
    """Configuration for alert system"""
    
    # General alert settings
    ALERT_COOLDOWN_PERIOD: int = 300  # Seconds between similar alerts
    MAX_ALERTS_PER_HOUR: int = 50
    MAX_ALERT_HISTORY: int = 1000
    
    # Email settings
    EMAIL_ALERTS_ENABLED: bool = True
    EMAIL_ALERT_LEVELS: List[str] = [
        "CRITICAL",
        "EMERGENCY"
    ]
    EMAIL_SENDER: str = "trading.bot@example.com"
    EMAIL_RECIPIENTS: str = "trader@example.com"
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 465
    SMTP_USERNAME: str = "your-email@gmail.com"
    SMTP_PASSWORD: str = "your-app-specific-password"
    
    # Slack settings
    SLACK_ALERTS_ENABLED: bool = True
    SLACK_ALERT_LEVELS: List[str] = [
        "WARNING",
        "CRITICAL",
        "EMERGENCY"
    ]
    SLACK_WEBHOOK_URL: str = "https://hooks.slack.com/services/your/webhook/url"
    
    # Webhook settings
    WEBHOOK_ALERTS_ENABLED: bool = False
    WEBHOOK_ALERT_LEVELS: List[str] = [
        "WARNING",
        "CRITICAL",
        "EMERGENCY"
    ]
    WEBHOOK_URL: str = "https://your-webhook-url"
    WEBHOOK_HEADERS: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": "Bearer your-token"
    }
