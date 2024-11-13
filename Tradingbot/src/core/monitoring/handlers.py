import logging
import smtplib
import json
import aiohttp
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime
from dataclasses import asdict

from .monitor import Alert, AlertLevel, ResourceType

class BaseAlertHandler:
    """Base class for alert handlers"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def handle(self, alert: Alert):
        """Handle alert notification"""
        try:
            await self._send_notification(alert)
            alert.resolved = await self._check_resolution(alert)
        except Exception as e:
            self.logger.error(f"Alert handling error: {str(e)}")
            
    async def _send_notification(self, alert: Alert):
        """Send notification - to be implemented by subclasses"""
        raise NotImplementedError

    async def _check_resolution(self, alert: Alert) -> bool:
        """Check if alert condition has been resolved"""
        return False

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert for notification"""
        resource_type = f"[{alert.resource_type.value}] " if alert.resource_type else ""
        return (
            f"ALERT [{alert.level.value}] {resource_type}- {alert.source}\n"
            f"Message: {alert.message}\n"
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Details: {json.dumps(alert.details, indent=2)}"
        )

class EmailAlertHandler(BaseAlertHandler):
    """Handles alerts via email"""
    
    async def _send_notification(self, alert: Alert):
        """Send email notification"""
        try:
            if not self._should_send_alert(alert):
                return
                
            message = MIMEMultipart()
            message["From"] = self.config.EMAIL_SENDER
            message["To"] = self.config.EMAIL_RECIPIENTS
            message["Subject"] = f"Trading Alert: {alert.level.value} - {alert.source}"
            
            body = self._format_alert_message(alert)
            message.attach(MIMEText(body, "plain"))
            
            async with aiohttp.ClientSession() as session:
                await self._send_email_async(message)
                
            self.logger.info(f"Email alert sent: {alert.level.value} - {alert.source}")
            
        except Exception as e:
            self.logger.error(f"Email alert error: {str(e)}")

    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent via email"""
        if not hasattr(self.config, 'EMAIL_ALERT_LEVELS'):
            return True
            
        return (
            alert.level.value in self.config.EMAIL_ALERT_LEVELS and
            (alert.resource_type is None or 
             alert.resource_type.value in self.config.EMAIL_RESOURCE_TYPES)
        )

    async def _send_email_async(self, message):
        """Send email asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            self._send_email_sync,
            message
        )

    def _send_email_sync(self, message):
        """Synchronous email sending"""
        with smtplib.SMTP_SSL(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
            server.login(self.config.SMTP_USERNAME, self.config.SMTP_PASSWORD)
            server.send_message(message)

class SlackAlertHandler(BaseAlertHandler):
    """Handles alerts via Slack"""
    
    async def _send_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            if not self._should_send_alert(alert):
                return
                
            message = self._format_slack_message(alert)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.SLACK_WEBHOOK_URL,
                    json=message,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API error: {await response.text()}")
                        
            self.logger.info(f"Slack alert sent: {alert.level.value} - {alert.source}")
            
        except Exception as e:
            self.logger.error(f"Slack alert error: {str(e)}")

    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent to Slack"""
        if not hasattr(self.config, 'SLACK_ALERT_LEVELS'):
            return True
            
        return (
            alert.level.value in self.config.SLACK_ALERT_LEVELS and
            (alert.resource_type is None or 
             alert.resource_type.value in self.config.SLACK_RESOURCE_TYPES)
        )

    def _format_slack_message(self, alert: Alert) -> Dict:
        """Format alert for Slack"""
        color = self._get_alert_color(alert.level)
        resource_type = f"[{alert.resource_type.value}] " if alert.resource_type else ""
        
        return {
            "attachments": [{
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🚨 Trading Alert: {alert.level.value} {resource_type}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Source:*\n{alert.source}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Message:*\n{alert.message}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Details:*\n```{json.dumps(alert.details, indent=2)}```"
                        }
                    }
                ]
            }]
        }
            
    def _get_alert_color(self, level: AlertLevel) -> str:
        """Get color for alert level"""
        colors = {
            AlertLevel.INFO: "#36a64f",  # Green
            AlertLevel.WARNING: "#ECB22E",  # Yellow
            AlertLevel.CRITICAL: "#E01E5A",  # Red
            AlertLevel.EMERGENCY: "#800000"  # Dark Red
        }
        return colors.get(level, "#000000")

class WebhookAlertHandler(BaseAlertHandler):
    """Handles alerts via custom webhook"""
    
    async def _send_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            if not self._should_send_alert(alert):
                return
                
            payload = {
                "level": alert.level.value,
                "source": alert.source,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resource_type": alert.resource_type.value if alert.resource_type else None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.WEBHOOK_URL,
                    json=payload,
                    headers=self.config.WEBHOOK_HEADERS
                ) as response:
                    if response.status not in [200, 201, 202]:
                        raise Exception(f"Webhook error: {await response.text()}")
                        
            self.logger.info(f"Webhook alert sent: {alert.level.value} - {alert.source}")
            
        except Exception as e:
            self.logger.error(f"Webhook alert error: {str(e)}")

    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent via webhook"""
        if not hasattr(self.config, 'WEBHOOK_ALERT_LEVELS'):
            return True
            
        return (
            alert.level.value in self.config.WEBHOOK_ALERT_LEVELS and
            (alert.resource_type is None or 
             alert.resource_type.value in self.config.WEBHOOK_RESOURCE_TYPES)
        )

class AlertHandlerManager:
    """Manages multiple alert handlers"""
    
    def __init__(self, config):
        self.config = config
        self.handlers = {}
        self.logger = logging.getLogger(__name__)
        self.alert_history = []
        self._initialize_handlers()
        
    def _initialize_handlers(self):
        """Initialize alert handlers based on config"""
        if hasattr(self.config, 'EMAIL_ALERTS_ENABLED') and self.config.EMAIL_ALERTS_ENABLED:
            self.handlers['email'] = EmailAlertHandler(self.config)
            
        if hasattr(self.config, 'SLACK_ALERTS_ENABLED') and self.config.SLACK_ALERTS_ENABLED:
            self.handlers['slack'] = SlackAlertHandler(self.config)
            
        if hasattr(self.config, 'WEBHOOK_ALERTS_ENABLED') and self.config.WEBHOOK_ALERTS_ENABLED:
            self.handlers['webhook'] = WebhookAlertHandler(self.config)
        
    async def handle_alert(self, alert: Alert):
        """Handle alert with all configured handlers"""
        try:
            self.alert_history.append(alert)
            
            # Trim history if needed
            if hasattr(self.config, 'MAX_ALERT_HISTORY'):
                self.alert_history = self.alert_history[-self.config.MAX_ALERT_HISTORY:]
            
            # Create tasks for all handlers
            tasks = [
                handler.handle(alert)
                for handler in self.handlers.values()
                if self._should_handle_alert(handler, alert)
            ]
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Alert handling error: {str(e)}")

    def _should_handle_alert(self, handler: BaseAlertHandler, alert: Alert) -> bool:
        """Check if handler should process alert"""
        if not hasattr(handler, '_should_send_alert'):
            return True
            
        try:
            return handler._should_send_alert(alert)
        except:
            return True

    def get_alert_history(self):
        """Get alert history"""
        return self.alert_history