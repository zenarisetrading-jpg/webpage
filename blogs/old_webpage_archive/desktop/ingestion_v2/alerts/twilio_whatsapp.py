"""
Twilio WhatsApp Alerter
=======================
Sends alerts via WhatsApp using Twilio API.

PRD Reference: EMAIL_INGESTION_PRD.md Section 16

Alert Severity:
- CRITICAL: System-wide failure
- HIGH: Single account ingestion failure
- MEDIUM: Data late (in-app only, no WhatsApp)
"""

import os
from typing import Optional

from ..constants import AlertSeverity


class TwilioWhatsAppAlerter:
    """
    WhatsApp alerter using Twilio API.
    
    Sends alerts for ingestion failures.
    """
    
    def __init__(
        self,
        account_sid: str = None,
        auth_token: str = None,
        from_number: str = None,
        to_number: str = None
    ):
        """
        Initialize Twilio WhatsApp alerter.
        
        Args:
            account_sid: Twilio Account SID (default from TWILIO_ACCOUNT_SID env)
            auth_token: Twilio Auth Token (default from TWILIO_AUTH_TOKEN env)
            from_number: WhatsApp sender (default from TWILIO_WHATSAPP_FROM env)
            to_number: WhatsApp recipient (default from TWILIO_WHATSAPP_TO env)
        """
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_WHATSAPP_FROM")
        self.to_number = to_number or os.getenv("TWILIO_WHATSAPP_TO")
        
        self._client = None
        self._enabled = all([
            self.account_sid,
            self.auth_token,
            self.from_number,
            self.to_number
        ])
    
    def _get_client(self):
        """Get or create Twilio client."""
        if self._client is None:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
            except ImportError:
                print("Warning: twilio not installed. Run: pip install twilio")
                return None
            except Exception as e:
                print(f"Warning: Failed to create Twilio client: {e}")
                return None
        return self._client
    
    def _format_message(
        self, 
        severity: str, 
        message: str, 
        account_id: Optional[str] = None
    ) -> str:
        """
        Format alert message with severity prefix.
        
        Args:
            severity: AlertSeverity value
            message: Alert message
            account_id: Optional account identifier
            
        Returns:
            Formatted message string
        """
        severity_emoji = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.HIGH: "⚠️",
            AlertSeverity.MEDIUM: "ℹ️",
        }
        
        emoji = severity_emoji.get(severity, "📢")
        prefix = f"{emoji} [{severity}]"
        
        if account_id:
            prefix += f" Account: {account_id}"
        
        return f"{prefix}\n\n{message}"
    
    def send_alert(
        self,
        severity: str,
        message: str,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Send alert via WhatsApp.
        
        PRD Section 16:
        - CRITICAL: Slack + Email (we use WhatsApp)
        - HIGH: Slack (we use WhatsApp)
        - MEDIUM: In-App only (no WhatsApp)
        
        Args:
            severity: AlertSeverity value
            message: Alert message
            account_id: Optional account identifier
            
        Returns:
            True if sent successfully, False otherwise
        """
        # MEDIUM severity = in-app only, skip WhatsApp
        if severity == AlertSeverity.MEDIUM:
            print(f"[MEDIUM] {message} (in-app only, no WhatsApp)")
            return True
        
        if not self._enabled:
            print(f"[{severity}] Alert (WhatsApp disabled): {message}")
            return False
        
        try:
            client = self._get_client()
            if not client:
                return False
            
            formatted_message = self._format_message(severity, message, account_id)
            
            # Send via WhatsApp
            result = client.messages.create(
                body=formatted_message,
                from_=self.from_number,
                to=self.to_number
            )
            
            print(f"[{severity}] Alert sent: {result.sid}")
            return True
            
        except Exception as e:
            print(f"[{severity}] Alert failed: {e}")
            return False
    
    def send_critical(self, message: str, account_id: Optional[str] = None) -> bool:
        """Send CRITICAL alert."""
        return self.send_alert(AlertSeverity.CRITICAL, message, account_id)
    
    def send_high(self, message: str, account_id: Optional[str] = None) -> bool:
        """Send HIGH alert."""
        return self.send_alert(AlertSeverity.HIGH, message, account_id)
    
    def send_ingestion_failed(
        self,
        account_id: str,
        reason: str,
        filename: Optional[str] = None
    ) -> bool:
        """
        Send alert for ingestion failure.
        
        Args:
            account_id: Account identifier
            reason: Failure reason
            filename: Optional filename that failed
            
        Returns:
            True if sent successfully
        """
        message = f"Ingestion failed"
        if filename:
            message += f" for file: {filename}"
        message += f"\n\nReason: {reason}"
        
        return self.send_high(message, account_id)
    
    def send_validation_rejected(
        self,
        account_id: str,
        sender: str,
        errors: list
    ) -> bool:
        """
        Send alert for validation rejection.
        
        Args:
            account_id: Account identifier
            sender: Email sender that was rejected
            errors: List of validation errors
            
        Returns:
            True if sent successfully
        """
        message = f"Email rejected from: {sender}\n\nErrors:\n"
        for err in errors:
            message += f"• {err}\n"
        
        return self.send_high(message, account_id)
