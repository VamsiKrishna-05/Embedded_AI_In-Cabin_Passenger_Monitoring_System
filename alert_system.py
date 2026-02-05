import time
import json
from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.violence_detected = False
        self.alert_count = 0
        
    def send_alert(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = f"[{timestamp}] ALERT: {message}"
        
        print(f"🚨 {alert_message}")
        
        with open("alerts.log", "a") as f:
            f.write(f"{alert_message}\n")
        
        self.alert_count += 1
        
    def check_violence_status(self, violence_status, probability):
        if violence_status == "Violence Detected" and not self.violence_detected:
            self.violence_detected = True
            alert_message = f"Violence detected with probability {probability:.2f}"
            self.send_alert(alert_message)
            
        elif violence_status == "No Violence" and self.violence_detected:
            self.violence_detected = False
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Violence situation resolved")
    
    def get_alert_stats(self):
        return {
            'total_alerts': self.alert_count,
            'current_status': 'Alert Active' if self.violence_detected else 'Normal'
        }
