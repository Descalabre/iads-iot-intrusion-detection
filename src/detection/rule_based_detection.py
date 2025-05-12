"""
Rule-based detection module for the Intrusion and Anomaly Detection System (IADS).
Implements various detection rules and thresholds for identifying anomalies and intrusions.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from src.core.config import logger, config
from src.core.db import DetectionEvents

@dataclass
class DetectionRule:
    """Class for defining detection rules."""
    name: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True

@dataclass
class DetectionResult:
    """Class for storing detection results."""
    timestamp: datetime
    rule_name: str
    severity: str
    description: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    additional_info: Dict[str, Any] = None

class RuleBasedDetector:
    """Implements rule-based detection logic for anomalies and intrusions."""
    
    def __init__(self):
        """Initialize the detector with default rules and thresholds."""
        # Load configuration
        self.config = config.get('detection', {})
        
        # Initialize rules
        self.rules: List[DetectionRule] = [
            DetectionRule(
                name="high_traffic_volume",
                description="Detect abnormally high network traffic volume",
                severity="medium"
            ),
            DetectionRule(
                name="repeated_failed_login",
                description="Detect multiple failed login attempts",
                severity="high"
            ),
            DetectionRule(
                name="unusual_port_access",
                description="Detect access to unusual ports",
                severity="medium"
            ),
            DetectionRule(
                name="data_exfiltration",
                description="Detect potential data exfiltration attempts",
                severity="critical"
            ),
            DetectionRule(
                name="suspicious_process",
                description="Detect suspicious process behavior",
                severity="high"
            )
        ]
        
        # Load thresholds from config or use defaults
        self.thresholds = {
            'traffic_volume_threshold': self.config.get('traffic_volume_threshold', 1000),
            'failed_login_threshold': self.config.get('failed_login_threshold', 5),
            'failed_login_window': self.config.get('failed_login_window', 300),  # 5 minutes
            'unusual_port_threshold': self.config.get('unusual_port_threshold', 0.01),
            'data_transfer_threshold': self.config.get('data_transfer_threshold', 100000000),  # 100MB
            'process_cpu_threshold': self.config.get('process_cpu_threshold', 90)
        }

    def detect_high_traffic(self, data: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect abnormally high network traffic volume.
        
        Args:
            data: DataFrame containing network traffic data
            
        Returns:
            List of DetectionResult objects
        """
        try:
            results = []
            if 'bytes_transferred' not in data.columns:
                logger.warning("bytes_transferred column not found in data")
                return results

            # Calculate rolling mean and standard deviation
            window_size = self.config.get('traffic_window_size', 100)
            rolling_mean = data['bytes_transferred'].rolling(window=window_size).mean()
            rolling_std = data['bytes_transferred'].rolling(window=window_size).std()
            
            # Detect anomalies (values more than 3 standard deviations from mean)
            anomalies = data[data['bytes_transferred'] > (rolling_mean + 3 * rolling_std)]
            
            for _, row in anomalies.iterrows():
                results.append(
                    DetectionResult(
                        timestamp=row.get('timestamp', datetime.now()),
                        rule_name="high_traffic_volume",
                        severity="medium",
                        description=f"Abnormally high traffic detected: {row['bytes_transferred']} bytes",
                        source_ip=row.get('source_ip'),
                        destination_ip=row.get('destination_ip'),
                        additional_info={'threshold': float(rolling_mean + 3 * rolling_std)}
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in high traffic detection: {str(e)}")
            raise

    def detect_failed_logins(self, data: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect multiple failed login attempts.
        
        Args:
            data: DataFrame containing login attempt data
            
        Returns:
            List of DetectionResult objects
        """
        try:
            results = []
            if 'login_success' not in data.columns or 'source_ip' not in data.columns:
                logger.warning("Required columns not found for login detection")
                return results

            # Group by source IP and count failed logins
            failed_logins = data[
                (data['login_success'] == False) &
                (data['timestamp'] >= datetime.now() - timedelta(seconds=self.thresholds['failed_login_window']))
            ].groupby('source_ip').size()
            
            # Detect IPs with too many failed attempts
            suspicious_ips = failed_logins[failed_logins >= self.thresholds['failed_login_threshold']]
            
            for ip, count in suspicious_ips.items():
                results.append(
                    DetectionResult(
                        timestamp=datetime.now(),
                        rule_name="repeated_failed_login",
                        severity="high",
                        description=f"Multiple failed login attempts detected: {count} attempts",
                        source_ip=ip,
                        additional_info={'failed_count': int(count)}
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in failed login detection: {str(e)}")
            raise

    def detect_unusual_ports(self, data: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect access to unusual ports.
        
        Args:
            data: DataFrame containing network connection data
            
        Returns:
            List of DetectionResult objects
        """
        try:
            results = []
            if 'destination_port' not in data.columns:
                logger.warning("destination_port column not found in data")
                return results

            # Calculate port frequency
            port_frequency = data['destination_port'].value_counts(normalize=True)
            unusual_ports = port_frequency[port_frequency < self.thresholds['unusual_port_threshold']]
            
            for port, freq in unusual_ports.items():
                suspicious_connections = data[data['destination_port'] == port]
                
                for _, conn in suspicious_connections.iterrows():
                    results.append(
                        DetectionResult(
                            timestamp=conn.get('timestamp', datetime.now()),
                            rule_name="unusual_port_access",
                            severity="medium",
                            description=f"Access to unusual port detected: {port}",
                            source_ip=conn.get('source_ip'),
                            destination_ip=conn.get('destination_ip'),
                            additional_info={'port': int(port), 'frequency': float(freq)}
                        )
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in unusual port detection: {str(e)}")
            raise

    def detect_data_exfiltration(self, data: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect potential data exfiltration attempts.
        
        Args:
            data: DataFrame containing network traffic data
            
        Returns:
            List of DetectionResult objects
        """
        try:
            results = []
            if 'bytes_transferred' not in data.columns or 'direction' not in data.columns:
                logger.warning("Required columns not found for data exfiltration detection")
                return results

            # Look for large outbound transfers
            suspicious_transfers = data[
                (data['direction'] == 'outbound') &
                (data['bytes_transferred'] > self.thresholds['data_transfer_threshold'])
            ]
            
            for _, transfer in suspicious_transfers.iterrows():
                results.append(
                    DetectionResult(
                        timestamp=transfer.get('timestamp', datetime.now()),
                        rule_name="data_exfiltration",
                        severity="critical",
                        description=f"Large outbound data transfer detected: {transfer['bytes_transferred']} bytes",
                        source_ip=transfer.get('source_ip'),
                        destination_ip=transfer.get('destination_ip'),
                        additional_info={'transfer_size': float(transfer['bytes_transferred'])}
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in data exfiltration detection: {str(e)}")
            raise

    def detect_suspicious_processes(self, data: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect suspicious process behavior.
        
        Args:
            data: DataFrame containing process monitoring data
            
        Returns:
            List of DetectionResult objects
        """
        try:
            results = []
            if 'process_name' not in data.columns or 'cpu_usage' not in data.columns:
                logger.warning("Required columns not found for process detection")
                return results

            # Detect high CPU usage processes
            suspicious_processes = data[data['cpu_usage'] > self.thresholds['process_cpu_threshold']]
            
            for _, process in suspicious_processes.iterrows():
                results.append(
                    DetectionResult(
                        timestamp=process.get('timestamp', datetime.now()),
                        rule_name="suspicious_process",
                        severity="high",
                        description=f"High CPU usage process detected: {process['process_name']}",
                        additional_info={
                            'process_name': process['process_name'],
                            'cpu_usage': float(process['cpu_usage'])
                        }
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in suspicious process detection: {str(e)}")
            raise

    def analyze_data(self, data: pd.DataFrame) -> List[DetectionResult]:
        """
        Analyze data using all enabled detection rules.
        
        Args:
            data: DataFrame containing the data to analyze
            
        Returns:
            List of DetectionResult objects from all rules
        """
        try:
            all_results = []
            
            # Apply each enabled rule
            for rule in self.rules:
                if not rule.enabled:
                    continue
                    
                if rule.name == "high_traffic_volume":
                    results = self.detect_high_traffic(data)
                elif rule.name == "repeated_failed_login":
                    results = self.detect_failed_logins(data)
                elif rule.name == "unusual_port_access":
                    results = self.detect_unusual_ports(data)
                elif rule.name == "data_exfiltration":
                    results = self.detect_data_exfiltration(data)
                elif rule.name == "suspicious_process":
                    results = self.detect_suspicious_processes(data)
                
                all_results.extend(results)
            
            # Log and store results
            for result in all_results:
                logger.info(
                    f"Detection: {result.rule_name} - {result.severity} - {result.description}"
                )
                
                # Store in database
                DetectionEvents.insert_event(
                    event_type=result.rule_name,
                    severity=result.severity,
                    description=result.description,
                    source_ip=result.source_ip,
                    destination_ip=result.destination_ip
                )
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise

    def update_rule_config(self, rule_updates: Dict[str, Any]) -> None:
        """
        Update rule configuration and thresholds.
        
        Args:
            rule_updates: Dictionary containing rule updates
        """
        try:
            # Update thresholds
            if 'thresholds' in rule_updates:
                self.thresholds.update(rule_updates['thresholds'])
            
            # Update rule enabled status
            if 'rules' in rule_updates:
                for rule in self.rules:
                    if rule.name in rule_updates['rules']:
                        rule.enabled = rule_updates['rules'][rule.name].get('enabled', rule.enabled)
            
            logger.info("Rule configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating rule configuration: {str(e)}")
            raise

# Export the classes
__all__ = ['RuleBasedDetector', 'DetectionRule', 'DetectionResult']
