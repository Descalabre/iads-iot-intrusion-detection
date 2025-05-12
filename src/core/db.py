"""
Database module for the Intrusion and Anomaly Detection System (IADS).
Handles all database operations including connection management and CRUD operations.
Uses connection pooling for better performance and resource management.
"""

import mysql.connector
from mysql.connector import Error, pooling
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from src.core.config import config, logger

# Database configuration with defaults
DB_CONFIG = {
    "host": config.get("database", {}).get("host", "localhost"),
    "database": config.get("database", {}).get("database", "iads_db"),
    "user": config.get("database", {}).get("user", "iads_user"),
    "password": config.get("database", {}).get("password", ""),
    "port": config.get("database", {}).get("port", 3306),
    "pool_name": "iads_pool",
    "pool_size": 5
}

class DatabaseManager:
    """Manages database connections and operations."""
    
    _pool: Optional[pooling.MySQLConnectionPool] = None
    
    @classmethod
    def initialize_pool(cls) -> None:
        """Initialize the connection pool if it doesn't exist."""
        if cls._pool is None:
            try:
                cls._pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name=DB_CONFIG["pool_name"],
                    pool_size=DB_CONFIG["pool_size"],
                    **{k: v for k, v in DB_CONFIG.items() if k not in ["pool_name", "pool_size"]}
                )
                logger.info("Database connection pool initialized successfully")
            except Error as e:
                logger.error(f"Error initializing connection pool: {e}")
                raise

    @classmethod
    @contextmanager
    def get_connection(cls):
        """
        Context manager for database connections.
        
        Yields:
            mysql.connector.MySQLConnection: Database connection from the pool
        
        Raises:
            Error: If connection cannot be established
        """
        if cls._pool is None:
            cls.initialize_pool()
        
        connection = None
        try:
            connection = cls._pool.get_connection()
            yield connection
        except Error as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Error as e:
                    logger.warning(f"Error closing connection: {e}")

    @staticmethod
    def create_tables() -> None:
        """Create necessary database tables if they don't exist."""
        create_table_queries = [
            """
            CREATE TABLE IF NOT EXISTS system_status (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                detections INT NOT NULL,
                cpu_usage FLOAT NOT NULL,
                uptime VARCHAR(50) NOT NULL,
                memory_usage FLOAT NOT NULL,
                INDEX idx_timestamp (timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS detection_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                description TEXT,
                source_ip VARCHAR(45),
                destination_ip VARCHAR(45),
                INDEX idx_timestamp (timestamp),
                INDEX idx_severity (severity)
            );
            """
        ]
        
        with DatabaseManager.get_connection() as connection:
            try:
                cursor = connection.cursor()
                for query in create_table_queries:
                    cursor.execute(query)
                connection.commit()
                logger.info("Database tables created successfully")
            except Error as e:
                logger.error(f"Error creating tables: {e}")
                raise
            finally:
                cursor.close()

class SystemStatus:
    """Handles system status related database operations."""
    
    @staticmethod
    def insert_status(detections: int, cpu_usage: float, uptime: str, memory_usage: float) -> None:
        """
        Insert a new system status record.
        
        Args:
            detections: Number of detections
            cpu_usage: CPU usage percentage
            uptime: System uptime
            memory_usage: Memory usage percentage
        """
        query = """
            INSERT INTO system_status (detections, cpu_usage, uptime, memory_usage)
            VALUES (%s, %s, %s, %s);
        """
        with DatabaseManager.get_connection() as connection:
            try:
                cursor = connection.cursor()
                cursor.execute(query, (detections, cpu_usage, uptime, memory_usage))
                connection.commit()
            except Error as e:
                logger.error(f"Error inserting status: {e}")
                raise
            finally:
                cursor.close()

    @staticmethod
    def get_latest_status() -> Optional[Dict[str, Any]]:
        """
        Get the most recent system status.
        
        Returns:
            Dict containing the latest status or None if no records exist
        """
        query = "SELECT * FROM system_status ORDER BY timestamp DESC LIMIT 1;"
        with DatabaseManager.get_connection() as connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(query)
                return cursor.fetchone()
            except Error as e:
                logger.error(f"Error querying latest status: {e}")
                raise
            finally:
                cursor.close()

    @staticmethod
    def get_status_history(hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get system status history for the specified time period.
        
        Args:
            hours: Number of hours of history to retrieve (default: 24)
        
        Returns:
            List of status records
        """
        query = """
            SELECT timestamp, detections, cpu_usage, memory_usage
            FROM system_status
            WHERE timestamp >= NOW() - INTERVAL %s HOUR
            ORDER BY timestamp;
        """
        with DatabaseManager.get_connection() as connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(query, (hours,))
                return cursor.fetchall()
            except Error as e:
                logger.error(f"Error querying status history: {e}")
                raise
            finally:
                cursor.close()

class DetectionEvents:
    """Handles detection events related database operations."""
    
    @staticmethod
    def insert_event(
        event_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None
    ) -> None:
        """
        Insert a new detection event.
        
        Args:
            event_type: Type of detection event
            severity: Event severity level
            description: Detailed description of the event
            source_ip: Source IP address (optional)
            destination_ip: Destination IP address (optional)
        """
        query = """
            INSERT INTO detection_events 
            (event_type, severity, description, source_ip, destination_ip)
            VALUES (%s, %s, %s, %s, %s);
        """
        with DatabaseManager.get_connection() as connection:
            try:
                cursor = connection.cursor()
                cursor.execute(query, (
                    event_type,
                    severity,
                    description,
                    source_ip,
                    destination_ip
                ))
                connection.commit()
            except Error as e:
                logger.error(f"Error inserting detection event: {e}")
                raise
            finally:
                cursor.close()

    @staticmethod
    def get_recent_events(limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent detection events.
        
        Args:
            limit: Maximum number of events to retrieve
        
        Returns:
            List of recent detection events
        """
        query = """
            SELECT *
            FROM detection_events
            ORDER BY timestamp DESC
            LIMIT %s;
        """
        with DatabaseManager.get_connection() as connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(query, (limit,))
                return cursor.fetchall()
            except Error as e:
                logger.error(f"Error querying recent events: {e}")
                raise
            finally:
                cursor.close()

# Initialize database tables on module import
try:
    DatabaseManager.create_tables()
except Error as e:
    logger.error(f"Failed to initialize database tables: {e}")
    logger.warning("Application may not function correctly without proper database setup")

# Export classes for external use
__all__ = ['DatabaseManager', 'SystemStatus', 'DetectionEvents']
