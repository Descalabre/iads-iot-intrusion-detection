"""
Ethical Compliance Module for IADS

This module handles the ethical approval and DPIA (Data Protection Impact Assessment)
compliance requirements for the IADS system. It provides functionality for creating,
submitting, and managing ethical approval forms and DPIA assessments.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Union

from src.core.db import get_db_connection

# Configure logging
logger = logging.getLogger(__name__)

class ComplianceError(Exception):
    """Custom exception for compliance-related errors."""
    pass

def create_compliance_tables() -> None:
    """
    Create the necessary database tables for ethical compliance if they don't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create ethics_compliance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ethics_compliance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                form_type VARCHAR(50) NOT NULL,
                form_data TEXT NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Create compliance_audit_log table for tracking changes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                compliance_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                action VARCHAR(50) NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (compliance_id) REFERENCES ethics_compliance(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        conn.commit()
        logger.info("Successfully created compliance tables")
        
    except Exception as e:
        logger.error(f"Error creating compliance tables: {e}")
        raise ComplianceError(f"Failed to create compliance tables: {str(e)}")
    finally:
        cursor.close()

def validate_compliance_form(form_data: Dict) -> None:
    """
    Validate the compliance form data against required fields and constraints.
    
    Args:
        form_data: Dictionary containing the form data
        
    Raises:
        ComplianceError: If validation fails
    """
    required_fields = [
        'data_nature',
        'dpia_needed',
        'data_flow',
        'data_sharing'
    ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if not form_data.get(field)]
    if missing_fields:
        raise ComplianceError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate data nature field
    valid_data_natures = ['anonymised', 'pseudoanonymised', 'non-anonymised']
    if form_data['data_nature'] not in valid_data_natures:
        raise ComplianceError(f"Invalid data nature. Must be one of: {', '.join(valid_data_natures)}")
    
    # Validate DPIA needed field
    if form_data['dpia_needed'] not in ['yes', 'no']:
        raise ComplianceError("DPIA needed field must be 'yes' or 'no'")

def submit_compliance_form(user_id: int, form_data: Dict) -> int:
    """
    Submit a new ethical compliance form.
    
    Args:
        user_id: ID of the user submitting the form
        form_data: Dictionary containing the form data
        
    Returns:
        int: ID of the newly created compliance record
        
    Raises:
        ComplianceError: If submission fails
    """
    try:
        # Validate form data
        validate_compliance_form(form_data)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert the compliance form
        cursor.execute('''
            INSERT INTO ethics_compliance 
            (user_id, form_type, form_data, status) 
            VALUES (?, ?, ?, ?)
        ''', (user_id, 'ethical_approval', json.dumps(form_data), 'pending'))
        
        compliance_id = cursor.lastrowid
        
        # Log the submission in audit log
        cursor.execute('''
            INSERT INTO compliance_audit_log 
            (compliance_id, user_id, action, details) 
            VALUES (?, ?, ?, ?)
        ''', (compliance_id, user_id, 'submit', 'Initial submission'))
        
        conn.commit()
        logger.info(f"Successfully submitted compliance form for user {user_id}")
        
        return compliance_id
        
    except Exception as e:
        logger.error(f"Error submitting compliance form: {e}")
        raise ComplianceError(f"Failed to submit compliance form: {str(e)}")
    finally:
        cursor.close()

def get_compliance_form(form_id: int) -> Optional[Dict]:
    """
    Retrieve a specific compliance form by ID.
    
    Args:
        form_id: ID of the compliance form to retrieve
        
    Returns:
        Optional[Dict]: The compliance form data or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, user_id, submission_date, form_type, form_data, status, last_modified 
            FROM ethics_compliance 
            WHERE id = ?
        ''', (form_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'user_id': result[1],
                'submission_date': result[2],
                'form_type': result[3],
                'form_data': json.loads(result[4]),
                'status': result[5],
                'last_modified': result[6]
            }
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving compliance form {form_id}: {e}")
        raise ComplianceError(f"Failed to retrieve compliance form: {str(e)}")
    finally:
        cursor.close()

def get_user_compliance_forms(user_id: int) -> List[Dict]:
    """
    Retrieve all compliance forms submitted by a specific user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List[Dict]: List of compliance forms
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, submission_date, form_type, form_data, status, last_modified 
            FROM ethics_compliance 
            WHERE user_id = ? 
            ORDER BY submission_date DESC
        ''', (user_id,))
        
        forms = []
        for row in cursor.fetchall():
            forms.append({
                'id': row[0],
                'submission_date': row[1],
                'form_type': row[2],
                'form_data': json.loads(row[3]),
                'status': row[4],
                'last_modified': row[5]
            })
        return forms
        
    except Exception as e:
        logger.error(f"Error retrieving compliance forms for user {user_id}: {e}")
        raise ComplianceError(f"Failed to retrieve user compliance forms: {str(e)}")
    finally:
        cursor.close()

def update_compliance_status(form_id: int, user_id: int, new_status: str, 
                           comment: Optional[str] = None) -> None:
    """
    Update the status of a compliance form and log the change.
    
    Args:
        form_id: ID of the compliance form
        user_id: ID of the user making the change
        new_status: New status to set ('approved', 'rejected', 'pending')
        comment: Optional comment explaining the status change
    """
    valid_statuses = ['approved', 'rejected', 'pending']
    if new_status not in valid_statuses:
        raise ComplianceError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Update the status
        cursor.execute('''
            UPDATE ethics_compliance 
            SET status = ?, last_modified = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_status, form_id))
        
        # Log the status change
        cursor.execute('''
            INSERT INTO compliance_audit_log 
            (compliance_id, user_id, action, details) 
            VALUES (?, ?, ?, ?)
        ''', (form_id, user_id, 'status_change', 
              f"Status changed to {new_status}" + (f": {comment}" if comment else "")))
        
        conn.commit()
        logger.info(f"Successfully updated compliance form {form_id} status to {new_status}")
        
    except Exception as e:
        logger.error(f"Error updating compliance form status: {e}")
        raise ComplianceError(f"Failed to update compliance status: {str(e)}")
    finally:
        cursor.close()

def get_compliance_audit_log(form_id: int) -> List[Dict]:
    """
    Retrieve the audit log for a specific compliance form.
    
    Args:
        form_id: ID of the compliance form
        
    Returns:
        List[Dict]: List of audit log entries
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT user_id, action, details, timestamp 
            FROM compliance_audit_log 
            WHERE compliance_id = ? 
            ORDER BY timestamp DESC
        ''', (form_id,))
        
        logs = []
        for row in cursor.fetchall():
            logs.append({
                'user_id': row[0],
                'action': row[1],
                'details': row[2],
                'timestamp': row[3]
            })
        return logs
        
    except Exception as e:
        logger.error(f"Error retrieving audit log for compliance form {form_id}: {e}")
        raise ComplianceError(f"Failed to retrieve compliance audit log: {str(e)}")
    finally:
        cursor.close()
