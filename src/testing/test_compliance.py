"""
Unit tests for the ethical compliance module.
"""

import unittest
import json
from datetime import datetime
from src.compliance.ethical_compliance import (
    create_compliance_tables,
    validate_compliance_form,
    submit_compliance_form,
    get_compliance_form,
    get_user_compliance_forms,
    update_compliance_status,
    get_compliance_audit_log,
    ComplianceError
)

class TestEthicalCompliance(unittest.TestCase):
    """Test cases for ethical compliance functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create necessary tables
        create_compliance_tables()
        
        # Sample valid form data
        self.valid_form_data = {
            'data_nature': 'anonymised',
            'dpia_needed': 'yes',
            'data_flow': 'Data will be collected through secure channels',
            'data_sharing': 'Data will be shared only with authorized personnel',
            'additional_comments': 'Test submission'
        }
        
        # Sample user ID for testing
        self.test_user_id = 1

    def test_validate_compliance_form_valid(self):
        """Test form validation with valid data."""
        try:
            validate_compliance_form(self.valid_form_data)
        except ComplianceError:
            self.fail("validate_compliance_form() raised ComplianceError unexpectedly!")

    def test_validate_compliance_form_missing_fields(self):
        """Test form validation with missing required fields."""
        invalid_form = self.valid_form_data.copy()
        del invalid_form['data_nature']
        
        with self.assertRaises(ComplianceError):
            validate_compliance_form(invalid_form)

    def test_validate_compliance_form_invalid_data_nature(self):
        """Test form validation with invalid data nature value."""
        invalid_form = self.valid_form_data.copy()
        invalid_form['data_nature'] = 'invalid_value'
        
        with self.assertRaises(ComplianceError):
            validate_compliance_form(invalid_form)

    def test_submit_compliance_form(self):
        """Test successful form submission."""
        try:
            form_id = submit_compliance_form(self.test_user_id, self.valid_form_data)
            self.assertIsInstance(form_id, int)
            self.assertGreater(form_id, 0)
        except ComplianceError as e:
            self.fail(f"submit_compliance_form() raised ComplianceError: {str(e)}")

    def test_get_compliance_form(self):
        """Test retrieval of submitted form."""
        # First submit a form
        form_id = submit_compliance_form(self.test_user_id, self.valid_form_data)
        
        # Then retrieve it
        form = get_compliance_form(form_id)
        
        self.assertIsNotNone(form)
        self.assertEqual(form['user_id'], self.test_user_id)
        self.assertEqual(form['status'], 'pending')
        self.assertEqual(json.loads(form['form_data'])['data_nature'], 
                        self.valid_form_data['data_nature'])

    def test_get_user_compliance_forms(self):
        """Test retrieval of all forms for a user."""
        # Submit multiple forms
        submit_compliance_form(self.test_user_id, self.valid_form_data)
        submit_compliance_form(self.test_user_id, self.valid_form_data)
        
        # Retrieve all forms
        forms = get_user_compliance_forms(self.test_user_id)
        
        self.assertIsInstance(forms, list)
        self.assertGreaterEqual(len(forms), 2)
        for form in forms:
            self.assertIn('id', form)
            self.assertIn('submission_date', form)
            self.assertIn('status', form)

    def test_update_compliance_status(self):
        """Test updating the status of a compliance form."""
        # Submit a form
        form_id = submit_compliance_form(self.test_user_id, self.valid_form_data)
        
        # Update its status
        try:
            update_compliance_status(
                form_id, 
                self.test_user_id, 
                'approved', 
                'Approved after review'
            )
        except ComplianceError as e:
            self.fail(f"update_compliance_status() raised ComplianceError: {str(e)}")
        
        # Verify the update
        form = get_compliance_form(form_id)
        self.assertEqual(form['status'], 'approved')

    def test_update_compliance_status_invalid(self):
        """Test updating status with invalid value."""
        form_id = submit_compliance_form(self.test_user_id, self.valid_form_data)
        
        with self.assertRaises(ComplianceError):
            update_compliance_status(
                form_id, 
                self.test_user_id, 
                'invalid_status'
            )

    def test_get_compliance_audit_log(self):
        """Test retrieval of audit log entries."""
        # Submit a form and update its status
        form_id = submit_compliance_form(self.test_user_id, self.valid_form_data)
        update_compliance_status(
            form_id, 
            self.test_user_id, 
            'approved', 
            'Approved after review'
        )
        
        # Get audit log
        logs = get_compliance_audit_log(form_id)
        
        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 2)  # Should have submit and status change entries
        
        # Verify log entries
        actions = [log['action'] for log in logs]
        self.assertIn('submit', actions)
        self.assertIn('status_change', actions)

    def test_get_nonexistent_form(self):
        """Test retrieval of non-existent form."""
        form = get_compliance_form(999999)  # Using an ID that shouldn't exist
        self.assertIsNone(form)

if __name__ == '__main__':
    unittest.main()
