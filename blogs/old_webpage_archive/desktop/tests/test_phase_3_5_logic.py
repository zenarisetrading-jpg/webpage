
import unittest
from uuid import uuid4
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest.mock import MagicMock
import sys

# MOCK STREAMLIT BEFORE IMPORTING SERVICE
mock_st = MagicMock()
mock_st.session_state = {}
sys.modules["streamlit"] = mock_st

from core.auth.models import User
from core.auth.permissions import Role, get_effective_role, has_permission_for_account
from core.auth.service import AuthService

class TestPhase3_5Logic(unittest.TestCase):
    
    def test_effective_role_downgrade(self):
        """Test 'Most Restrictive Wins' logic."""
        # Global: ADMIN, Override: VIEWER -> VIEWER
        self.assertEqual(get_effective_role("ADMIN", "VIEWER"), "VIEWER")
        
        # Global: OWNER, Override: OPERATOR -> OPERATOR
        self.assertEqual(get_effective_role("OWNER", "OPERATOR"), "OPERATOR")
        
        # Global: VIEWER, Override: OPERATOR -> VIEWER (Cannot upgrade)
        # Note: logic returns MIN. Viewer (1) < Operator (2).
        self.assertEqual(get_effective_role("VIEWER", "OPERATOR"), "VIEWER")
        
        # No Override
        self.assertEqual(get_effective_role("ADMIN", None), "ADMIN")
        
    def test_permission_context(self):
        """Test has_permission_for_account."""
        acc_id = uuid4()
        
        # User: Global ADMIN, Override VIEWER for acc_id
        user = User(
            id=uuid4(), organization_id=uuid4(), email="test", password_hash="x",
            role=Role.ADMIN, billable=True, status="ACTIVE", created_at=None,
            account_overrides={acc_id: Role.VIEWER}
        )
        
        # Should NOT be able to run_optimizer on acc_id
        self.assertFalse(has_permission_for_account(user, "run_optimizer", acc_id))
        
        # SHOULD be able to manage_users (Global check only? No, permission checks check effective role)
        # If we pass acc_id to manage_users check, it will use effective role (VIEWER).
        # Viewer cannot manage users.
        self.assertFalse(has_permission_for_account(user, "manage_users", acc_id))
        
        # Should be able to view_dashboards
        self.assertTrue(has_permission_for_account(user, "view_dashboards", acc_id))
        
        # Different Account (No override)
        other_acc_id = uuid4()
        self.assertTrue(has_permission_for_account(user, "run_optimizer", other_acc_id))


if __name__ == "__main__":
    unittest.main()
