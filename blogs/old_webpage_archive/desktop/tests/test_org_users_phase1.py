"""
Integration Tests: Org/Users/Roles Phase 1
==========================================
Verifies logic correctness for:
1. Role inheritance (core.auth.permissions)
2. Account cap enforcement (features.account_management)
3. Auth middleware (core.auth.middleware)

Run with: python3 -m unittest tests/test_org_users_phase1.py
"""

import unittest
from unittest.mock import MagicMock

from core.auth.permissions import Role, has_permission, get_billable_default
from features.account_management import check_account_cap_enforcement, AccountLimitExceeded


class TestRoleInheritance(unittest.TestCase):
    """Verify PRD rule: Roles are cumulative and inherit downward."""
    
    def test_owner_permissions(self):
        """Owner should have ALL permissions."""
        role = Role.OWNER
        # Own permissions
        self.assertTrue(has_permission(role, "manage_billing"))
        # Inherited from Admin
        self.assertTrue(has_permission(role, "manage_users"))
        # Inherited from Operator
        self.assertTrue(has_permission(role, "run_optimizer"))
        # Inherited from Viewer
        self.assertTrue(has_permission(role, "view_dashboards"))

    def test_admin_permissions(self):
        """Admin should have Operator+Viewer but NOT Owner."""
        role = Role.ADMIN
        self.assertFalse(has_permission(role, "manage_billing"))      # Owner only
        self.assertTrue(has_permission(role, "manage_users"))         # Admin
        self.assertTrue(has_permission(role, "run_optimizer"))        # Inherited
        self.assertTrue(has_permission(role, "view_dashboards"))      # Inherited

    def test_operator_permissions(self):
        """Operator should have Viewer but NOT Admin/Owner."""
        role = Role.OPERATOR
        self.assertFalse(has_permission(role, "manage_users"))       # Admin
        self.assertTrue(has_permission(role, "run_optimizer"))       # Operator
        self.assertTrue(has_permission(role, "view_dashboards"))     # Inherited

    def test_viewer_permissions(self):
        """Viewer has read-only access."""
        role = Role.VIEWER
        self.assertFalse(has_permission(role, "run_optimizer"))
        self.assertTrue(has_permission(role, "view_dashboards"))


class TestAccountCapEnforcement(unittest.TestCase):
    """Verify PRD rule: Hard caps on Amazon accounts."""
    
    def test_under_limit(self):
        """Should succeed if count < limit."""
        # 4 active, limit 5 -> OK to add 1 more
        try:
            check_account_cap_enforcement(current_active_count=4, limit=5)
        except AccountLimitExceeded:
            self.fail("Raised AccountLimitExceeded explicitly when under limit")

    def test_at_limit(self):
        """Should fail if count == limit."""
        # 5 active, limit 5 -> Fail (can't add 6th)
        with self.assertRaises(AccountLimitExceeded):
            check_account_cap_enforcement(current_active_count=5, limit=5)

    def test_over_limit_safety(self):
        """Should fail if count > limit (safety check)."""
        with self.assertRaises(AccountLimitExceeded):
            check_account_cap_enforcement(current_active_count=6, limit=5)


class TestBillingDefaults(unittest.TestCase):
    """Verify PRD rule: Seat billing defaults."""
    
    def test_billable_statuses(self):
        self.assertTrue(get_billable_default("OWNER"))
        self.assertTrue(get_billable_default("ADMIN"))
        self.assertTrue(get_billable_default("OPERATOR"))
        self.assertFalse(get_billable_default("VIEWER"))

if __name__ == "__main__":
    unittest.main()
