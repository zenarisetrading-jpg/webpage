"""
Core Auth Package
=================
Single source of truth for authentication and authorization.

PRD Reference: ORG_USERS_ROLES_PRD.md

RULE: All permission checks must use core.auth.permissions
"""

from .permissions import (
    Role,
    ROLE_HIERARCHY,
    PERMISSION_MATRIX,
    has_permission,
    get_role_level,
    can_manage_role,
    get_billable_default,
    require_permission,
    require_role,
)

__all__ = [
    "Role",
    "ROLE_HIERARCHY",
    "PERMISSION_MATRIX",
    "has_permission",
    "get_role_level",
    "can_manage_role",
    "get_billable_default",
    "require_permission",
    "require_role",
]
