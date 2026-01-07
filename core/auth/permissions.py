"""
Permissions Module (CANONICAL SOURCE)
=====================================
Single source of truth for all role-based permissions.

PRD Reference: ORG_USERS_ROLES_PRD.md §7

RULE: No permission checks outside this file.

Locked Terminology:
- Organization (not Company, Tenant, Account)
- User (not Member, Person)
- Role (OWNER, ADMIN, OPERATOR, VIEWER)
- AmazonAccount (not Store, Client, Profile)
- Override (Per-account role restriction)
- billable (boolean on User)
- amazon_account_limit (integer on Organization)
"""

from enum import Enum
from typing import List, Optional
from uuid import UUID


# =============================================================================
# ROLE DEFINITIONS (LOCKED)
# =============================================================================

class Role(str, Enum):
    """
    User roles in hierarchical order.
    Higher roles inherit all lower role permissions.
    """
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"


# =============================================================================
# ROLE HIERARCHY (LOCKED)
# =============================================================================

ROLE_HIERARCHY = {
    Role.OWNER: 4,
    Role.ADMIN: 3,
    Role.OPERATOR: 2,
    Role.VIEWER: 1,
}

# String version for DB compatibility
ROLE_HIERARCHY_STR = {
    "OWNER": 4,
    "ADMIN": 3,
    "OPERATOR": 2,
    "VIEWER": 1,
}


# =============================================================================
# PERMISSION MATRIX (LOCKED)
# =============================================================================
# Each permission lists the MINIMUM role required.
# Higher roles automatically inherit.
#
# RULE 1: Operator = "Execution, Not Configuration". 
#         Can run workflows but cannot change system state/config.
#
# RULE 2: Simulator = "Analytical Only".
#         Safe for Viewers as it has no side effects.

PERMISSION_MATRIX = {
    # Owner-only
    "manage_billing": [Role.OWNER],
    "delete_organization": [Role.OWNER],
    "transfer_ownership": [Role.OWNER],
    
    # Admin+ (Owner inherits)
    "manage_users": [Role.ADMIN],
    "manage_accounts": [Role.ADMIN],
    "configure_ingestion": [Role.ADMIN],
    "invite_users": [Role.ADMIN],
    
    # Operator+ (Admin, Owner inherit)
    "run_optimizer": [Role.OPERATOR],
    "upload_files": [Role.OPERATOR],
    "trigger_ingestion": [Role.OPERATOR],
    "download_reports": [Role.OPERATOR],
    
    # Viewer+ (All roles)
    "view_dashboards": [Role.VIEWER],
    "view_accounts": [Role.VIEWER],
    "view_reports": [Role.VIEWER],
}


# =============================================================================
# PERMISSION CHECKING (CANONICAL FUNCTION)
# =============================================================================

def has_permission(user_role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.
    
    Uses CUMULATIVE hierarchy - higher roles inherit all lower permissions.
    
    Args:
        user_role: User's role string (OWNER, ADMIN, OPERATOR, VIEWER)
        permission: Permission key from PERMISSION_MATRIX
        
    Returns:
        True if user has permission, False otherwise
        
    Example:
        has_permission("OWNER", "run_optimizer")  # True (4 >= 2)
        has_permission("VIEWER", "run_optimizer")  # False (1 < 2)
    """
    if permission not in PERMISSION_MATRIX:
        return False
    
    allowed_roles = PERMISSION_MATRIX[permission]
    user_level = ROLE_HIERARCHY_STR.get(user_role, 0)
    
    return any(
        user_level >= ROLE_HIERARCHY_STR.get(role.value, 0)
        for role in allowed_roles
    )


def get_role_level(role: str) -> int:
    """Get numeric level for a role."""
    return ROLE_HIERARCHY_STR.get(role, 0)


def can_manage_role(manager_role: str, target_role: str) -> bool:
    """
    Check if a manager can modify a target user's role.
    
    Rule: Can only manage roles below your own level.
    """
    manager_level = get_role_level(manager_role)
    target_level = get_role_level(target_role)
    return manager_level > target_level


# =============================================================================
# PHASE 3.5: ACCOUNT CONTEXT AWARENESS (LOCKED logic)
# =============================================================================

def get_effective_role(user_global_role: str, override_role: Optional[str] = None) -> str:
    """
    Determine effective role for a specific account context.
    
    Rule (Phase 3.5): downgrade-only.
    Effective Role = MIN(Global Role, Override Role)
    """
    if not override_role:
        return user_global_role
        
    global_level = get_role_level(user_global_role)
    override_level = get_role_level(override_role)
    
    # Return whichever is lower (more restrictive)
    if override_level < global_level:
        return override_role
    return user_global_role


def has_permission_for_account(user, permission: str, account_id: Optional[UUID] = None) -> bool:
    """
    Check if user has permission for a specific account context.
    
    Args:
        user: User model instance (must have global 'role' and 'account_overrides')
        permission: Permission key
        account_id: Context account (optional)
        
    Returns:
        bool
    """
    # 1. Get Override if exists
    override_role_str = None
    if account_id and hasattr(user, 'account_overrides') and account_id in user.account_overrides:
        role_obj = user.account_overrides[account_id]
        if hasattr(role_obj, 'value'): # Handle Enum
             override_role_str = role_obj.value
        else:
             override_role_str = str(role_obj)
             
    # 2. Calculate Effective Role
    effective_role = get_effective_role(user.role.value, override_role_str)
    
    # 3. Check Base Permission
    return has_permission(effective_role, permission)


# =============================================================================
# BILLABLE DEFAULTS BY ROLE (LOCKED)
# =============================================================================

BILLABLE_DEFAULTS = {
    Role.OWNER: True,
    Role.ADMIN: True,
    Role.OPERATOR: True,
    Role.VIEWER: False,  # Read-only = free by default
}

def get_billable_default(role: str) -> bool:
    """Get default billable status for a role."""
    try:
        return BILLABLE_DEFAULTS.get(Role(role), True)
    except ValueError:
        return True


# =============================================================================
# CONVENIENCE DECORATORS (FOR MIDDLEWARE)
# =============================================================================

def require_permission(permission: str):
    """
    Decorator to enforce permission check.
    
    Usage:
        @require_permission("manage_users")
        def add_user(request):
            ...
    """
    def decorator(func):
        def wrapper(request, *args, **kwargs):
            user_role = getattr(request, 'user_role', None)
            if not user_role or not has_permission(user_role, permission):
                raise PermissionError(f"Permission denied: {permission}")
            return func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_role(*roles: str):
    """
    Decorator to require specific role(s).
    
    Usage:
        @require_role("OWNER", "ADMIN")
        def admin_action(request):
            ...
    """
    def decorator(func):
        def wrapper(request, *args, **kwargs):
            user_role = getattr(request, 'user_role', None)
            if user_role not in roles:
                raise PermissionError(f"Role required: {roles}")
            return func(request, *args, **kwargs)
        return wrapper
    return decorator
