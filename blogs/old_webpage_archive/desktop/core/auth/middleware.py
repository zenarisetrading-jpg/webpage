"""
Auth Middleware
===============
decorators for enforcing authentication and permissions.
PRD Reference: ORG_USERS_ROLES_PRD.md §14

Relies ONLY on core.auth.permissions.has_permission().
"""

from functools import wraps
from typing import Any, Callable, Optional, Union

from .permissions import has_permission


class AuthError(Exception):
    """Base class for authentication errors."""
    pass


class PermissionDenied(AuthError):
    """Raised when user lacks required permission."""
    pass


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require a logged-in user.
    
    Strategies to find user:
    1. `request.user` attribute (Flask/Django style)
    2. `user` keyword argument
    3. `st.session_state.user` (Streamlit style - if applicable)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Check for user in kwargs
        user = kwargs.get('user')
        
        # 2. Check for request-like object in args[0]
        if not user and args:
            request = args[0]
            user = getattr(request, 'user', None)
            
        # 3. Validation
        if not user:
            # We don't check Streamlit session state here to keep this pure logic
            # The caller should inject the user context
            raise AuthError("Authentication required: No user found in context.")
            
        return func(*args, **kwargs)
    return wrapper


def require_permission(permission: str) -> Callable:
    """
    Decorator to enforce permission check.
    
    Args:
        permission: Permission key from PERMISSION_MATRIX
        
    Raises:
        PermissionDenied: If user role lacks permission
        AuthError: If no user/role found
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Resolve User/Role
            user = kwargs.get('user')
            if not user and args:
                request = args[0]
                user = getattr(request, 'user', None)
                
            if not user:
                raise AuthError("Authentication required for permission check.")
                
            # 2. Extract Role
            # Handle both object with .role attribute or dict
            role = getattr(user, 'role', None) 
            if not role and isinstance(user, dict):
                 role = user.get('role')
                 
            if not role:
                raise AuthError("User has no role assigned.")
                
            # 3. Check Permission (Canonical)
            # Ensure role is string for the check
            role_str = str(role.value) if hasattr(role, 'value') else str(role)
            
            if not has_permission(role_str, permission):
                raise PermissionDenied(f"Permission '{permission}' denied for role '{role_str}'")
                
            return func(*args, **kwargs)
        return wrapper
    return decorator # Note: logic bug here in original thought (nested wrapper), fixing below.
