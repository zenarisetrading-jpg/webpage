"""
Auth Models
===========
Data models for the Organization, User, and Role system.
PRD Reference: ORG_USERS_ROLES_PRD.md §5, §6

These models map 1:1 to the database schema in migrations/002_org_users_schema.sql.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict
from uuid import UUID

from .permissions import Role  # Canonical source for Role enum


@dataclass
class Organization:
    """
    Represents a paying entity (Agency or Individual Seller).
    Use 'id' to scope all queries.
    """
    id: UUID
    name: str
    type: str  # 'AGENCY' or 'SELLER'
    subscription_plan: Optional[str]
    amazon_account_limit: int
    seat_price: Decimal
    status: str  # 'ACTIVE' or 'SUSPENDED'
    created_at: datetime


@dataclass
class User:
    """
    Represents a login identity belonging to an Organization.
    
    PASSWORD RULES (LOCKED):
    - Must use bcrypt hashes (never cleartext)
    - Invite generation creates 72h one-time token
    - Passwords never logged
    """
    id: UUID
    organization_id: UUID
    email: str
    password_hash: str
    role: Role
    billable: bool
    status: str  # 'ACTIVE' or 'DISABLED'
    created_at: datetime
    last_login_at: Optional[datetime] = None
    # Phase 3 Security
    must_reset_password: bool = False
    password_updated_at: Optional[datetime] = None
    
    # Phase 3.5: Account Access Overrides (Downgrade Only)
    # Mapping: amazon_account_id -> Role (VIEWER or OPERATOR)
    account_overrides: Dict[UUID, Role] = field(default_factory=dict)


@dataclass
class PasswordChangeResult:
    """
    Typed result for password operations.
    Avoids boolean blindness.
    """
    success: bool
    reason: Optional[str] = None
    # For future extensibility (e.g. rate limit info)


@dataclass
class AmazonAccount:
    """
    Represents a connected Amazon Seller Central account + Marketplace.
    Owned by Organization, not User.
    """
    id: UUID
    organization_id: UUID
    display_name: str
    marketplace: str
    status: str  # 'ACTIVE' or 'DISABLED'
    created_at: datetime
