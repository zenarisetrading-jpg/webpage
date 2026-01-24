"""
Account Management
==================
Logic for Amazon Account management and cap enforcement.
PRD Reference: ORG_USERS_ROLES_PRD.md §10

Rules:
- Count ACTIVE accounts only
- Disabled accounts do NOT count
- Reactivation checks cap
- Exceeding cap raises AccountLimitExceeded
"""

from typing import Optional, Protocol, Any
from uuid import UUID

class AccountLimitExceeded(Exception):
    """Raised when organization account limit would be exceeded."""
    pass


# SQL Queries (Defined here for clarity/reuse)
COUNT_ACTIVE_ACCOUNTS_SQL = """
    SELECT COUNT(*) 
    FROM amazon_accounts 
    WHERE organization_id = %s AND status = 'ACTIVE';
"""

GET_ORG_LIMIT_SQL = """
    SELECT amazon_account_limit 
    FROM organizations 
    WHERE id = %s;
"""


def check_account_cap_enforcement(
    current_active_count: int, 
    limit: int
) -> None:
    """
    Pure logic: Enforces the account limit.
    
    Args:
        current_active_count: Number of currently ACTIVE accounts
        limit: Max allowed accounts for the org
        
    Raises:
        AccountLimitExceeded: If adding one more would exceed limit
        
    Note: Call this BEFORE adding a new account or reactivating a disabled one.
    """
    # If we are about to add/reactivate 1 account, 
    # we fail if current count is ALREADY at (or above) limit.
    if current_active_count >= limit:
        raise AccountLimitExceeded(
            f"Organization limit reached ({limit} accounts). "
            "Please upgrade your plan to add more."
        )


from dataclasses import dataclass

@dataclass
class AccountValidationResult:
    allowed: bool
    reason: Optional[str] = None
    active_count: int = 0
    limit: int = 0

class DBExecutor(Protocol):
    """Protocol for database executor dependence."""
    # Updated to match PostgresManager interface or be adaptable
    def fetch_one(self, query: str, params: tuple) -> Any: ...

def validate_new_account_request(
    db: Any, # Typed as Any to accept PostgresManager or Protocol
    organization_id: str
) -> AccountValidationResult:
    """
    Full validation flow for adding/reactivating an account.
    Returns typed result instead of raising exception.
    """
    try:
        # 1. Get Limit
        # Using db.fetch_one assuming it returns a tuple or None
        # Adapter logic might be needed if db is raw cursor
        
        # Check if db has fetch_one (our PostgresManager does via self.conn if tailored, 
        # but standard PostgresManager uses pandas or raw cursors. 
        # Let's assume we pass a cursor-like wrapper or modify this to use standard SQL execution.)
        
        # ACTUALLY: Let's make this simple and robust. 
        # We will assume db is the PostgresManager instance which has a _get_connection usually...
        # But to be safe and clean, let's use the query directly if we can, or expect a query runner function.
        
        # Let's standardize on passing the PostgresManager and using internal method or raw connection.
        # For Phase 2, we'll use a local connection helper inside here if db is just connection info,
        # OR we accept the PostgresManager and use its public API if available.
        # Since PostgresManager is complex, let's use a fresh connection for this critical check.
        pass
    except Exception:
        pass
        
    # RE-IMPLEMENTATION with Robust Logic
    import psycopg2
    import os
    
    # Direct DB Access for Critical Guardrail
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        return AccountValidationResult(False, "System Configuration Error: DB URL missing")
        
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # 1. Get Limit
        cur.execute(GET_ORG_LIMIT_SQL, (str(organization_id),))
        row = cur.fetchone()
        if not row:
            conn.close()
            return AccountValidationResult(False, "Organization not found")
        limit = row[0]
        
        # 2. Get Count
        cur.execute(COUNT_ACTIVE_ACCOUNTS_SQL, (str(organization_id),))
        count = cur.fetchone()[0]
        
        conn.close()
        
        # 3. Check
        if count >= limit:
            return AccountValidationResult(
                False, 
                f"Limit reached ({limit}). Upgrade needed.", 
                count, 
                limit
            )
            
        return AccountValidationResult(True, None, count, limit)
        
    except Exception as e:
        print(f"Validation Error: {e}")
        return AccountValidationResult(False, "System Error during validation")
