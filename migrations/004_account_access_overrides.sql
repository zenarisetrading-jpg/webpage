-- Migration 004: Account Access Overrides (Phase 3.5)
-- Explicit downgrade-only support for agency use case

-- Safety check: drop if exists to ensure clean slate after failed run
DROP TABLE IF EXISTS user_account_overrides;

CREATE TABLE user_account_overrides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amazon_account_id UUID NOT NULL REFERENCES amazon_accounts(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('VIEWER', 'OPERATOR')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    
    -- One override per user per account
    UNIQUE(user_id, amazon_account_id)
);

-- Optimize lookups
CREATE INDEX idx_user_account_overrides_user ON user_account_overrides(user_id);
CREATE INDEX idx_user_account_overrides_account ON user_account_overrides(amazon_account_id);

COMMENT ON TABLE user_account_overrides IS 'Per-account access restrictions (downgrades only). Override role must be <= user global role.';

-- DB-LEVEL ENFORCEMENT via TRIGGER (since CHECK constraints can't use subqueries)

CREATE OR REPLACE FUNCTION check_account_override_downgrade()
RETURNS TRIGGER AS $$
DECLARE
    user_global_role VARCHAR;
    role_hierarchy INTEGER;
    global_hierarchy INTEGER;
BEGIN
    -- Get user's global role
    SELECT role INTO user_global_role FROM users WHERE id = NEW.user_id;

    -- Map roles to hierarchy level (Higher number = Higher permission)
    -- VIEWER = 1
    -- OPERATOR = 2
    -- ADMIN = 3
    -- OWNER = 4
    
    -- Get hierarchy for NEW override role
    IF NEW.role = 'VIEWER' THEN role_hierarchy := 1;
    ELSIF NEW.role = 'OPERATOR' THEN role_hierarchy := 2;
    ELSE RAISE EXCEPTION 'Invalid override role: %', NEW.role;
    END IF;

    -- Get hierarchy for global role
    IF user_global_role = 'VIEWER' THEN global_hierarchy := 1;
    ELSIF user_global_role = 'OPERATOR' THEN global_hierarchy := 2;
    ELSIF user_global_role = 'ADMIN' THEN global_hierarchy := 3;
    ELSIF user_global_role = 'OWNER' THEN global_hierarchy := 4;
    ELSE 
        -- Fallback for safety, though global role should be valid
        global_hierarchy := 0; 
    END IF;

    -- ENFORCEMENT: Override cannot be higher than global role
    IF role_hierarchy > global_hierarchy THEN
        RAISE EXCEPTION 'Constraint Violation: Cannot set override role % (level %) higher than global role % (level %)', 
            NEW.role, role_hierarchy, user_global_role, global_hierarchy;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_override_downgrade
BEFORE INSERT OR UPDATE ON user_account_overrides
FOR EACH ROW
EXECUTE FUNCTION check_account_override_downgrade();
