-- Migration: 002_org_users_schema.sql
-- Description: Foundation schema for Organization, Users, Roles & Account Limits
-- Version: 1.0 (Locked)
-- PRD Reference: ORG_USERS_ROLES_PRD.md §15

-- =============================================================================
-- 1. ORGANIZATIONS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(20) NOT NULL CHECK (type IN ('AGENCY', 'SELLER')),
    subscription_plan VARCHAR(50), -- Nullable for now, defined by billing system later
    amazon_account_limit INT NOT NULL DEFAULT 5,
    seat_price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'SUSPENDED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 2. USERS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id), -- Specific constraint to new table only
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    -- Roles locked to Enum in code: OWNER, ADMIN, OPERATOR, VIEWER
    role VARCHAR(20) NOT NULL CHECK (role IN ('OWNER', 'ADMIN', 'OPERATOR', 'VIEWER')),
    billable BOOLEAN NOT NULL DEFAULT true,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DISABLED')),
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

--Index for faster lookups by email
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
--Index for fetching all users in an org
CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(organization_id);

-- =============================================================================
-- 3. AMAZON ACCOUNTS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS amazon_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    display_name VARCHAR(255) NOT NULL,
    marketplace VARCHAR(50) NOT NULL, -- e.g. "US", "UK", "UAE"
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DISABLED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fetching accounts by org
CREATE INDEX IF NOT EXISTS idx_amazon_accounts_org_id ON amazon_accounts(organization_id);
