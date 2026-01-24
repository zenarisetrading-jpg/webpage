-- Migration: 003_add_password_security
-- Description: Adds columns for password lifecycle management (forced reset, audit)

-- 1. Add 'must_reset_password' flag
-- Default is FALSE (normal state).
-- Set to TRUE when Admin resets password to a temp one.
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS must_reset_password BOOLEAN NOT NULL DEFAULT FALSE;

-- 2. Add 'password_updated_at' audit column
-- Tracks when the password was last changed.
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS password_updated_at TIMESTAMPTZ;

-- 3. Backfill existing users (Optional but clean)
-- Assume current passwords are valid/fresh enough.
UPDATE users SET must_reset_password = FALSE WHERE must_reset_password IS NULL;
