-- Beta Signups Table for Landing Page Form
-- This table stores beta access requests from the landing page

CREATE TABLE IF NOT EXISTS beta_signups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(50), -- seller, agency, inhouse, other
    accounts VARCHAR(20), -- 1, 2-5, 6-20, 20+
    monthly_spend VARCHAR(50), -- <10k, 10k-50k, 50k-200k, 200k+
    goal TEXT, -- Optional free text
    source VARCHAR(50) DEFAULT 'landing_page', -- Where they signed up from
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending' -- pending, contacted, approved, declined
);

-- Index for email lookups
CREATE INDEX IF NOT EXISTS idx_beta_signups_email ON beta_signups(email);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_beta_signups_status ON beta_signups(status);

-- Row Level Security: Allow anonymous inserts from landing page
ALTER TABLE beta_signups ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can insert (for public signup form)
CREATE POLICY "Allow anonymous insert" ON beta_signups
    FOR INSERT
    WITH CHECK (true);

-- Policy: Only authenticated users can read (for admin dashboard)
CREATE POLICY "Authenticated users can read" ON beta_signups
    FOR SELECT
    USING (auth.role() = 'authenticated');

COMMENT ON TABLE beta_signups IS 'Stores beta access requests from the landing page signup form';
