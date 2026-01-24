-- ============================================================================
-- INGESTION V2 SCHEMA MIGRATION
-- ============================================================================
-- PRD Reference: EMAIL_INGESTION_PRD.md Section 12, 13
-- Purpose: Create V2 tables for new ingestion infrastructure
-- Safety: This is ADDITIVE ONLY - no changes to existing tables
-- ============================================================================

-- Ingestion Version ENUM (for namespace isolation)
DO $$ BEGIN
    CREATE TYPE ingestion_version AS ENUM ('v1', 'v2');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Ingestion Source ENUM
DO $$ BEGIN
    CREATE TYPE ingestion_source AS ENUM ('EMAIL', 'API', 'MANUAL');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Ingestion Status ENUM
DO $$ BEGIN
    CREATE TYPE ingestion_status AS ENUM (
        'RECEIVED',
        'PROCESSING', 
        'COMPLETED',
        'FAILED',
        'QUARANTINE',
        'DUPLICATE_IGNORED'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- TABLE: ingestion_events_v2
-- Purpose: Audit trail for every ingestion attempt
-- PRD Reference: Section 13
-- ============================================================================
CREATE TABLE IF NOT EXISTS ingestion_events_v2 (
    ingestion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL,
    
    -- Source tracking
    source ingestion_source NOT NULL,
    status ingestion_status NOT NULL DEFAULT 'RECEIVED',
    
    -- File reference
    raw_file_path VARCHAR(512),
    source_fingerprint VARCHAR(128),  -- hash(sender + filename + date_range)
    
    -- Timestamps
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    
    -- Error handling
    failure_reason TEXT,
    
    -- Metadata (sender, filename, row counts, warnings)
    metadata JSONB DEFAULT '{}',
    
    -- Constraints
    CONSTRAINT unique_fingerprint_per_account UNIQUE (account_id, source_fingerprint)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ingestion_events_v2_account 
    ON ingestion_events_v2(account_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_events_v2_status 
    ON ingestion_events_v2(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_events_v2_received 
    ON ingestion_events_v2(received_at DESC);

-- ============================================================================
-- TABLE: search_terms_v2
-- Purpose: Normalized search term data from ingested reports
-- PRD Reference: Section 12
-- ============================================================================
CREATE TABLE IF NOT EXISTS search_terms_v2 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL,
    ingestion_id UUID NOT NULL REFERENCES ingestion_events_v2(ingestion_id),
    
    -- Report data (from CSV)
    report_date DATE NOT NULL,
    campaign_name VARCHAR(512) NOT NULL,
    ad_group_name VARCHAR(512) NOT NULL,
    search_term TEXT NOT NULL,
    
    -- Metrics
    impressions INTEGER NOT NULL DEFAULT 0,
    clicks INTEGER NOT NULL DEFAULT 0,
    spend DECIMAL(10,2) NOT NULL DEFAULT 0,
    sales_7d DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_search_terms_v2_account 
    ON search_terms_v2(account_id);
CREATE INDEX IF NOT EXISTS idx_search_terms_v2_ingestion 
    ON search_terms_v2(ingestion_id);
CREATE INDEX IF NOT EXISTS idx_search_terms_v2_date 
    ON search_terms_v2(report_date);
CREATE INDEX IF NOT EXISTS idx_search_terms_v2_campaign 
    ON search_terms_v2(campaign_name);

-- Composite index for deduplication checks
CREATE INDEX IF NOT EXISTS idx_search_terms_v2_composite 
    ON search_terms_v2(account_id, report_date, campaign_name, ad_group_name, search_term);

-- ============================================================================
-- COMMENTS (for documentation)
-- ============================================================================
COMMENT ON TABLE ingestion_events_v2 IS 'V2 Ingestion audit trail - tracks every ingestion attempt';
COMMENT ON TABLE search_terms_v2 IS 'V2 Normalized search term data - isolated from V1 tables';
COMMENT ON COLUMN ingestion_events_v2.source_fingerprint IS 'hash(sender + filename + date_range) for duplicate detection';
COMMENT ON COLUMN ingestion_events_v2.metadata IS 'JSONB: {sender, filename, row_count, dropped_rows, warnings[]}';
