-- Migration: Add customer_search_term column to target_stats
-- This column stores the actual search query that triggered the ad
-- Separate from target_text which stores the targeting expression (for bids)
--
-- Run with: psql -d your_database -f add_cst_column.sql

-- Add the new column (nullable to support existing data)
ALTER TABLE target_stats 
ADD COLUMN IF NOT EXISTS customer_search_term TEXT;

-- Create index for efficient harvest/negative detection queries
CREATE INDEX IF NOT EXISTS idx_target_stats_cst 
ON target_stats(customer_search_term);

-- Update the unique constraint to include customer_search_term
-- First drop the old constraint
ALTER TABLE target_stats 
DROP CONSTRAINT IF EXISTS target_stats_client_id_start_date_campaign_name_ad_group_na_key;

-- Create new unique constraint including CST
-- Note: We use COALESCE to handle NULL CST values for backward compatibility
CREATE UNIQUE INDEX IF NOT EXISTS target_stats_unique_row
ON target_stats(client_id, start_date, campaign_name, ad_group_name, target_text, COALESCE(customer_search_term, ''));

-- Verify the column was added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'target_stats' AND column_name = 'customer_search_term';
