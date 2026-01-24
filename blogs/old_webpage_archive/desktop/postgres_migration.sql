-- PostgreSQL Schema Migration for Impact Analyzer Fix
-- This script adds the new columns to the actions_log table

ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS winner_source_campaign TEXT;
ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS new_campaign_name TEXT;
ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS before_match_type TEXT;
ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS after_match_type TEXT;

-- Update existing harvest actions to populate campaign name as "new_campaign_name"
UPDATE actions_log 
SET new_campaign_name = campaign_name
WHERE action_type = 'harvest' 
  AND new_campaign_name IS NULL;

-- Verification query
SELECT 
    action_type,
    target_text,
    campaign_name,
    winner_source_campaign,
    new_campaign_name,
    before_match_type,
    after_match_type,
    action_date
FROM actions_log
WHERE action_date >= CURRENT_DATE - INTERVAL '30 days'
LIMIT 10;
