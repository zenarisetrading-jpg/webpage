"""
Script to apply PostgreSQL schema migration for Impact Analyzer Fix
"""
import os
import psycopg2
from pathlib import Path

# Database URL from environment
db_url = os.getenv('DATABASE_URL', 'postgresql://postgres.wuakeiwxkjvhsnmkzywz:Zen%40rise%40123%21@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres')

print("Connecting to PostgreSQL...")
conn = psycopg2.connect(db_url)
cursor = conn.cursor()

try:
    print("Adding new columns to actions_log table...")
    
    # Add columns with IF NOT EXISTS for safety
    cursor.execute("ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS winner_source_campaign TEXT;")
    cursor.execute("ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS new_campaign_name TEXT;")
    cursor.execute("ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS before_match_type TEXT;")
    cursor.execute("ALTER TABLE actions_log ADD COLUMN IF NOT EXISTS after_match_type TEXT;")
    
    print("✓ Columns added successfully")
    
    # Backfill existing harvest actions
    print("Backfilling existing harvest actions...")
    cursor.execute("""
        UPDATE actions_log 
        SET new_campaign_name = campaign_name
        WHERE action_type = 'harvest' 
          AND new_campaign_name IS NULL
    """)
    rows_updated = cursor.rowcount
    print(f"✓ Updated {rows_updated} harvest actions")
    
    # Commit changes
    conn.commit()
    print("\n✓ PostgreSQL migration completed successfully!")
    
    # Verify
    print("\nVerifying migration...")
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'actions_log' 
        AND column_name IN ('winner_source_campaign', 'new_campaign_name', 'before_match_type', 'after_match_type')
        ORDER BY column_name
    """)
    
    print("\nNew columns in actions_log:")
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]}")
    
except Exception as e:
    conn.rollback()
    print(f"✗ Error: {e}")
    raise
finally:
    cursor.close()
    conn.close()
    print("\nConnection closed.")
