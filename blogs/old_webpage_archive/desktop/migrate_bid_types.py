#!/usr/bin/env python3
"""
Migration Script: Standardize BID_UPDATE to BID_CHANGE

This script updates all legacy BID_UPDATE action types to BID_CHANGE
for consistent reporting across the Impact Dashboard.

Usage:
    python migrate_bid_types.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.db_manager import db_manager

def main():
    print("=" * 60)
    print("BID ACTION TYPE MIGRATION")
    print("=" * 60)
    print("\nThis will update all BID_UPDATE records to BID_CHANGE")
    print("in the actions_log table for consistent reporting.\n")
    
    # Show current database path
    print(f"Database: {db_manager.db_path}")
    print(f"Database exists: {db_manager.db_path.exists()}\n")
    
    # Run migration
    print("Running migration...")
    result = db_manager.migrate_bid_action_types()
    
    print("\n" + "=" * 60)
    print("MIGRATION RESULTS")
    print("=" * 60)
    print(f"Records updated: {result['updated_count']}")
    print(f"Total BID_CHANGE records: {result['total_bid_actions']}")
    print(f"\n{result['message']}")
    print("=" * 60)
    
    if result['updated_count'] > 0:
        print("\n✅ Migration completed successfully!")
        print("   Your Impact Dashboard will now show a single 'BID_CHANGE' category")
        print("   combining all bid optimization actions.\n")
    else:
        print("\nℹ️  No migration needed - database is already up to date.\n")

if __name__ == "__main__":
    main()
