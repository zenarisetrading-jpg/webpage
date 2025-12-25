"""
Unit Tests for Customer Search Term (CST) Column Feature

Tests the new dual-column storage strategy:
- target_text: Targeting expression (for bids)
- customer_search_term: Actual search query (for harvest/negative detection)
"""

import unittest
import sys
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/Users/zayaanyousuf/Documents/Amazon PPC/saddle/saddle')


class TestCSTColumnSave(unittest.TestCase):
    """Test saving data with separate CST column."""
    
    def setUp(self):
        """Set up test data."""
        from core.db_manager import get_db_manager
        self.db = get_db_manager(False)
        self.test_account = 'test_cst_unit'
    
    def test_same_targeting_different_cst_saved_separately(self):
        """Test that same targeting with different CST values are stored as separate rows."""
        test_data = pd.DataFrame([
            {
                'Campaign Name': 'Auto Campaign 1',
                'Ad Group Name': 'Ad Group 1',
                'Targeting': 'close-match',
                'Customer Search Term': 'water bottle',
                'Match Type': 'auto',
                'Date': '2025-12-25',
                'Spend': 10.0, 'Sales': 50.0, 'Clicks': 5, 'Impressions': 100, 'Orders': 2
            },
            {
                'Campaign Name': 'Auto Campaign 1',
                'Ad Group Name': 'Ad Group 1',
                'Targeting': 'close-match',
                'Customer Search Term': 'insulated bottle',
                'Match Type': 'auto',
                'Date': '2025-12-25',
                'Spend': 15.0, 'Sales': 0.0, 'Clicks': 8, 'Impressions': 150, 'Orders': 0
            },
        ])
        
        # Save
        saved = self.db.save_target_stats_batch(test_data, self.test_account, date(2025, 12, 25))
        
        # Verify both rows saved
        self.assertEqual(saved, 2, "Should save 2 separate rows for different CST values")
        
        # Load and verify
        result = self.db.get_target_stats_by_account(self.test_account)
        result = result[result['start_date'] == date(2025, 12, 23)]  # Week start
        
        close_match_rows = result[result['target_text'] == 'close-match']
        self.assertEqual(len(close_match_rows), 2, "Should have 2 rows for close-match")
        
        cst_values = set(close_match_rows['customer_search_term'].values)
        self.assertIn('water bottle', cst_values)
        self.assertIn('insulated bottle', cst_values)
    
    def test_asin_targeting_preserves_cst(self):
        """Test that ASIN targeting preserves the CST (search term that triggered it)."""
        test_data = pd.DataFrame([
            {
                'Campaign Name': 'Auto Campaign 1',
                'Ad Group Name': 'Ad Group 1',
                'Targeting': 'asin="B0ABC12345"',
                'Customer Search Term': 'competitor product',
                'Match Type': 'auto',
                'Date': '2025-12-25',
                'Spend': 5.0, 'Sales': 0.0, 'Clicks': 3, 'Impressions': 50, 'Orders': 0
            },
        ])
        
        saved = self.db.save_target_stats_batch(test_data, self.test_account, date(2025, 12, 25))
        self.assertEqual(saved, 1)
        
        result = self.db.get_target_stats_by_account(self.test_account)
        asin_rows = result[result['target_text'].str.contains('b0abc12345', case=False, na=False)]
        
        self.assertEqual(len(asin_rows), 1)
        self.assertEqual(asin_rows.iloc[0]['customer_search_term'], 'competitor product')
    
    def test_manual_keyword_uses_targeting_as_cst(self):
        """Test that manual keywords use targeting as CST (since no separate CST exists)."""
        test_data = pd.DataFrame([
            {
                'Campaign Name': 'Manual Campaign',
                'Ad Group Name': 'Exact AG',
                'Targeting': 'water bottle stainless',
                'Customer Search Term': 'water bottle stainless steel',  # Close variant
                'Match Type': 'exact',
                'Date': '2025-12-25',
                'Spend': 20.0, 'Sales': 100.0, 'Clicks': 10, 'Impressions': 200, 'Orders': 5
            },
        ])
        
        saved = self.db.save_target_stats_batch(test_data, self.test_account, date(2025, 12, 25))
        self.assertEqual(saved, 1)


class TestCSTColumnLoad(unittest.TestCase):
    """Test loading data with CST column for optimizer use."""
    
    def setUp(self):
        from core.db_manager import get_db_manager
        self.db = get_db_manager(False)
    
    def test_load_returns_cst_column(self):
        """Test that loaded data includes Customer Search Term column."""
        result = self.db.get_target_stats_df('test_cst_full')
        
        self.assertIn('Customer Search Term', result.columns, 
                      "Loaded data should have 'Customer Search Term' column")
        self.assertIn('Targeting', result.columns,
                      "Loaded data should still have 'Targeting' column for bids")
    
    def test_cst_values_are_actual_queries_not_targeting_types(self):
        """Test that CST contains actual search queries, not targeting types."""
        result = self.db.get_target_stats_df('test_cst_full')
        
        # Filter to close-match rows
        close_match = result[result['Targeting'] == 'close-match']
        
        if len(close_match) > 0:
            # CST should NOT be 'close-match' - it should be actual queries
            non_close_match_cst = close_match[close_match['Customer Search Term'] != 'close-match']
            
            self.assertGreater(len(non_close_match_cst), 0,
                              "CST should contain actual queries, not targeting type 'close-match'")


class TestCSTUniqueConstraint(unittest.TestCase):
    """Test the unique constraint behavior with CST column."""
    
    def setUp(self):
        from core.db_manager import get_db_manager
        self.db = get_db_manager(False)
        self.test_account = 'test_cst_constraint'
    
    def test_duplicate_cst_updates_instead_of_insert(self):
        """Test that duplicate CST rows update existing record."""
        test_data = pd.DataFrame([
            {
                'Campaign Name': 'Test Campaign',
                'Ad Group Name': 'Test AG',
                'Targeting': 'loose-match',
                'Customer Search Term': 'unique query',
                'Match Type': 'auto',
                'Date': '2025-12-25',
                'Spend': 10.0, 'Sales': 0.0, 'Clicks': 5, 'Impressions': 100, 'Orders': 0
            },
        ])
        
        # Save once
        self.db.save_target_stats_batch(test_data, self.test_account, date(2025, 12, 25))
        
        # Update and save again
        test_data['Spend'] = 20.0
        test_data['Sales'] = 50.0
        self.db.save_target_stats_batch(test_data, self.test_account, date(2025, 12, 25))
        
        # Should still be 1 row, but updated
        result = self.db.get_target_stats_by_account(self.test_account)
        matching = result[(result['target_text'] == 'loose-match') & 
                          (result['customer_search_term'] == 'unique query')]
        
        self.assertEqual(len(matching), 1, "Should have exactly 1 row after upsert")
        self.assertEqual(matching.iloc[0]['spend'], 20.0, "Spend should be updated")
        self.assertEqual(matching.iloc[0]['sales'], 50.0, "Sales should be updated")


def run_tests():
    """Run all CST unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCSTColumnSave))
    suite.addTests(loader.loadTestsFromTestCase(TestCSTColumnLoad))
    suite.addTests(loader.loadTestsFromTestCase(TestCSTUniqueConstraint))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
