"""
Database Manager Module

SQLite persistence with upsert logic and test/live mode support.
Uses pathlib for cross-platform path handling.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import date, datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import uuid
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Use explicit path relative to this file's location
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars


class DatabaseManager:
    """
    SQLite database manager with upsert support.
    
    Usage:
        db = DatabaseManager(Path("data/ppc_live.db"))
        db.save_weekly_stats("client_a", date(2024, 1, 1), date(2024, 1, 7), 100.0, 500.0, 5.0)
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (will be created if doesn't exist)
        """
        self.db_path = Path(db_path)
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._init_schema()
    
    @property
    def placeholder(self) -> str:
        """SQL parameter placeholder for SQLite."""
        return "?"
    
    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like row access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Weekly Stats Table with UNIQUE constraint for upsert
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weekly_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    spend REAL DEFAULT 0,
                    sales REAL DEFAULT 0,
                    roas REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, start_date)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_weekly_stats_client_date 
                ON weekly_stats(client_id, start_date)
            """)
            
            # ==========================================
            # TARGET STATS TABLE (Granular Performance)
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS target_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    campaign_name TEXT NOT NULL,
                    ad_group_name TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    match_type TEXT,
                    spend REAL DEFAULT 0,
                    sales REAL DEFAULT 0,
                    clicks INTEGER DEFAULT 0,
                    impressions INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, start_date, campaign_name, ad_group_name, target_text)
                )
            """)
            
            # Index for target stats queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_target_stats_lookup 
                ON target_stats(client_id, start_date, campaign_name)
            """)
            
            # MIGRATION: Ensure 'orders' column exists
            try:
                cursor.execute("SELECT orders FROM target_stats LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE target_stats ADD COLUMN orders INTEGER DEFAULT 0")
            
            # ==========================================
            # ACTIONS LOG TABLE (Change History)
            # Uses UNIQUE constraint to enable upsert (overwrite, not duplicate)
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    client_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    entity_name TEXT,
                    action_type TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    reason TEXT,
                    campaign_name TEXT,
                    ad_group_name TEXT,
                    target_text TEXT,
                    match_type TEXT,
                    UNIQUE(client_id, action_date, target_text, action_type, campaign_name)
                )
            """)
            
            # Index for action log queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_actions_log_batch 
                ON actions_log(batch_id, action_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_actions_log_client 
                ON actions_log(client_id, action_date)
            """)
            
            # ==========================================
            # MAPPING TABLES (Persistence)
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS category_mappings (
                    client_id TEXT NOT NULL,
                    sku TEXT NOT NULL,
                    category TEXT,
                    sub_category TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (client_id, sku)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advertised_product_cache (
                    client_id TEXT NOT NULL,
                    campaign_name TEXT,
                    ad_group_name TEXT,
                    sku TEXT,
                    asin TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, campaign_name, ad_group_name, sku)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bulk_mappings (
                    client_id TEXT NOT NULL,
                    campaign_name TEXT,
                    campaign_id TEXT,
                    ad_group_name TEXT,
                    ad_group_id TEXT,
                    keyword_text TEXT,
                    keyword_id TEXT,
                    targeting_expression TEXT,
                    targeting_id TEXT,
                    sku TEXT,
                    match_type TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, campaign_name, ad_group_name, keyword_text, targeting_expression)
                )
            """)
            
            # MIGRATION: Add missing columns if table already exists
            for col_name in ['keyword_id', 'targeting_id', 'keyword_text', 'targeting_expression', 'match_type']:
                try:
                    cursor.execute(f"SELECT {col_name} FROM bulk_mappings LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute(f"ALTER TABLE bulk_mappings ADD COLUMN {col_name} TEXT")

            # ==========================================
            # ACCOUNTS TABLE (Multi-Account Support)
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    account_name TEXT NOT NULL,
                    account_type TEXT DEFAULT 'brand',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ==========================================
            # ACCOUNT HEALTH METRICS TABLE (Persistent)
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_health_metrics (
                    client_id TEXT PRIMARY KEY,
                    health_score REAL DEFAULT 0,
                    roas_score REAL DEFAULT 0,
                    waste_score REAL DEFAULT 0,
                    cvr_score REAL DEFAULT 0,
                    waste_ratio REAL DEFAULT 0,
                    wasted_spend REAL DEFAULT 0,
                    current_roas REAL DEFAULT 0,
                    current_acos REAL DEFAULT 0,
                    cvr REAL DEFAULT 0,
                    total_spend REAL DEFAULT 0,
                    total_sales REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # default_client auto-creation REMOVED.
            # Explicit account creation required.

    
    # ==========================================
    # UPSERT OPERATIONS
    # ==========================================
    
    def save_weekly_stats(
        self, 
        client_id: str, 
        start_date: date, 
        end_date: date, 
        spend: float, 
        sales: float, 
        roas: Optional[float] = None
    ) -> int:
        """
        Save or update weekly stats using upsert logic.
        
        If a record with the same (client_id, start_date) exists, it will be replaced.
        
        Args:
            client_id: Client identifier
            start_date: Week start date
            end_date: Week end date
            spend: Total spend for the period
            sales: Total sales for the period
            roas: Return on ad spend (calculated if not provided)
            
        Returns:
            Row ID of inserted/updated record
        """
        if roas is None:
            roas = sales / spend if spend > 0 else 0.0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # INSERT OR REPLACE handles the upsert based on UNIQUE constraint
            cursor.execute("""
                INSERT OR REPLACE INTO weekly_stats 
                (client_id, start_date, end_date, spend, sales, roas, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (client_id, start_date.isoformat(), end_date.isoformat(), spend, sales, roas))
            
            return cursor.lastrowid
    
    def save_weekly_stats_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        Batch save multiple weekly stats records.
        
        Args:
            records: List of dicts with keys: client_id, start_date, end_date, spend, sales, roas
            
        Returns:
            Number of records saved
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for record in records:
                roas = record.get('roas')
                if roas is None:
                    spend = record.get('spend', 0)
                    sales = record.get('sales', 0)
                    roas = sales / spend if spend > 0 else 0.0
                
                start_date = record['start_date']
                end_date = record['end_date']
                
                # Handle both date objects and strings
                if isinstance(start_date, date):
                    start_date = start_date.isoformat()
                if isinstance(end_date, date):
                    end_date = end_date.isoformat()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO weekly_stats 
                    (client_id, start_date, end_date, spend, sales, roas, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    record['client_id'], 
                    start_date, 
                    end_date, 
                    record.get('spend', 0), 
                    record.get('sales', 0), 
                    roas
                ))
            
            return len(records)

    def save_target_stats_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        Batch save target/keyword performance stats.
        
        Args:
            records: List of dicts with: 
                     client_id, start_date, campaign_name, ad_group_name, 
                     target_text, match_type, spend, sales, orders, clicks, impressions
        """
        if not records:
            return 0
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data tuples
            data = []
            for r in records:
                start_date = r.get('start_date')
                if isinstance(start_date, date):
                    start_date = start_date.isoformat()
                elif isinstance(start_date, pd.Timestamp):
                    start_date = start_date.strftime('%Y-%m-%d')
                    
                data.append((
                    r.get('client_id', 'default_client'),
                    start_date,
                    r.get('campaign_name', ''),
                    r.get('ad_group_name', ''),
                    r.get('target_text', ''),
                    r.get('match_type', '-'),
                    r.get('spend', 0.0),
                    r.get('sales', 0.0),
                    r.get('orders', 0),
                    r.get('clicks', 0),
                    r.get('impressions', 0)
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO target_stats 
                (client_id, start_date, campaign_name, ad_group_name, target_text, match_type,
                 spend, sales, orders, clicks, impressions, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, data)
            
            return len(data)
    
    # ==========================================
    # QUERY OPERATIONS
    # ==========================================
    
    def get_all_weekly_stats(self) -> List[Dict[str, Any]]:
        """Get all weekly stats records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM weekly_stats ORDER BY start_date DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats_by_client(self, client_id: str) -> List[Dict[str, Any]]:
        """Get weekly stats for a specific client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM weekly_stats WHERE client_id = ? ORDER BY start_date DESC",
                (client_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_target_stats_by_account(self, account_id: str, limit: int = 50000) -> pd.DataFrame:
        """Get target stats for an account as DataFrame."""
        import pandas as pd
        with self._get_connection() as conn:
            query = """
                SELECT * FROM target_stats 
                WHERE client_id = ? 
                ORDER BY start_date DESC 
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(account_id, limit))
            return df
            
    def get_target_stats_df(self, client_id: str = 'default_client') -> pd.DataFrame:
        """
        Retrieve ALL target stats for a client as DataFrame (not just latest week).
        Used by Account Overview to display full historical data.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            query = """
                SELECT 
                    start_date as Date,
                    campaign_name as 'Campaign Name',
                    ad_group_name as 'Ad Group Name',
                    target_text as Targeting,
                    match_type as 'Match Type',
                    spend as Spend,
                    sales as Sales,
                    orders as Orders,
                    clicks as Clicks,
                    impressions as Impressions
                FROM target_stats 
                WHERE client_id = ? 
                ORDER BY start_date DESC
            """
            df = pd.read_sql(query, conn, params=(client_id,))
            
            # Post-processing types
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            return df
        finally:
            conn.close()
    
    def get_stats_by_date_range(
        self, 
        start_date: date, 
        end_date: date, 
        client_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get weekly stats within a date range.
        
        Args:
            start_date: Range start date
            end_date: Range end date
            client_id: Optional client filter
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if client_id:
                cursor.execute("""
                    SELECT * FROM weekly_stats 
                    WHERE start_date >= ? AND start_date <= ? AND client_id = ?
                    ORDER BY start_date DESC
                """, (start_date.isoformat(), end_date.isoformat(), client_id))
            else:
                cursor.execute("""
                    SELECT * FROM weekly_stats 
                    WHERE start_date >= ? AND start_date <= ?
                    ORDER BY start_date DESC
                """, (start_date.isoformat(), end_date.isoformat()))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_unique_clients(self) -> List[str]:
        """Get list of unique client IDs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT client_id FROM weekly_stats ORDER BY client_id")
            return [row[0] for row in cursor.fetchall()]
    
    # ==========================================
    # DELETE OPERATIONS
    # ==========================================
    
    def delete_stats_by_client(self, client_id: str) -> int:
        """Delete all stats and actions for a specific client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM weekly_stats WHERE client_id = ?", (client_id,))
            rows = cursor.rowcount
            
            cursor.execute("DELETE FROM target_stats WHERE client_id = ?", (client_id,))
            rows += cursor.rowcount
            
            cursor.execute("DELETE FROM actions_log WHERE client_id = ?", (client_id,))
            rows += cursor.rowcount
            
            return rows
    
    def clear_all_stats(self) -> int:
        """Clear all weekly stats (use with caution!)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM weekly_stats")
            return cursor.rowcount
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_connection_status(self) -> tuple[str, str]:
        """
        Check database connection status.
        Returns: (status_message, status_color)
        """
        if self.db_path.exists():
            return "Connected", "green"
        return "Not Connected (File missing)", "red"

    # ==========================================
    # ACCOUNT HEALTH METRICS OPERATIONS
    # ==========================================
    
    def save_account_health(self, client_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Save or update account health metrics.
        
        Args:
            client_id: Account identifier
            metrics: Dict with health_score, roas_score, waste_score, cvr_score, etc.
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO account_health_metrics 
                (client_id, health_score, roas_score, waste_score, cvr_score, 
                 waste_ratio, wasted_spend, current_roas, current_acos, cvr,
                 total_spend, total_sales, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                client_id,
                metrics.get('health_score', 0),
                metrics.get('roas_score', 0),
                metrics.get('efficiency_score', metrics.get('waste_score', 0)),  # Map efficiency_score to waste_score column
                metrics.get('cvr_score', 0),
                metrics.get('waste_ratio', 0),
                metrics.get('wasted_spend', 0),
                metrics.get('current_roas', 0),
                metrics.get('current_acos', 0),
                metrics.get('cvr', 0),
                metrics.get('total_spend', 0),
                metrics.get('total_sales', 0)
            ))
            return True
    
    def get_account_health(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Get account health metrics from database.
        
        Args:
            client_id: Account identifier
            
        Returns:
            Dict with health metrics or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM account_health_metrics WHERE client_id = ?
            """, (client_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT client_id) as unique_clients,
                    MIN(start_date) as earliest_date,
                    MAX(start_date) as latest_date,
                    SUM(spend) as total_spend,
                    SUM(sales) as total_sales
                FROM weekly_stats
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return cursor.fetchone() is not None
    
    # ==========================================
    # TARGET STATS OPERATIONS
    # ==========================================
    
    def save_target_stats_batch(self, df: pd.DataFrame, client_id: str, start_date: Union[date, str] = None) -> int:
        """
        Save granular target-level performance stats from Search Term Report.
        
        If a date column exists in the data, splits by ISO week for WoW comparison.
        Otherwise uses the provided start_date.
        
        Args:
            df: Search Term Report DataFrame with columns:
                Campaign Name, Ad Group Name, Customer Search Term/Targeting,
                Spend, Sales, Clicks, Impressions, Match Type
                Optional: Date, Start Date, Report Date (for weekly splitting)
            client_id: Client identifier
            start_date: Fallback report start date (used if no date column found)
            
        Returns:
            Number of records saved
        """
        if df is None or df.empty:
            return 0
        
        # Identify target column
        # CRITICAL: For auto campaigns, use "Targeting" column (close-match, loose-match, etc.)
        # NOT "Customer Search Term" (which has ASINs/search queries)
        target_col = None
        
        # Check if we have auto campaigns
        has_auto = False
        if 'Match Type' in df.columns:
            has_auto = df['Match Type'].astype(str).str.lower().isin(['auto', '-']).any()
        
        if has_auto and 'Targeting' in df.columns:
            # For auto campaigns, prioritize Targeting column
            target_col = 'Targeting'
        else:
            # For other campaigns, look for these columns in order
            for col in ['Customer Search Term', 'Targeting', 'Keyword Text']:
                if col in df.columns:
                    target_col = col
                    break
        
        if target_col is None:
            return 0
        
        # Required columns check
        required = ['Campaign Name', 'Ad Group Name']
        if not all(col in df.columns for col in required):
            return 0
        
        # Aggregation columns
        agg_cols = {}
        if 'Spend' in df.columns:
            agg_cols['Spend'] = 'sum'
        if 'Sales' in df.columns:
            agg_cols['Sales'] = 'sum'
        if 'Clicks' in df.columns:
            agg_cols['Clicks'] = 'sum'
        if 'Impressions' in df.columns:
            agg_cols['Impressions'] = 'sum'
        if 'Orders' in df.columns:
            agg_cols['Orders'] = 'sum'
        if 'Match Type' in df.columns:
            agg_cols['Match Type'] = 'first'
        
        if not agg_cols:
            return 0
        
        # Create working copy
        df_copy = df.copy()
        
        # ==========================================
        # DUAL COLUMN HANDLING: Targeting + CST
        # ==========================================
        # For bid optimization: Use "Targeting" column (targeting types like close-match, substitutes)
        # For harvest: Use "Customer Search Term" column (actual user search queries)
        
        # Determine which column to use for target_text (bid optimization)
        if 'Targeting' in df_copy.columns:
            target_col = 'Targeting'
        elif 'Customer Search Term' in df_copy.columns:
            target_col = 'Customer Search Term'
        elif 'Keyword Text' in df_copy.columns:
            target_col = 'Keyword Text'
        else:
            return 0
        
        # Determine CST column (for harvest)
        cst_col = 'Customer Search Term' if 'Customer Search Term' in df_copy.columns else None
        
        # ==========================================
        # WEEKLY SPLITTING LOGIC
        # ==========================================
        # Detect date column for weekly splitting
        date_col = None
        for col in ['Date', 'Start Date', 'Report Date', 'date', 'start_date']:
            if col in df_copy.columns:
                date_col = col
                break
        
        if date_col:
            # Handle Date Range strings (e.g., "Nov 03, 2024 - Nov 09, 2024") through splitting
            # SmartMapper aliases "Date Range" to "Date", so this column might contain ranges string
            if df_copy[date_col].dtype == object and df_copy[date_col].astype(str).str.contains(' - ').any():
                 df_copy[date_col] = df_copy[date_col].astype(str).str.split(' - ').str[0]

            # Parse dates and extract ISO week start (Monday)
            df_copy['_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy['_week_start'] = df_copy['_date'].dt.to_period('W-MON').dt.start_time.dt.date
            
            # Group by week
            weeks = df_copy['_week_start'].dropna().unique()
        else:
            # No date column - use provided start_date
            if start_date is None:
                start_date = datetime.now().date()
            elif isinstance(start_date, str):
                try:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                except:
                    start_date = datetime.now().date()
            
            # Normalize to Monday of the week for consistent deduplication
            # This ensures that if I upload on Tuesday, and then again on Friday, 
            # both map to Monday, triggering the REPLACE logic.
            days_since_monday = start_date.weekday()
            week_start_monday = start_date - timedelta(days=days_since_monday)
            
            df_copy['_week_start'] = week_start_monday
            weeks = [week_start_monday]
        
        # Create normalized grouping keys
        df_copy['_camp_norm'] = df_copy['Campaign Name'].astype(str).str.lower().str.strip()
        df_copy['_ag_norm'] = df_copy['Ad Group Name'].astype(str).str.lower().str.strip()
        df_copy['_target_norm'] = df_copy[target_col].astype(str).str.lower().str.strip()
        
        # Create CST norm column (for harvest - keep one representative CST per group)
        if cst_col:
            df_copy['_cst_norm'] = df_copy[cst_col].astype(str).str.lower().str.strip()
            agg_cols['_cst_norm'] = 'first'  # Keep first CST for each targeting group
        
        total_saved = 0
        
        # Process each week separately
        for week_start in weeks:
            if pd.isna(week_start):
                continue
                
            week_data = df_copy[df_copy['_week_start'] == week_start]
            
            if week_data.empty:
                continue
            
            # Aggregate by normalized keys for this week
            grouped = week_data.groupby(['_camp_norm', '_ag_norm', '_target_norm']).agg(agg_cols).reset_index()
            
            # Format week start date
            if isinstance(week_start, date):
                week_start_str = week_start.isoformat()
            else:
                week_start_str = str(week_start)[:10]  # Handle datetime
            
            # Batch insert for this week
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in grouped.iterrows():
                    # Use the pre-normalized columns from groupby
                    match_type_norm = str(row.get('Match Type', '')).lower().strip()
                    cst_value = row.get('_cst_norm', '') if cst_col else ''
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO target_stats 
                        (client_id, start_date, campaign_name, ad_group_name, target_text, 
                         match_type, spend, sales, orders, clicks, impressions, customer_search_term, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        client_id,
                        week_start_str,
                        row['_camp_norm'],
                        row['_ag_norm'],
                        row['_target_norm'],
                        match_type_norm,
                        float(row.get('Spend', 0) or 0),
                        float(row.get('Sales', 0) or 0),
                        int(row.get('Orders', 0) or 0),
                        int(row.get('Clicks', 0) or 0),
                        int(row.get('Impressions', 0) or 0),
                        cst_value
                    ))
                
                total_saved += len(grouped)
        
        return total_saved
    
    def get_target_stats(self, client_id: str, start_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Get target-level stats for a client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if start_date:
                cursor.execute("""
                    SELECT * FROM target_stats 
                    WHERE client_id = ? AND start_date = ?
                    ORDER BY campaign_name, ad_group_name
                """, (client_id, start_date.isoformat() if isinstance(start_date, date) else start_date))
            else:
                cursor.execute("""
                    SELECT * FROM target_stats 
                    WHERE client_id = ?
                    ORDER BY start_date DESC, campaign_name, ad_group_name
                """, (client_id,))
            
            return [dict(row) for row in cursor.fetchall()]

    # ==========================================
    # MAPPING PERSISTENCE OPERATIONS
    # ==========================================

    def save_category_mapping(self, df: pd.DataFrame, client_id: str):
        """Save category mapping to database."""
        if df is None or df.empty:
            return 0
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data: SKU, Category, Sub-Category
            data = []
            
            # Identify columns
            sku_col = df.columns[0] # Assume first col is SKU
            cat_col = next((c for c in df.columns if 'category' in c.lower() and 'sub' not in c.lower()), None)
            sub_col = next((c for c in df.columns if 'sub' in c.lower()), None)
            
            for _, row in df.iterrows():
                data.append((
                    client_id,
                    str(row[sku_col]),
                    str(row[cat_col]) if cat_col and pd.notna(row[cat_col]) else None,
                    str(row[sub_col]) if sub_col and pd.notna(row[sub_col]) else None
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO category_mappings (client_id, sku, category, sub_category, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, data)
            
            return len(data)

    def get_category_mappings(self, client_id: str) -> pd.DataFrame:
        """Get category map for client."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            return pd.read_sql("SELECT sku as SKU, category as Category, sub_category as 'Sub-Category' FROM category_mappings WHERE client_id = ?", conn, params=(client_id,))
        finally:
            conn.close()

    def save_advertised_product_map(self, df: pd.DataFrame, client_id: str):
        """Save advertised product report cache."""
        if df is None or df.empty:
            return 0
            
        required = ['Campaign Name', 'Ad Group Name']
        if not all(c in df.columns for c in required):
            return 0
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = []
            
            sku_col = 'SKU' if 'SKU' in df.columns else None
            asin_col = 'ASIN' if 'ASIN' in df.columns else None
            
            for _, row in df.iterrows():
                data.append((
                    client_id,
                    row['Campaign Name'],
                    row['Ad Group Name'],
                    str(row[sku_col]) if sku_col and pd.notna(row[sku_col]) else None,
                    str(row[asin_col]) if asin_col and pd.notna(row[asin_col]) else None
                ))
                
            cursor.executemany("""
                INSERT OR REPLACE INTO advertised_product_cache 
                (client_id, campaign_name, ad_group_name, sku, asin, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, data)
            
            return len(data)

    def get_advertised_product_map(self, client_id: str) -> pd.DataFrame:
        """Get advertised product cache."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            return pd.read_sql("SELECT campaign_name as 'Campaign Name', ad_group_name as 'Ad Group Name', sku as SKU, asin as ASIN FROM advertised_product_cache WHERE client_id = ?", conn, params=(client_id,))
        finally:
            conn.close()

    def save_bulk_mapping(self, df: pd.DataFrame, client_id: str):
        """Save bulk ID mapping to database, including bid data."""
        if df is None or df.empty:
            return 0
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = []
            
            # Map columns
            # Standard: Campaign Name, CampaignId, Ad Group Name, AdGroupId, SKU
            # Use 'SKU' or 'msku' or 'vendor_sku'
            sku_col = next((c for c in df.columns if c.lower() in ['sku', 'msku', 'vendor sku', 'vendor_sku']), None)
            
            # Locate ID columns
            cid_col = 'CampaignId' if 'CampaignId' in df.columns else None
            aid_col = 'AdGroupId' if 'AdGroupId' in df.columns else None
            kwid_col = 'KeywordId' if 'KeywordId' in df.columns else None
            tid_col = 'TargetingId' if 'TargetingId' in df.columns else None
            
            # Locate text columns (keyword / targeting expression)
            kw_text_col = next((c for c in df.columns if c.lower() in ['keyword text', 'customer search term']), None)
            tgt_expr_col = next((c for c in df.columns if c.lower() in ['product targeting expression', 'targetingexpression']), None)
            mt_col = 'Match Type' if 'Match Type' in df.columns else None
            
            # Locate bid columns
            agb_col = next((c for c in df.columns if c.lower() in ['ad group default bid', 'adgroupdefaultbid']), None)
            bid_col = 'Bid' if 'Bid' in df.columns else None
            
            for _, row in df.iterrows():
                # We need at least Campaign Name
                if 'Campaign Name' not in row:
                    continue
                
                # Parse bid values
                agb_val = None
                if agb_col and pd.notna(row.get(agb_col)):
                    try:
                        agb_val = float(row[agb_col])
                    except:
                        pass
                        
                bid_val = None
                if bid_col and pd.notna(row.get(bid_col)):
                    try:
                        bid_val = float(row[bid_col])
                    except:
                        pass
                    
                data.append((
                    client_id,
                    str(row['Campaign Name']),
                    str(row[cid_col]) if cid_col and pd.notna(row.get(cid_col)) else None,
                    str(row.get('Ad Group Name')) if 'Ad Group Name' in df.columns and pd.notna(row.get('Ad Group Name')) else None,
                    str(row[aid_col]) if aid_col and pd.notna(row.get(aid_col)) else None,
                    str(row[kw_text_col]) if kw_text_col and pd.notna(row.get(kw_text_col)) else None,
                    str(row[kwid_col]) if kwid_col and pd.notna(row.get(kwid_col)) else None,
                    str(row[tgt_expr_col]) if tgt_expr_col and pd.notna(row.get(tgt_expr_col)) else None,
                    str(row[tid_col]) if tid_col and pd.notna(row.get(tid_col)) else None,
                    str(row[sku_col]) if sku_col and pd.notna(row.get(sku_col)) else None,
                    str(row[mt_col]) if mt_col and pd.notna(row.get(mt_col)) else None,
                    agb_val,  # Ad Group Default Bid
                    bid_val   # Keyword/Target Bid
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO bulk_mappings 
                (client_id, campaign_name, campaign_id, ad_group_name, ad_group_id, 
                 keyword_text, keyword_id, targeting_expression, targeting_id, sku, match_type,
                 ad_group_default_bid, keyword_bid, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, data)
            
            return len(data)

    def get_bulk_mapping(self, client_id: str) -> pd.DataFrame:
        """Get bulk mapping from database, including bid data."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            return pd.read_sql("""
                SELECT 
                    campaign_name as 'Campaign Name', 
                    campaign_id as 'CampaignId', 
                    ad_group_name as 'Ad Group Name', 
                    ad_group_id as 'AdGroupId',
                    keyword_text as 'Customer Search Term',
                    keyword_id as 'KeywordId',
                    targeting_expression as 'Product Targeting Expression',
                    targeting_id as 'TargetingId',
                    sku as SKU,
                    match_type as 'Match Type',
                    ad_group_default_bid as 'Ad Group Default Bid',
                    keyword_bid as 'Bid'
                FROM bulk_mappings 
                WHERE client_id = ?
            """, conn, params=(client_id,))
        finally:
            conn.close()
    
    def get_all_clients(self) -> List[str]:
        """Get list of all client IDs with data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT DISTINCT client_id FROM target_stats ORDER BY client_id")
                return [row[0] for row in cursor.fetchall()]
            except:
                return []
    
    # ==========================================
    # ACTIONS LOG OPERATIONS
    # ==========================================
    
    def log_action_batch(self, actions: List[Dict[str, Any]], client_id: str, batch_id: Optional[str] = None, action_date: Optional[str] = None) -> int:
        """
        Bulk insert actions into the actions log.
        
        Args:
            actions: List of action dictionaries with keys:
                entity_name, action_type, old_value, new_value, reason,
                campaign_name, ad_group_name, target_text, match_type
            client_id: Client identifier
            batch_id: Unique batch identifier (auto-generated if not provided)
            action_date: Report date for time-lag matching (uses current time if not provided)
            
        Returns:
            Number of actions logged
        """
        if not actions:
            return 0
        
        if batch_id is None:
            batch_id = str(uuid.uuid4())[:8]
        
        # Use report date if provided, otherwise current timestamp
        if action_date:
            # Ensure it's a string in ISO format
            date_str = str(action_date)[:10] if action_date else datetime.now().isoformat()
        else:
            date_str = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for action in actions:
                cursor.execute("""
                    INSERT OR REPLACE INTO actions_log 
                    (action_date, client_id, batch_id, entity_name, action_type, old_value, new_value, 
                     reason, campaign_name, ad_group_name, target_text, match_type,
                     winner_source_campaign, new_campaign_name, before_match_type, after_match_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    client_id,
                    batch_id,
                    action.get('entity_name', ''),
                    action.get('action_type', 'UNKNOWN'),
                    str(action.get('old_value', '')),
                    str(action.get('new_value', '')),
                    action.get('reason', ''),
                    action.get('campaign_name', ''),
                    action.get('ad_group_name', ''),
                    action.get('target_text', ''),
                    action.get('match_type', ''),
                    action.get('winner_source_campaign'),  # New field
                    action.get('new_campaign_name'),  # New field
                    action.get('before_match_type'),  # New field
                    action.get('after_match_type')  # New field
                ))
            
            return len(actions)
    
    def get_actions_by_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all actions for a specific batch."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM actions_log 
                WHERE batch_id = ?
                ORDER BY action_date
            """, (batch_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_actions_by_client(self, client_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent actions for a client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM actions_log 
                WHERE client_id = ?
                ORDER BY action_date DESC
                LIMIT ?
            """, (client_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_action_summary(self, client_id: str) -> Dict[str, Any]:
        """Get summary of actions by type for a client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    action_type,
                    COUNT(*) as count,
                    MIN(action_date) as first_action,
                    MAX(action_date) as last_action
                FROM actions_log
                WHERE client_id = ?
                GROUP BY action_type
            """, (client_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================
    # IMPACT ANALYSIS OPERATIONS
    # ==========================================
    
    def get_action_impact(
        self, 
        client_id: str, 
        before_date: Union[date, str] = None, 
        after_date: Union[date, str] = None,
        window_days: int = 7
    ) -> pd.DataFrame:
        """
        Calculate impact using rule-based expected outcomes.
        
        OPTIMIZED: Uses batch SQL for data retrieval, then applies rules in Python.
        
        Rules:
        - NEGATIVE → After = $0 (blocked), impact = cost saved
        - HARVEST → Source After = $0, 10% lift assumption
        - BID_CHANGE → Use observed data (unpredictable)
        - PAUSE → After = $0
        """
        with self._get_connection() as conn:
            # Get the 2 most recent upload dates
            dates_query = """
                SELECT DISTINCT start_date 
                FROM target_stats 
                WHERE client_id = ?
                ORDER BY start_date DESC
                LIMIT 2
            """
            dates_df = pd.read_sql_query(dates_query, conn, params=(client_id,))
            
            if len(dates_df) < 2:
                return pd.DataFrame()
            
            after_upload_date = dates_df.iloc[0]['start_date']
            before_upload_date = dates_df.iloc[1]['start_date']
            
            # Get all creditable actions
            actions_query = """
                SELECT 
                    a.id, a.action_date, a.action_type, a.target_text, a.campaign_name,
                    a.ad_group_name, a.old_value, a.new_value, a.reason,
                    a.winner_source_campaign, a.new_campaign_name
                FROM actions_log a
                WHERE a.client_id = ?
                AND LOWER(a.action_type) NOT IN ('hold', 'monitor', 'flagged')
                ORDER BY a.action_date DESC
            """
            actions_df = pd.read_sql_query(actions_query, conn, params=(client_id,))
            
            if actions_df.empty:
                return pd.DataFrame()
            
            # Get target-level stats for BEFORE period
            before_target_df = pd.read_sql_query("""
                SELECT LOWER(target_text) as target_lower, LOWER(campaign_name) as campaign_lower,
                       SUM(spend) as spend, SUM(sales) as sales
                FROM target_stats WHERE client_id = ? AND start_date = ?
                GROUP BY LOWER(target_text), LOWER(campaign_name)
            """, conn, params=(client_id, before_upload_date))
            
            # Get target-level stats for AFTER period
            after_target_df = pd.read_sql_query("""
                SELECT LOWER(target_text) as target_lower, LOWER(campaign_name) as campaign_lower,
                       SUM(spend) as spend, SUM(sales) as sales
                FROM target_stats WHERE client_id = ? AND start_date = ?
                GROUP BY LOWER(target_text), LOWER(campaign_name)
            """, conn, params=(client_id, after_upload_date))
            
            # Get campaign-level stats (fallback)
            before_campaign_df = pd.read_sql_query("""
                SELECT LOWER(campaign_name) as campaign_lower, SUM(spend) as spend, SUM(sales) as sales
                FROM target_stats WHERE client_id = ? AND start_date = ?
                GROUP BY LOWER(campaign_name)
            """, conn, params=(client_id, before_upload_date))
            
            after_campaign_df = pd.read_sql_query("""
                SELECT LOWER(campaign_name) as campaign_lower, SUM(spend) as spend, SUM(sales) as sales
                FROM target_stats WHERE client_id = ? AND start_date = ?
                GROUP BY LOWER(campaign_name)
            """, conn, params=(client_id, after_upload_date))
        
        # Create lookup dicts
        before_target_lookup = {(row['target_lower'], row['campaign_lower']): row for _, row in before_target_df.iterrows()}
        after_target_lookup = {(row['target_lower'], row['campaign_lower']): row for _, row in after_target_df.iterrows()}
        before_campaign_lookup = {row['campaign_lower']: row for _, row in before_campaign_df.iterrows()}
        after_campaign_lookup = {row['campaign_lower']: row for _, row in after_campaign_df.iterrows()}
        
        results = []
        
        for _, action in actions_df.iterrows():
            target_lower = str(action['target_text']).lower() if action['target_text'] else ''
            campaign_lower = str(action['campaign_name']).lower() if action['campaign_name'] else ''
            action_type = str(action['action_type']).upper()
            reason = str(action['reason']).lower() if action['reason'] else ''
            
            # Get before data
            before_key = (target_lower, campaign_lower)
            if before_key in before_target_lookup:
                before_data = before_target_lookup[before_key]
                match_level = 'target'
            else:
                before_data = before_campaign_lookup.get(campaign_lower, {'spend': 0, 'sales': 0})
                match_level = 'campaign'
            
            before_spend = float(before_data.get('spend', 0) or 0)
            before_sales = float(before_data.get('sales', 0) or 0)
            
            # Get observed after data (for validation)
            after_key = (target_lower, campaign_lower)
            if after_key in after_target_lookup:
                after_data = after_target_lookup[after_key]
            else:
                after_data = after_campaign_lookup.get(campaign_lower, {'spend': 0, 'sales': 0})
            
            observed_after_spend = float(after_data.get('spend', 0) or 0)
            observed_after_sales = float(after_data.get('sales', 0) or 0)
            
            # APPLY RULES BASED ON ACTION TYPE
            if action_type in ['NEGATIVE', 'NEGATIVE_ADD']:
                # RULE: Blocked keyword → After = $0, impact = cost saved
                after_spend = 0.0
                after_sales = 0.0
                delta_spend = -before_spend
                delta_sales = -before_sales
                impact_score = before_spend  # Positive = cost saved
                
                if 'isolation' in reason or 'harvest' in reason:
                    attribution = 'isolation_negative'
                    validation = 'Part of harvest consolidation'
                    impact_score = 0
                elif before_spend == 0:
                    attribution = 'preventative'
                    validation = 'Preventative - no spend to save'
                    impact_score = 0
                else:
                    attribution = 'cost_avoidance'
                    validation = '⚠️ NOT IMPLEMENTED' if observed_after_spend > 0 else '✓ Confirmed blocked'
            
            elif action_type == 'HARVEST':
                # RULE: Source → $0, 10% lift assumption
                after_spend = 0.0
                after_sales = 0.0
                delta_spend = 0.0
                delta_sales = before_sales * 0.10
                impact_score = delta_sales
                attribution = 'harvest'
                validation = '⚠️ Source still active' if observed_after_spend > 0 else '✓ Harvested to exact'
            
            elif 'BID' in action_type:
                # BID_CHANGE: Use observed data (can't predict)
                after_spend = observed_after_spend
                after_sales = observed_after_sales
                delta_spend = observed_after_spend - before_spend
                delta_sales = observed_after_sales - before_sales
                impact_score = delta_sales - delta_spend
                attribution = 'direct_causation'
                validation = 'Observed data'
            
            elif 'PAUSE' in action_type:
                # RULE: Paused → After = $0
                after_spend = 0.0
                after_sales = 0.0
                delta_spend = -before_spend
                delta_sales = -before_sales
                impact_score = delta_sales - delta_spend
                attribution = 'structural_change'
                validation = '⚠️ Still has spend' if observed_after_spend > 0 else '✓ Confirmed paused'
            
            else:
                # Unknown - use observed
                after_spend = observed_after_spend
                after_sales = observed_after_sales
                delta_spend = observed_after_spend - before_spend
                delta_sales = observed_after_sales - before_sales
                impact_score = delta_sales - delta_spend
                attribution = 'unknown'
                validation = f'Unknown: {action_type}'
            
            is_winner = impact_score > 0
            
            results.append({
                'action_date': action['action_date'],
                'action_type': action_type,
                'target_text': action['target_text'],
                'campaign_name': action['campaign_name'],
                'ad_group_name': action['ad_group_name'],
                'old_value': action['old_value'],
                'new_value': action['new_value'],
                'reason': action['reason'],
                'before_spend': before_spend,
                'before_sales': before_sales,
                'after_spend': after_spend,
                'after_sales': after_sales,
                'observed_after_spend': observed_after_spend,
                'observed_after_sales': observed_after_sales,
                'before_date': str(before_upload_date),
                'after_date': str(after_upload_date),
                'delta_sales': delta_sales,
                'delta_spend': delta_spend,
                'impact_score': impact_score,
                'is_winner': is_winner,
                'attribution': attribution,
                'validation_status': validation,
                'match_level': match_level,
                'winner_source_campaign': action['winner_source_campaign'],
                'new_campaign_name': action['new_campaign_name'],
                'attributed_delta_sales': delta_sales if attribution == 'direct_causation' else 0,
                'attributed_delta_spend': delta_spend if attribution == 'direct_causation' else 0
            })
        
        return pd.DataFrame(results)
    
    def get_impact_summary(self, client_id: str, before_date: Union[date, str] = None, after_date: Union[date, str] = None) -> Dict[str, Any]:
        """
        Get aggregate impact metrics using ACCOUNT-LEVEL proration.
        
        Returns:
            dict with: total_actions, winners, losers, win_rate,
                      net_sales_impact (account-level), net_spend_change (account-level),
                      avg_roas_change, roi, before_period, after_period
        """
        impact_df = self.get_action_impact(client_id, before_date, after_date)
        
        if impact_df.empty:
            return {
                'total_actions': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'net_sales_impact': 0,
                'net_spend_change': 0,
                'avg_roas_change': 0,
                'roi': 0,
                'by_action_type': {},
                'before_period': '',
                'after_period': ''
            }
        
        # Get ACCOUNT-LEVEL deltas from DataFrame attrs (ground truth - no double-counting)
        account_delta_sales = impact_df.attrs.get('account_delta_sales', 0)
        account_delta_spend = impact_df.attrs.get('account_delta_spend', 0)
        before_period = impact_df.attrs.get('before_period', '')
        after_period = impact_df.attrs.get('after_period', '')
        
        # Filter to rows with actual before spend data (participated in before period)
        has_data = impact_df['before_spend'].notna() & (impact_df['before_spend'] > 0)
        impact_with_data = impact_df[has_data]
        
        total = len(impact_with_data)
        winners = int((impact_with_data['is_winner'] == True).sum()) if total > 0 else 0
        losers = total - winners
        win_rate = (winners / total * 100) if total > 0 else 0
        
        # Use ACCOUNT-LEVEL deltas for net sales/spend (ground truth, no double-counting)
        net_sales = float(account_delta_sales)
        net_spend = float(account_delta_spend)
        
        # Average ROAS change across actions
        avg_roas_change = float(impact_with_data['delta_roas'].mean()) if total > 0 else 0
        
        # ROI = (Net Sales Impact - Net Spend Change) / |Net Spend Change|
        # This is profit impact / additional spend
        roi = ((net_sales - net_spend) / abs(net_spend)) if net_spend != 0 else 0
        
        # Breakdown by action type using ATTRIBUTED deltas (prorated, sums to account total)
        by_type = {}
        if total > 0:
            for action_type in impact_with_data['action_type'].unique():
                type_data = impact_with_data[impact_with_data['action_type'] == action_type]
                by_type[action_type] = {
                    'count': len(type_data),
                    'net_sales': float(type_data['attributed_delta_sales'].sum()) if 'attributed_delta_sales' in type_data.columns else float(type_data['delta_sales'].sum()),
                    'net_spend': float(type_data['attributed_delta_spend'].sum()) if 'attributed_delta_spend' in type_data.columns else float(type_data['delta_spend'].sum()),
                    'winners': int((type_data['is_winner'] == True).sum())
                }
        
        return {
            'total_actions': total,
            'winners': winners,
            'losers': losers,
            'win_rate': win_rate,
            'net_sales_impact': net_sales,
            'net_spend_change': net_spend,
            'avg_roas_change': avg_roas_change,
            'roi': roi,
            'by_action_type': by_type,
            'before_period': before_period,
            'after_period': after_period
        }
    
    def get_available_dates(self, client_id: str) -> List[str]:
        """Get list of dates with target_stats data for a client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT start_date FROM target_stats
                WHERE client_id = ?
                ORDER BY start_date DESC
            """, (client_id,))
            return [row[0] for row in cursor.fetchall()]
    
    def get_reference_data_status(self) -> Dict[str, Any]:
        """
        Check freshness of reference data (target_stats and actions_log).
        
        Returns:
            dict with: exists, last_updated, days_ago, is_stale (>30 days)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check target_stats
            cursor.execute("""
                SELECT MAX(updated_at) as last_update, COUNT(*) as count
                FROM target_stats
            """)
            row = cursor.fetchone()
            
            if row and row[0]:
                last_update_str = row[0]
                record_count = row[1]
                
                # Parse datetime
                try:
                    if 'T' in last_update_str:
                        last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                    else:
                        last_update = datetime.strptime(last_update_str, '%Y-%m-%d %H:%M:%S')
                except:
                    last_update = datetime.now()
                
                days_ago = (datetime.now() - last_update).days
                
                return {
                    'exists': True,
                    'last_updated': last_update,
                    'days_ago': days_ago,
                    'is_stale': days_ago > 30,
                    'record_count': record_count
                }
            else:
                return {
                    'exists': False,
                    'last_updated': None,
                    'days_ago': None,
                    'is_stale': True,
                    'record_count': 0
                }
    
    # ==========================================
    # ACCOUNT MANAGEMENT OPERATIONS
    # ==========================================
    
    def create_account(self, account_id: str, account_name: str, account_type: str = 'brand', metadata: dict = None) -> bool:
        """Create a new account."""
        import json
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                metadata_json = json.dumps(metadata) if metadata else '{}'
                cursor.execute("""
                    INSERT INTO accounts (account_id, account_name, account_type, metadata)
                    VALUES (?, ?, ?, ?)
                """, (account_id, account_name, account_type, metadata_json))
                return True
        except sqlite3.IntegrityError:
            return False  # Account already exists
    
    def get_all_accounts(self) -> List[tuple]:
        """Get all accounts as list of (account_id, account_name, account_type) tuples."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT account_id, account_name, account_type FROM accounts ORDER BY account_name")
            return cursor.fetchall()
    
    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get account details by ID."""
        import json
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except:
                        result['metadata'] = {}
                return result
            return None
    
    def delete_account(self, account_id: str) -> int:
        """Delete account and ALL associated data (weekly_stats, target_stats, actions_log)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete from all tables
            cursor.execute("DELETE FROM weekly_stats WHERE client_id = ?", (account_id,))
            cursor.execute("DELETE FROM target_stats WHERE client_id = ?", (account_id,))
            cursor.execute("DELETE FROM actions_log WHERE client_id = ?", (account_id,))
            cursor.execute("DELETE FROM accounts WHERE account_id = ?", (account_id,))
            
            return cursor.rowcount
    
    def reassign_data(self, from_account: str, to_account: str, date_range: tuple) -> int:
        """Move data between accounts for a date range."""
        start_date, end_date = date_range
        
        # Convert to strings if dates
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            total_updated = 0
            
            # Update target_stats
            cursor.execute("""
                UPDATE target_stats SET client_id = ?
                WHERE client_id = ? AND start_date BETWEEN ? AND ?
            """, (to_account, from_account, start_date, end_date))
            total_updated += cursor.rowcount
            
            # Update weekly_stats
            cursor.execute("""
                UPDATE weekly_stats SET client_id = ?
                WHERE client_id = ? AND start_date BETWEEN ? AND ?
            """, (to_account, from_account, start_date, end_date))
            total_updated += cursor.rowcount
            
            # Update actions_log
            cursor.execute("""
                UPDATE actions_log SET client_id = ?
                WHERE client_id = ? AND DATE(action_date) BETWEEN ? AND ?
            """, (to_account, from_account, start_date, end_date))
            total_updated += cursor.rowcount
            
            return total_updated


# ==========================================
# TEST MODE HELPER

    # =========================================
    # DATA MIGRATION UTILITIES
    # =========================================
    
    def migrate_bid_action_types(self) -> Dict[str, Any]:
        """
        Migrate legacy BID_UPDATE action types to BID_CHANGE for consistency.
        
        Handles duplicates by:
        1. Identifying BID_UPDATE records that would conflict with existing BID_CHANGE records
        2. Deleting those BID_UPDATE duplicates (keeping the BID_CHANGE version)
        3. Updating remaining BID_UPDATE records to BID_CHANGE
        
        Returns:
            Dict with 'updated_count', 'deleted_count', 'total_bid_actions', and 'message'
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count before migration
            cursor.execute("SELECT COUNT(*) FROM actions_log WHERE action_type = 'BID_UPDATE'")
            before_count = cursor.fetchone()[0]
            
            if before_count == 0:
                return {
                    'updated_count': 0,
                    'deleted_count': 0,
                    'total_bid_actions': 0, 
                    'message': 'No BID_UPDATE records found - already migrated or no legacy data'
                }
            
            # Step 1: Delete BID_UPDATE records that would conflict with existing BID_CHANGE records
            cursor.execute("""
                DELETE FROM actions_log
                WHERE action_type = 'BID_UPDATE'
                AND EXISTS (
                    SELECT 1 FROM actions_log AS t2
                    WHERE t2.action_type = 'BID_CHANGE'
                    AND t2.client_id = actions_log.client_id
                    AND t2.action_date = actions_log.action_date
                    AND t2.target_text = actions_log.target_text
                    AND t2.campaign_name = actions_log.campaign_name
                )
            """)
            
            deleted_count = cursor.rowcount
            
            # Step 2: Update remaining BID_UPDATE records to BID_CHANGE
            cursor.execute("""
                UPDATE actions_log 
                SET action_type = 'BID_CHANGE' 
                WHERE action_type = 'BID_UPDATE'
            """)
            
            updated_count = cursor.rowcount
            
            # Count total bid change actions after migration
            cursor.execute("SELECT COUNT(*) FROM actions_log WHERE action_type = 'BID_CHANGE'")
            total_after = cursor.fetchone()[0]
            
            message_parts = []
            if deleted_count > 0:
                message_parts.append(f"Deleted {deleted_count} duplicate BID_UPDATE records")
            if updated_count > 0:
                message_parts.append(f"Updated {updated_count} BID_UPDATE→BID_CHANGE")
            message_parts.append(f"Total BID_CHANGE actions: {total_after}")
            
            return {
                'updated_count': updated_count,
                'deleted_count': deleted_count,
                'total_bid_actions': total_after,
                'message': '✅ ' + '. '.join(message_parts)
            }


# =========================================

def get_db_manager(test_mode: bool = False) -> DatabaseManager:
    """
    Factory function to get appropriate database manager.
    
    Args:
        test_mode: If True, use test database; otherwise use live database
        
    Returns:
        DatabaseManager instance
    """
    base_path = Path(__file__).parent.parent / "data"
    
    if test_mode:
        db_path = base_path / "ppc_test.db"
    else:
        db_path = base_path / "ppc_live.db"
    
    return DatabaseManager(db_path)

# ==========================================
# GLOBAL INSTANCE
# ==========================================
# Default path for live database
DEFAULT_DB_PATH = Path("data/ppc_live.db")
db_manager = DatabaseManager(DEFAULT_DB_PATH)


def get_db_manager(test_mode: bool = False):
    """Factory to get appropriate DB manager instance."""
    # Check for Cloud Database URL
    db_url = os.getenv("DATABASE_URL")
    
    if db_url and not test_mode:
        try:
            from core.postgres_manager import PostgresManager
            return PostgresManager(db_url)
        except Exception as e:
            print(f"Failed to connect to Cloud DB, falling back to SQLite: {e}")
            pass
            
    if test_mode:
        return DatabaseManager(Path("data/ppc_test.db"))
    return db_manager


