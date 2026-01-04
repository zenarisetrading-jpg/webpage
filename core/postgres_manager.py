"""
PostgreSQL Database Manager for Supabase Integration.

Implements the same interface as DatabaseManager but uses psycopg2 and PostgreSQL syntax.
Handles 'ON CONFLICT' for upserts instead of 'INSERT OR REPLACE'.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
from typing import Optional, List, Dict, Any, Union
from datetime import date, datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import uuid
import time
import functools

# ==========================================
# BID VALIDATION CONFIGURATION
# ==========================================
BID_VALIDATION_CONFIG = {
    "cpc_match_threshold": 0.20,              # 20% tolerance (per PRD 4.5.4)
    "cpc_directional_threshold": 0.03,        # >3% CPC change (more sensitive)
    "min_impressions_before": 20,             # Lowered for long-tail (was 50)
    "impressions_increase_threshold": 0.15,   # +15% for bid ups (was 20%)
    "impressions_decrease_threshold": 0.10,   # -10% for bid downs (was 15%)
    "combined_impressions_threshold": 0.08,   # For combined validation (was 10%)
}

# ==========================================
# HARVEST VALIDATION CONFIGURATION
# ==========================================
HARVEST_VALIDATION_CONFIG = {
    # Tier thresholds (source spend drop %)
    "complete_block_threshold": 0,        # $0 spend = complete
    "near_complete_threshold": 0.90,      # 90%+ drop = near complete
    "strong_migration_threshold": 0.75,   # 75%+ drop with growth = strong
    "partial_migration_threshold": 0.50,  # 50%+ drop with 25%+ growth = partial
    
    # Exact match growth requirement for lower tiers
    "exact_growth_required_for_partial": 0.25,  # 25% growth
    
    # Minimum source spend to validate (avoid noise)
    "min_source_before_spend": 5.0,
}

# ==========================================
# PERFORMANCE: Simple TTL Cache
# ==========================================
class TTLCache:
    """Simple time-based cache for query results."""
    def __init__(self, ttl_seconds: int = 60):
        self._cache = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self._cache[key] = (value, time.time())
    
    def clear(self):
        self._cache.clear()

# Global cache instance
_query_cache = TTLCache(ttl_seconds=60)

def retry_on_connection_error(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying database operations with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                        print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        # Reset connection pool on error
                        if hasattr(args[0], '_reset_pool'):
                            args[0]._reset_pool()
            raise last_error
        return wrapper
    return decorator

class PostgresManager:
    """
    PostgreSQL persistence for Supabase / Cloud Postgres.
    Uses connection pooling with retry logic and health checking.
    """
    
    _pool = None  # Class-level connection pool
    _pool_lock = None  # For thread safety
    
    def __init__(self, db_url: str):
        """
        Initialize Postgres manager with resilient connection pooling.
        
        Args:
            db_url: Postgres connection string (postgres://user:pass@host:port/db)
        """
        self.db_url = db_url
        self._init_pool()
        self._init_schema()
    
    def _init_pool(self):
        """Initialize or reinitialize connection pool with optimal settings."""
        if PostgresManager._pool is not None:
            return
        
        # Parse and add connection options for resilience
        # Add timeout and keepalive settings
        dsn = self.db_url
        if '?' not in dsn:
            dsn += '?'
        else:
            dsn += '&'
        dsn += 'connect_timeout=10&keepalives=1&keepalives_idle=30&keepalives_interval=5&keepalives_count=3'
        
        PostgresManager._pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=5,  # Reduced from 10 to prevent exhaustion
            dsn=dsn
        )
    
    def _reset_pool(self):
        """Reset connection pool after errors."""
        if PostgresManager._pool is not None:
            try:
                PostgresManager._pool.closeall()
            except:
                pass
            PostgresManager._pool = None
        self._init_pool()
    
    @property
    def placeholder(self) -> str:
        """SQL parameter placeholder for Postgres."""
        return "%s"
    
    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections with health check."""
        conn = None
        try:
            conn = PostgresManager._pool.getconn()
            
            # Health check: test if connection is alive
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except:
                # Connection is stale, get a fresh one
                PostgresManager._pool.putconn(conn, close=True)
                conn = PostgresManager._pool.getconn()
            
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                PostgresManager._pool.putconn(conn)
    
    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Weekly Stats Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS weekly_stats (
                        id SERIAL PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        start_date DATE NOT NULL,
                        end_date DATE NOT NULL,
                        spend DOUBLE PRECISION DEFAULT 0,
                        sales DOUBLE PRECISION DEFAULT 0,
                        roas DOUBLE PRECISION DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(client_id, start_date)
                    )
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_weekly_stats_client_date ON weekly_stats(client_id, start_date)")
                
                # Target Stats Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS target_stats (
                        id SERIAL PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        start_date DATE NOT NULL,
                        campaign_name TEXT NOT NULL,
                        ad_group_name TEXT NOT NULL,
                        target_text TEXT NOT NULL,
                        match_type TEXT,
                        spend DOUBLE PRECISION DEFAULT 0,
                        sales DOUBLE PRECISION DEFAULT 0,
                        clicks INTEGER DEFAULT 0,
                        impressions INTEGER DEFAULT 0,
                        orders INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(client_id, start_date, campaign_name, ad_group_name, target_text)
                    )
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_target_stats_lookup ON target_stats(client_id, start_date, campaign_name)")
                
                # Actions Log Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS actions_log (
                        id SERIAL PRIMARY KEY,
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
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_log_batch ON actions_log(batch_id, action_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_log_client ON actions_log(client_id, action_date)")
                
                # Category Mappings
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
                
                # Advertised Product Cache
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
                
                # Bulk Mappings
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
                
                # Accounts
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
                
                # Account Health Metrics (for Home page cockpit)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS account_health_metrics (
                        client_id TEXT PRIMARY KEY,
                        health_score DOUBLE PRECISION,
                        roas_score DOUBLE PRECISION,
                        waste_score DOUBLE PRECISION,
                        cvr_score DOUBLE PRECISION,
                        waste_ratio DOUBLE PRECISION,
                        wasted_spend DOUBLE PRECISION,
                        current_roas DOUBLE PRECISION,
                        current_acos DOUBLE PRECISION,
                        cvr DOUBLE PRECISION,
                        total_spend DOUBLE PRECISION,
                        total_sales DOUBLE PRECISION,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

    def save_weekly_stats(self, client_id: str, start_date: date, end_date: date, spend: float, sales: float, roas: Optional[float] = None) -> int:
        if roas is None:
            roas = sales / spend if spend > 0 else 0.0
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    INSERT INTO weekly_stats (client_id, start_date, end_date, spend, sales, roas, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (client_id, start_date) DO UPDATE SET
                        end_date = EXCLUDED.end_date,
                        spend = EXCLUDED.spend,
                        sales = EXCLUDED.sales,
                        roas = EXCLUDED.roas,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (client_id, start_date, end_date, spend, sales, roas))
                result = cursor.fetchone()
                return result['id'] if result else 0

    def save_target_stats_batch(self, df: pd.DataFrame, client_id: str, start_date: Union[date, str] = None) -> int:
        """
        Save granular target-level performance stats from Search Term Report.
        
        SYNCED WITH SQLite VERSION - includes auto campaign handling.
        """
        if df is None or df.empty:
            return 0
        
        # ==========================================
        # DUAL COLUMN STRATEGY
        # ==========================================
        # target_text = Targeting expression (close-match, asin=, keyword) → FOR BIDS
        # customer_search_term = Actual search query → FOR HARVEST/NEGATIVE
        #
        # We need BOTH columns:
        # - Bids for auto campaigns must use targeting type (close-match, loose-match)
        # - Harvest detection needs actual search queries
        
        # Column for BIDDING (targeting types, keywords, ASINs)
        target_col = None
        for col in ['Targeting', 'Customer Search Term', 'Keyword Text']:
            if col in df.columns and not df[col].isna().all():
                target_col = col
                break
        
        # Column for HARVEST/NEGATIVE (actual search queries)
        cst_col = None
        if 'Customer Search Term' in df.columns and not df['Customer Search Term'].isna().all():
            cst_col = 'Customer Search Term'
        
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
        
        
        # Note: target_col is already determined above, prioritizing Customer Search Term
        
        # ==========================================
        # WEEKLY SPLITTING LOGIC
        # ==========================================
        date_col = None
        for col in ['Date', 'Start Date', 'Report Date', 'date', 'start_date']:
            if col in df_copy.columns:
                date_col = col
                break
        
        if date_col:
            # Handle Date Range strings
            if df_copy[date_col].dtype == object and df_copy[date_col].astype(str).str.contains(' - ').any():
                df_copy[date_col] = df_copy[date_col].astype(str).str.split(' - ').str[0]
            
            df_copy['_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy['_week_start'] = df_copy['_date'].dt.to_period('W-MON').dt.start_time.dt.date
            weeks = df_copy['_week_start'].dropna().unique()
        else:
            if start_date is None:
                start_date = datetime.now().date()
            elif isinstance(start_date, str):
                try:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                except:
                    start_date = datetime.now().date()
            
            days_since_monday = start_date.weekday()
            week_start_monday = start_date - timedelta(days=days_since_monday)
            df_copy['_week_start'] = week_start_monday
            weeks = [week_start_monday]
        
        # Create normalized grouping keys
        df_copy['_camp_norm'] = df_copy['Campaign Name'].astype(str).str.lower().str.strip()
        df_copy['_ag_norm'] = df_copy['Ad Group Name'].astype(str).str.lower().str.strip()
        df_copy['_target_norm'] = df_copy[target_col].astype(str).str.lower().str.strip()
        
        # Add CST normalization for harvest/negative detection
        if cst_col:
            df_copy['_cst_norm'] = df_copy[cst_col].astype(str).str.lower().str.strip()
        else:
            df_copy['_cst_norm'] = df_copy['_target_norm']  # Fallback to target if no CST
        
        total_saved = 0
        
        for week_start in weeks:
            if pd.isna(week_start):
                continue
            
            week_data = df_copy[df_copy['_week_start'] == week_start]
            if week_data.empty:
                continue
            
            # Group by Campaign/AdGroup/Target/CST to preserve search term granularity
            grouped = week_data.groupby(['_camp_norm', '_ag_norm', '_target_norm', '_cst_norm']).agg(agg_cols).reset_index()
            
            week_start_str = week_start.isoformat() if isinstance(week_start, date) else str(week_start)[:10]
            
            # Prepare records for bulk insert (now includes CST)
            records = []
            for _, row in grouped.iterrows():
                match_type_norm = str(row.get('Match Type', '')).lower().strip()
                records.append((
                    client_id,
                    week_start_str,
                    row['_camp_norm'],
                    row['_ag_norm'],
                    row['_target_norm'],
                    row['_cst_norm'],  # NEW: customer_search_term
                    match_type_norm,
                    float(row.get('Spend', 0) or 0),
                    float(row.get('Sales', 0) or 0),
                    int(row.get('Orders', 0) or 0),
                    int(row.get('Clicks', 0) or 0),
                    int(row.get('Impressions', 0) or 0)
                ))
            
            if records:
                with self._get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        execute_values(cursor, """
                            INSERT INTO target_stats 
                            (client_id, start_date, campaign_name, ad_group_name, target_text, 
                             customer_search_term, match_type, spend, sales, orders, clicks, impressions)
                            VALUES %s
                            ON CONFLICT ON CONSTRAINT target_stats_unique_row 
                            DO UPDATE SET
                                spend = EXCLUDED.spend,
                                sales = EXCLUDED.sales,
                                orders = EXCLUDED.orders,
                                clicks = EXCLUDED.clicks,
                                impressions = EXCLUDED.impressions,
                                match_type = EXCLUDED.match_type,
                                updated_at = CURRENT_TIMESTAMP
                        """, records)
                
                total_saved += len(records)
        
        return total_saved

    def get_all_weekly_stats(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM weekly_stats ORDER BY start_date DESC")
                return cursor.fetchall()
    
    def get_stats_by_client(self, client_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM weekly_stats WHERE client_id = %s ORDER BY start_date DESC", (client_id,))
                return cursor.fetchall()

    def get_target_stats_by_account(self, account_id: str, limit: int = 50000) -> pd.DataFrame:
        with self._get_connection() as conn:
            query = "SELECT * FROM target_stats WHERE client_id = %s ORDER BY start_date DESC LIMIT %s"
            return pd.read_sql_query(query, conn, params=(account_id, limit))

    # @st.cache_data(ttl=300)  <-- Can't use decorator on method easily without refactoring self. 
    # Instead, we will implement a direct fetch with optional streamlit caching at the call site or 
    # simpler: just remove the custom cache for now to force fresh data until we move this to a helper.
    # Actually, for this specific large query, let's just disable the broken custom cache.
    
    @retry_on_connection_error()
    def get_target_stats_df(self, client_id: str = 'default_client') -> pd.DataFrame:
        """Get large historical dataset. Custom caching removed to fix stale data issues."""
        
        # cache_key = f'target_stats_df_{client_id}'
        # cached = _query_cache.get(cache_key)
        # if cached is not None:
        #     return cached

        with self._get_connection() as conn:
            query = """
                SELECT 
                    start_date as "Date",
                    campaign_name as "Campaign Name",
                    ad_group_name as "Ad Group Name",
                    target_text as "Targeting",
                    customer_search_term as "Customer Search Term",
                    match_type as "Match Type",
                    spend as "Spend",
                    sales as "Sales",
                    orders as "Orders",
                    clicks as "Clicks",
                    impressions as "Impressions"
                FROM target_stats 
                WHERE client_id = %s 
                ORDER BY start_date DESC
            """
            df = pd.read_sql(query, conn, params=(client_id,))
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                print(f"DEBUG: DB fetch for {client_id}: {len(df)} rows. Range: {df['Date'].min()} to {df['Date'].max()}")
        
        # _query_cache.set(cache_key, df)
        return df
    
            
    def get_stats_by_date_range(self, start_date: date, end_date: date, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if client_id:
                    cursor.execute("""
                        SELECT * FROM weekly_stats 
                        WHERE start_date >= %s AND start_date <= %s AND client_id = %s
                        ORDER BY start_date DESC
                    """, (start_date, end_date, client_id))
                else:
                    cursor.execute("""
                        SELECT * FROM weekly_stats 
                        WHERE start_date >= %s AND start_date <= %s
                        ORDER BY start_date DESC
                    """, (start_date, end_date))
                
                return cursor.fetchall()
    
    def get_unique_clients(self) -> List[str]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT DISTINCT client_id FROM weekly_stats ORDER BY client_id")
                return [row['client_id'] for row in cursor.fetchall()]

    def delete_stats_by_client(self, client_id: str) -> int:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("DELETE FROM weekly_stats WHERE client_id = %s", (client_id,))
                rows = cursor.rowcount
                cursor.execute("DELETE FROM target_stats WHERE client_id = %s", (client_id,))
                rows += cursor.rowcount
                cursor.execute("DELETE FROM actions_log WHERE client_id = %s", (client_id,))
                rows += cursor.rowcount
                return rows

    def clear_all_stats(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("DELETE FROM weekly_stats")
                return cursor.rowcount

    def get_connection_status(self) -> tuple[str, str]:
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT 1")
            return "Connected (Postgres)", "green"
        except Exception as e:
            return f"Error: {str(e)}", "red"

    def get_stats_summary(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
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
                rows = cursor.fetchall() # Fetch all results
                
                # Convert to DataFrame
                df = pd.DataFrame(rows)
                
                # Standardize columns to match UI expectations (snake_case -> Title Case)
                rename_map = {
                    'date': 'Date',
                    'campaign_name': 'Campaign Name',
                    'ad_group_name': 'Ad Group Name', 
                    'targeting': 'Targeting',
                    'match_type': 'Match Type',
                    'impressions': 'Impressions',
                    'clicks': 'Clicks',
                    'spend': 'Spend',
                    'sales': 'Sales',
                    'orders': 'Orders',
                    'units': 'Units',
                    'ctr': 'CTR',
                    'cpc': 'CPC',
                    'cvr': 'CVR', 
                    'roas': 'ROAS',
                    'acos': 'ACoS',
                    'customer_search_term': 'Customer Search Term'
                }
                df = df.rename(columns=rename_map)
                
                # Ensure numerical types
                num_cols = ['Impressions', 'Clicks', 'Spend', 'Sales', 'Orders', 'Units']
                for col in num_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                # Ensure date type
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                return df

    def save_category_mapping(self, df: pd.DataFrame, client_id: str):
        if df is None or df.empty: return 0
        
        sku_col = df.columns[0]
        cat_col = next((c for c in df.columns if 'category' in c.lower() and 'sub' not in c.lower()), None)
        sub_col = next((c for c in df.columns if 'sub' in c.lower()), None)
        
        data = []
        for _, row in df.iterrows():
            data.append((
                client_id,
                str(row[sku_col]),
                str(row[cat_col]) if cat_col and pd.notna(row[cat_col]) else None,
                str(row[sub_col]) if sub_col and pd.notna(row[sub_col]) else None
            ))
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                execute_values(cursor, """
                    INSERT INTO category_mappings (client_id, sku, category, sub_category)
                    VALUES %s
                    ON CONFLICT (client_id, sku) DO UPDATE SET
                        category = EXCLUDED.category,
                        sub_category = EXCLUDED.sub_category,
                        updated_at = CURRENT_TIMESTAMP
                """, data)
        return len(data)

    def get_category_mappings(self, client_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql("SELECT sku as SKU, category as Category, sub_category as \"Sub-Category\" FROM category_mappings WHERE client_id = %s", conn, params=(client_id,))

    def save_advertised_product_map(self, df: pd.DataFrame, client_id: str):
        if df is None or df.empty: return 0
        
        required = ['Campaign Name', 'Ad Group Name']
        if not all(c in df.columns for c in required): return 0
        
        sku_col = 'SKU' if 'SKU' in df.columns else None
        asin_col = 'ASIN' if 'ASIN' in df.columns else None
        
        data = []
        for _, row in df.iterrows():
            data.append((
                client_id,
                row['Campaign Name'],
                row['Ad Group Name'],
                str(row[sku_col]) if sku_col and pd.notna(row[sku_col]) else None,
                str(row[asin_col]) if asin_col and pd.notna(row[asin_col]) else None
            ))
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                execute_values(cursor, """
                    INSERT INTO advertised_product_cache 
                    (client_id, campaign_name, ad_group_name, sku, asin)
                    VALUES %s
                    ON CONFLICT (client_id, campaign_name, ad_group_name, sku) DO UPDATE SET
                        asin = EXCLUDED.asin,
                        updated_at = CURRENT_TIMESTAMP
                """, data)
        return len(data)

    def get_advertised_product_map(self, client_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql("SELECT campaign_name as \"Campaign Name\", ad_group_name as \"Ad Group Name\", sku as SKU, asin as ASIN FROM advertised_product_cache WHERE client_id = %s", conn, params=(client_id,))

    def save_bulk_mapping(self, df: pd.DataFrame, client_id: str):
        if df is None or df.empty: return 0
        
        sku_col = next((c for c in df.columns if c.lower() in ['sku', 'msku', 'vendor sku', 'vendor_sku']), None)
        cid_col = 'CampaignId' if 'CampaignId' in df.columns else None
        aid_col = 'AdGroupId' if 'AdGroupId' in df.columns else None
        kwid_col = 'KeywordId' if 'KeywordId' in df.columns else None
        tid_col = 'TargetingId' if 'TargetingId' in df.columns else None
        
        kw_text_col = next((c for c in df.columns if c.lower() in ['keyword text', 'customer search term']), None)
        tgt_expr_col = next((c for c in df.columns if c.lower() in ['product targeting expression', 'targetingexpression']), None)
        mt_col = 'Match Type' if 'Match Type' in df.columns else None
        
        data = []
        for _, row in df.iterrows():
            if 'Campaign Name' not in row: continue
            
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
                str(row[mt_col]) if mt_col and pd.notna(row.get(mt_col)) else None
            ))
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                execute_values(cursor, """
                    INSERT INTO bulk_mappings 
                    (client_id, campaign_name, campaign_id, ad_group_name, ad_group_id, 
                        keyword_text, keyword_id, targeting_expression, targeting_id, sku, match_type)
                    VALUES %s
                    ON CONFLICT (client_id, campaign_name, ad_group_name, keyword_text, targeting_expression) DO UPDATE SET
                        campaign_id = EXCLUDED.campaign_id,
                        ad_group_id = EXCLUDED.ad_group_id,
                        keyword_id = EXCLUDED.keyword_id,
                        targeting_id = EXCLUDED.targeting_id,
                        sku = EXCLUDED.sku,
                        match_type = EXCLUDED.match_type,
                        updated_at = CURRENT_TIMESTAMP
                """, data)
        return len(data)

    def get_bulk_mapping(self, client_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql("""
                SELECT 
                    campaign_name as "Campaign Name", 
                    campaign_id as "CampaignId", 
                    ad_group_name as "Ad Group Name", 
                    ad_group_id as "AdGroupId",
                    keyword_text as "Customer Search Term",
                    keyword_id as "KeywordId",
                    targeting_expression as "Product Targeting Expression",
                    targeting_id as "TargetingId",
                    sku as "SKU",
                    match_type as "Match Type"
                FROM bulk_mappings 
                WHERE client_id = %s
            """, conn, params=(client_id,))

    def log_action_batch(self, actions: List[Dict[str, Any]], client_id: str, batch_id: Optional[str] = None, action_date: Optional[str] = None) -> int:
        if not actions: return 0
        if batch_id is None: batch_id = str(uuid.uuid4())[:8]
        if action_date:
            date_str = str(action_date)[:10] if action_date else datetime.now().isoformat()
        else:
            date_str = datetime.now().isoformat()
            
        data = []
        # Track unique constraint keys to prevent duplicates in same batch
        seen_keys = set()
        
        for action in actions:
            # Unique key matches ON CONFLICT constraint
            unique_key = (
                client_id,
                date_str[:10],  # action_date
                action.get('target_text', ''),
                action.get('action_type', 'UNKNOWN'),
                action.get('campaign_name', '')
            )
            
            # Skip if we've already seen this combination
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)
            
            data.append((
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
                action.get('winner_source_campaign'),
                action.get('new_campaign_name'),
                action.get('before_match_type'),
                action.get('after_match_type')
            ))
        
        if not data:
            return 0
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                execute_values(cursor, """
                    INSERT INTO actions_log 
                    (action_date, client_id, batch_id, entity_name, action_type, old_value, new_value, 
                     reason, campaign_name, ad_group_name, target_text, match_type,
                     winner_source_campaign, new_campaign_name, before_match_type, after_match_type)
                    VALUES %s
                    ON CONFLICT (client_id, action_date, target_text, action_type, campaign_name) 
                    DO UPDATE SET
                        batch_id = EXCLUDED.batch_id,
                        entity_name = EXCLUDED.entity_name,
                        old_value = EXCLUDED.old_value,
                        new_value = EXCLUDED.new_value,
                        reason = EXCLUDED.reason,
                        ad_group_name = EXCLUDED.ad_group_name,
                        match_type = EXCLUDED.match_type,
                        winner_source_campaign = EXCLUDED.winner_source_campaign,
                        new_campaign_name = EXCLUDED.new_campaign_name,
                        before_match_type = EXCLUDED.before_match_type,
                        after_match_type = EXCLUDED.after_match_type
                """, data)
        return len(data)

    def delete_action_batch(self, client_id: str, batch_id: str) -> int:
        """Delete a specific action batch (for undo functionality)."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "DELETE FROM actions_log WHERE client_id = %s AND batch_id = %s",
                    (client_id, batch_id)
                )
                return cursor.rowcount

    def clear_todays_actions(self, client_id: str) -> int:
        """Delete all actions logged today for a client."""
        from datetime import date
        today = date.today().isoformat()
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "DELETE FROM actions_log WHERE client_id = %s AND DATE(action_date) = %s",
                    (client_id, today)
                )
                return cursor.rowcount


    def create_account(self, account_id: str, account_name: str, account_type: str = 'brand', metadata: dict = None) -> bool:
        import json
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    metadata_json = json.dumps(metadata) if metadata else '{}'
                    cursor.execute("""
                        INSERT INTO accounts (account_id, account_name, account_type, metadata)
                        VALUES (%s, %s, %s, %s)
                    """, (account_id, account_name, account_type, metadata_json))
                    return True
        except psycopg2.IntegrityError:
            return False

    @retry_on_connection_error()
    def get_all_accounts(self) -> List[tuple]:
        """Get all accounts with caching."""
        cache_key = 'all_accounts'
        cached = _query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT account_id, account_name, account_type FROM accounts ORDER BY account_name")
                result = [(row['account_id'], row['account_name'], row['account_type']) for row in cursor.fetchall()]
        
        _query_cache.set(cache_key, result)
        return result

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        import json
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM accounts WHERE account_id = %s", (account_id,))
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

    # ==========================================
    # ACCOUNT HEALTH METHODS
    # ==========================================
    
    def save_account_health(self, client_id: str, metrics: Dict[str, Any]) -> bool:
        """Save or update account health metrics."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        INSERT INTO account_health_metrics 
                        (client_id, health_score, roas_score, waste_score, cvr_score,
                         waste_ratio, wasted_spend, current_roas, current_acos, cvr,
                         total_spend, total_sales, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (client_id) DO UPDATE SET
                            health_score = EXCLUDED.health_score,
                            roas_score = EXCLUDED.roas_score,
                            waste_score = EXCLUDED.waste_score,
                            cvr_score = EXCLUDED.cvr_score,
                            waste_ratio = EXCLUDED.waste_ratio,
                            wasted_spend = EXCLUDED.wasted_spend,
                            current_roas = EXCLUDED.current_roas,
                            current_acos = EXCLUDED.current_acos,
                            cvr = EXCLUDED.cvr,
                            total_spend = EXCLUDED.total_spend,
                            total_sales = EXCLUDED.total_sales,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        client_id,
                        float(metrics.get('health_score', 0)),
                        float(metrics.get('roas_score', 0)),
                        float(metrics.get('efficiency_score', metrics.get('waste_score', 0))),
                        float(metrics.get('cvr_score', 0)),
                        float(metrics.get('waste_ratio', 0)),
                        float(metrics.get('wasted_spend', 0)),
                        float(metrics.get('current_roas', 0)),
                        float(metrics.get('current_acos', 0)),
                        float(metrics.get('cvr', 0)),
                        float(metrics.get('total_spend', 0)),
                        float(metrics.get('total_sales', 0))
                    ))
            return True
        except Exception as e:
            print(f"Failed to save account health: {e}")
            return False
    
    def get_account_health(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get account health metrics from database."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM account_health_metrics WHERE client_id = %s",
                    (client_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    @retry_on_connection_error()
    def get_available_dates(self, client_id: str) -> List[str]:
        """Get list of unique action dates for a client with caching."""
        cache_key = f'dates_{client_id}'
        cached = _query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT DISTINCT start_date
                    FROM target_stats 
                    WHERE client_id = %s 
                    ORDER BY start_date DESC
                """, (client_id,))
                result = [str(row['start_date']) for row in cursor.fetchall()]
        
        _query_cache.set(cache_key, result)
        return result
    
    @retry_on_connection_error()
    def get_action_impact(self, client_id: str, before_days: int = 14, after_days: int = 14) -> pd.DataFrame:
        """
        Calculate impact using rule-based expected outcomes with caching.
        
        Uses multi-horizon measurement:
        - before_days: Fixed at 14 days (baseline period)
        - after_days: 14, 30, or 60 days (measurement horizon)
        """
        cache_key = f'impact_{client_id}_{before_days}_{after_days}'
        cached = _query_cache.get(cache_key)
        if cached is not None:
            return cached

        # Calculate intervals based on before_days and after_days
        after_minus_1 = after_days - 1
        
        # Original LATERAL join query - accurate but slower
        # Indexes added for performance: idx_ts_client_campaign, idx_ts_client_target, idx_ts_client_cst
        query = """
            WITH aggregated_actions AS (
                -- Group daily actions into weekly buckets
                SELECT 
                    LOWER(target_text) as target_lower,
                    CASE 
                        WHEN LOWER(target_text) LIKE 'asin=%%' THEN 
                            LOWER(REPLACE(REPLACE(target_text, 'asin="', ''), '"', ''))
                        ELSE LOWER(target_text)
                    END as normalized_target_lower,
                    LOWER(campaign_name) as campaign_lower,
                    LOWER(ad_group_name) as ad_group_lower,
                    target_text, campaign_name, ad_group_name, match_type, action_type,
                    DATE(action_date - (MOD((EXTRACT(DOW FROM action_date)::int - 2 + 7), 7)) * INTERVAL '1 day') as week_start,
                    MAX(action_date) as action_date,
                    (ARRAY_AGG(old_value ORDER BY action_date ASC))[1] as old_value,
                    (ARRAY_AGG(new_value ORDER BY action_date DESC))[1] as new_value,
                    STRING_AGG(DISTINCT reason, '; ') as reason,
                    MAX(action_date) - INTERVAL '%(before_days)s days' as before_start,
                    MAX(action_date) - INTERVAL '1 day' as before_end,
                    MAX(action_date) as after_start,
                    MAX(action_date) + INTERVAL '%(after_minus_1)s days' as after_end
                FROM actions_log
                WHERE client_id = %(client_id)s
                  AND LOWER(action_type) NOT IN ('hold', 'monitor', 'flagged')
                GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            ),
            latest_data AS (
                SELECT MAX(start_date) as latest_date FROM target_stats WHERE client_id = %(client_id)s
            )
            SELECT 
                a.action_date, 
                a.action_type, 
                a.target_text, 
                a.campaign_name,
                a.ad_group_name,
                a.match_type,
                a.old_value, 
                a.new_value, 
                a.reason,
                a.before_start as before_date,
                a.before_end as before_end_date,
                a.after_start as after_date,
                LEAST(a.after_end, ld.latest_date) as after_end_date,
                -- Count actual data days in windows
                (SELECT COUNT(DISTINCT start_date) FROM target_stats 
                 WHERE client_id = %(client_id)s 
                   AND start_date >= a.before_start AND start_date <= a.before_end) as actual_before_days,
                (SELECT COUNT(DISTINCT start_date) FROM target_stats 
                 WHERE client_id = %(client_id)s 
                   AND start_date >= a.after_start AND start_date <= LEAST(a.after_end, ld.latest_date)) as actual_after_days,
                -- BEFORE stats via LATERAL (per-action window)
                COALESCE(bs.spend, bcs.spend, bc.spend, 0) as before_spend,
                COALESCE(bs.sales, bcs.sales, bc.sales, 0) as before_sales,
                COALESCE(bs.clicks, bcs.clicks, bc.clicks, 0) as before_clicks,
                COALESCE(bs.impressions, bcs.impressions, bc.impressions, 0) as before_impressions,
                -- AFTER stats via LATERAL (per-action window)
                COALESCE(afs.spend, afcs.spend, ac.spend, 0) as observed_after_spend,
                COALESCE(afs.sales, afcs.sales, ac.sales, 0) as observed_after_sales,
                COALESCE(afs.clicks, afcs.clicks, ac.clicks, 0) as after_clicks,
                COALESCE(afs.impressions, afcs.impressions, ac.impressions, 0) as after_impressions,
                CASE 
                    WHEN bs.spend IS NOT NULL THEN 'target'
                    WHEN bcs.spend IS NOT NULL THEN 'cst'
                    ELSE 'campaign' 
                END as match_level,
                r30.rolling_spc as rolling_30d_spc
            FROM aggregated_actions a
            CROSS JOIN latest_data ld
            -- BEFORE: target_text match (for BID_CHANGE)
            LEFT JOIN LATERAL (
                SELECT SUM(spend) as spend, SUM(sales) as sales, SUM(clicks) as clicks, SUM(impressions) as impressions
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.target_text) = a.target_lower
                  AND LOWER(t.campaign_name) = a.campaign_lower
                  AND t.start_date >= a.before_start AND t.start_date <= a.before_end
            ) bs ON TRUE
            -- BEFORE: CST match (for NEGATIVE/HARVEST)
            LEFT JOIN LATERAL (
                SELECT SUM(spend) as spend, SUM(sales) as sales, SUM(clicks) as clicks, SUM(impressions) as impressions
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.customer_search_term) = a.normalized_target_lower
                  AND t.start_date >= a.before_start AND t.start_date <= a.before_end
            ) bcs ON a.action_type IN ('NEGATIVE', 'NEGATIVE_ADD', 'HARVEST')
            -- BEFORE: Campaign fallback
            LEFT JOIN LATERAL (
                SELECT SUM(spend) as spend, SUM(sales) as sales, SUM(clicks) as clicks, SUM(impressions) as impressions
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.campaign_name) = a.campaign_lower
                  AND t.start_date >= a.before_start AND t.start_date <= a.before_end
            ) bc ON bs.spend IS NULL AND bcs.spend IS NULL
            -- AFTER: target_text match (for BID_CHANGE)
            LEFT JOIN LATERAL (
                SELECT SUM(spend) as spend, SUM(sales) as sales, SUM(clicks) as clicks, SUM(impressions) as impressions
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.target_text) = a.target_lower
                  AND LOWER(t.campaign_name) = a.campaign_lower
                  AND t.start_date >= a.after_start AND t.start_date <= LEAST(a.after_end, ld.latest_date)
            ) afs ON TRUE
            -- AFTER: CST match (for NEGATIVE/HARVEST)
            LEFT JOIN LATERAL (
                SELECT SUM(spend) as spend, SUM(sales) as sales, SUM(clicks) as clicks, SUM(impressions) as impressions
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.customer_search_term) = a.normalized_target_lower
                  AND t.start_date >= a.after_start AND t.start_date <= LEAST(a.after_end, ld.latest_date)
            ) afcs ON a.action_type IN ('NEGATIVE', 'NEGATIVE_ADD', 'HARVEST')
            -- AFTER: Campaign fallback
            LEFT JOIN LATERAL (
                SELECT SUM(spend) as spend, SUM(sales) as sales, SUM(clicks) as clicks, SUM(impressions) as impressions
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.campaign_name) = a.campaign_lower
                  AND t.start_date >= a.after_start AND t.start_date <= LEAST(a.after_end, ld.latest_date)
            ) ac ON afs.spend IS NULL AND afcs.spend IS NULL
            -- Rolling 30d stats for baseline
            LEFT JOIN LATERAL (
                SELECT CASE WHEN SUM(clicks) > 0 THEN SUM(sales) / SUM(clicks) ELSE NULL END as rolling_spc
                FROM target_stats t
                WHERE t.client_id = %(client_id)s
                  AND LOWER(t.target_text) = a.target_lower
                  AND LOWER(t.campaign_name) = a.campaign_lower
                  AND t.start_date >= ld.latest_date - INTERVAL '30 days'
            ) r30 ON TRUE
            ORDER BY a.action_date DESC
        """
        
        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params={
                'client_id': client_id,
                'after_minus_1': after_minus_1,
                'before_days': before_days
            })
        
        if df.empty:
            _query_cache.set(cache_key, df)
            return df
            
        # ==========================================
        # NORMALIZATION: Symmetrical Comparison
        # ==========================================
        # If before window has 4 weeks of data and after only has 2 weeks,
        # we scale the 'after' up to be comparable (apples-to-apples).
        for idx in df.index:
            b_days = float(df.at[idx, 'actual_before_days'] or 0)
            a_days = float(df.at[idx, 'actual_after_days'] or 0)
            
            # Normalization factor (to make 'before' comparable to 'after')
            # If after_days is 3 and before_days is 7, multiply before by 3/7
            if b_days > 0 and a_days > 0 and b_days != a_days:
                ratio = a_days / b_days
                df.at[idx, 'before_spend'] *= ratio
                df.at[idx, 'before_sales'] *= ratio
                df.at[idx, 'before_clicks'] *= ratio
        
        # Normalize action types
        df['action_type'] = df['action_type'].str.upper()
        
        # ==========================================
        # LAYER 1: ACCOUNT BASELINE CALCULATION
        # ==========================================
        # Calculate account-wide spend and ROAS changes to normalize validation
        total_before_spend = df['before_spend'].sum()
        total_after_spend = df['observed_after_spend'].sum()
        total_before_sales = df['before_sales'].sum()
        total_after_sales = df['observed_after_sales'].sum()
        
        # Baseline metrics (stored for later use)
        baseline_spend_change = (total_after_spend / total_before_spend - 1) if total_before_spend > 0 else 0
        baseline_roas_before = total_before_sales / total_before_spend if total_before_spend > 0 else 0
        baseline_roas_after = total_after_sales / total_after_spend if total_after_spend > 0 else 0
        baseline_roas_change = (baseline_roas_after / baseline_roas_before - 1) if baseline_roas_before > 0 else 0
        
        # Store in dataframe for downstream use
        df['_baseline_spend_change'] = baseline_spend_change
        df['_baseline_roas_change'] = baseline_roas_change
        
        # Initialize columns
        df['after_spend'] = 0.0
        df['after_sales'] = 0.0
        df['delta_spend'] = 0.0
        df['delta_sales'] = 0.0
        df['impact_score'] = 0.0
        df['attribution'] = 'direct_causation'
        df['validation_status'] = ''
        df['spend_avoided'] = 0.0
        
        # RULE 1: NEGATIVE → After = $0, impact = cost saved
        neg_mask = df['action_type'].isin(['NEGATIVE', 'NEGATIVE_ADD'])
        df.loc[neg_mask, 'after_spend'] = 0.0
        
        # REST OF THE VALIDATION LOGIC REMAINS SAME...
        # (Skipping to deduplication removal)
        # ...
        df.loc[neg_mask, 'after_sales'] = 0.0
        df.loc[neg_mask, 'delta_spend'] = -df.loc[neg_mask, 'before_spend']
        df.loc[neg_mask, 'delta_sales'] = -df.loc[neg_mask, 'before_sales']
        df.loc[neg_mask, 'impact_score'] = df.loc[neg_mask, 'before_spend']  # Positive = cost saved
        df.loc[neg_mask, 'attribution'] = 'cost_avoidance'
        
        # Check if negative was actually implemented
        # Include all levels where we have actual search term data (target, cst, cst_account)
        # Only 'campaign' level means no search term match
        has_target_match = df['match_level'].isin(['target', 'cst', 'cst_account'])
        
        # Clear case: Target found in after window with spend = keyword still active
        neg_not_impl = neg_mask & has_target_match & (df['observed_after_spend'] > 0)
        df.loc[neg_not_impl, 'validation_status'] = '⚠️ NOT IMPLEMENTED'
        
        # NORMALIZED VALIDATION for NEG
        # Target is "confirmed blocked" only if spend dropped significantly MORE than baseline
        # threshold: at least 50% below baseline change, or 100% drop (to $0)
        target_spend_change = (df['observed_after_spend'] / df['before_spend'] - 1).fillna(-1)
        threshold = min(baseline_spend_change - 0.5, -0.95)  # At least 50% worse than baseline
        
        # Clear case: Target found with $0 spend = definitely blocked
        neg_impl_zero = neg_mask & has_target_match & (df['observed_after_spend'] == 0)
        df.loc[neg_impl_zero, 'validation_status'] = '✓ Confirmed blocked'
        
        # Normalized case: Significant drop beyond baseline
        neg_impl_normalized = neg_mask & has_target_match & (df['observed_after_spend'] > 0) & (target_spend_change < threshold)
        df.loc[neg_impl_normalized, 'validation_status'] = '✓ Normalized match'
        
        # Unclear: Target not found in after window (could be blocked or just no data)
        neg_unknown = neg_mask & ~has_target_match
        df.loc[neg_unknown, 'validation_status'] = '◐ Unverified (no target data)'
        
        # Special: Preventative negatives
        prev_mask = neg_mask & (df['before_spend'] == 0)
        df.loc[prev_mask, 'attribution'] = 'preventative'
        df.loc[prev_mask, 'impact_score'] = 0
        df.loc[prev_mask, 'validation_status'] = 'Preventative - no spend to save'
        
        # Special: Isolation negatives
        reason_lower = df['reason'].fillna('').str.lower()
        iso_mask = neg_mask & (reason_lower.str.contains('isolation|harvest'))
        df.loc[iso_mask, 'attribution'] = 'isolation_negative'
        df.loc[iso_mask, 'impact_score'] = 0
        df.loc[iso_mask, 'validation_status'] = 'Part of harvest consolidation'
        
        # RULE 2: HARVEST → Tiered migration validation
        harv_mask = df['action_type'] == 'HARVEST'
        df.loc[harv_mask, 'after_spend'] = 0.0
        df.loc[harv_mask, 'after_sales'] = 0.0
        df.loc[harv_mask, 'delta_spend'] = 0.0
        df.loc[harv_mask, 'delta_sales'] = df.loc[harv_mask, 'before_sales'] * 0.10
        df.loc[harv_mask, 'impact_score'] = df.loc[harv_mask, 'delta_sales']
        df.loc[harv_mask, 'attribution'] = 'harvest'
        
        # Tiered harvest validation based on source drop % and exact match growth
        min_spend = HARVEST_VALIDATION_CONFIG['min_source_before_spend']
        near_complete = HARVEST_VALIDATION_CONFIG['near_complete_threshold']
        strong_thresh = HARVEST_VALIDATION_CONFIG['strong_migration_threshold']
        partial_thresh = HARVEST_VALIDATION_CONFIG['partial_migration_threshold']
        exact_growth_req = HARVEST_VALIDATION_CONFIG['exact_growth_required_for_partial']
        
        for idx in df[harv_mask].index:
            source_before = df.at[idx, 'before_spend']
            source_after = df.at[idx, 'observed_after_spend']
            target_text = str(df.at[idx, 'target_text']).lower().strip()
            
            # Check minimum source spend threshold
            if source_before < min_spend:
                df.at[idx, 'validation_status'] = '◐ Unverified (low baseline)'
                continue
            
            # Calculate source drop percentage
            source_drop_pct = (source_before - source_after) / source_before
            
            # Look up exact match spend for this term (from same data)
            exact_matches = df[
                (df['target_text'].str.lower().str.strip() == target_text) &
                (df['match_type'].str.lower() == 'exact')
            ]
            exact_after_spend = exact_matches['observed_after_spend'].sum() if len(exact_matches) > 0 else 0
            exact_before_spend = exact_matches['before_spend'].sum() if len(exact_matches) > 0 else 0
            exact_growth = exact_after_spend - exact_before_spend
            exact_growth_pct = exact_growth / max(exact_before_spend, 1) if exact_before_spend > 0 else (1.0 if exact_after_spend > 0 else 0)
            
            # Store metrics for transparency
            df.at[idx, 'source_drop_pct'] = round(source_drop_pct * 100, 1)
            df.at[idx, 'exact_growth'] = round(exact_growth, 2)
            
            # TIER 1: Complete block ($0)
            if source_after == 0:
                df.at[idx, 'validation_status'] = '✓ Harvested (complete)'
                continue
            
            # TIER 2: Near-complete (≥90% drop)
            if source_drop_pct >= near_complete:
                df.at[idx, 'validation_status'] = '✓ Harvested (90%+ blocked)'
                continue
            
            # TIER 3: Strong migration (≥75% drop)
            if source_drop_pct >= strong_thresh:
                df.at[idx, 'validation_status'] = '✓ Harvested (migrated)'
                continue
            
            # TIER 4: Partial migration (≥50% drop)
            if source_drop_pct >= partial_thresh:
                df.at[idx, 'validation_status'] = '✓ Harvested (partial)'
                continue
            
            # TIER 5: Incomplete
            if source_drop_pct > 0:
                df.at[idx, 'validation_status'] = f'⚠️ Migration {source_drop_pct*100:.0f}%'
            else:
                df.at[idx, 'validation_status'] = '⚠️ Source still active'
        
        # RULE 3: BID_CHANGE + VISIBILITY_BOOST → Incremental Revenue = before_spend * (roas_after - roas_before)
        # VISIBILITY_BOOST is treated same as BID_UP for impact calculation
        bid_mask = df['action_type'].str.contains('BID|VISIBILITY_BOOST', na=False, regex=True)
        df.loc[bid_mask, 'after_spend'] = df.loc[bid_mask, 'observed_after_spend']
        df.loc[bid_mask, 'after_sales'] = df.loc[bid_mask, 'observed_after_sales']
        df.loc[bid_mask, 'delta_spend'] = df.loc[bid_mask, 'observed_after_spend'] - df.loc[bid_mask, 'before_spend']
        df.loc[bid_mask, 'delta_sales'] = df.loc[bid_mask, 'observed_after_sales'] - df.loc[bid_mask, 'before_sales']
        
        # ==========================================
        # LAYER 2: DIRECTIONAL CPC VALIDATION
        # ==========================================
        # Parse old_value/new_value to determine if BID_UP or BID_DOWN
        def parse_bid_direction(row):
            old_str = str(row.get('old_value', '')).strip()
            new_str = str(row.get('new_value', '')).strip()
            
            # If old_value is missing, can't determine direction
            if not old_str or old_str == 'None' or old_str == 'nan':
                return 'UNKNOWN'
            
            try:
                old_val = float(old_str.replace('$', '').replace(',', ''))
                new_val = float(new_str.replace('$', '').replace(',', ''))
                return 'DOWN' if new_val < old_val else 'UP'
            except:
                return 'UNKNOWN'
        
        # Calculate individual ROAS changes for each action
        for idx in df[bid_mask].index:
            b_spend = df.at[idx, 'before_spend']
            b_sales = df.at[idx, 'before_sales']
            a_spend = df.at[idx, 'observed_after_spend']
            a_sales = df.at[idx, 'observed_after_sales']
            b_clicks = df.at[idx, 'before_clicks'] if 'before_clicks' in df.columns else 0
            a_clicks = df.at[idx, 'after_clicks'] if 'after_clicks' in df.columns else 0
            
            # Extract impressions for new validation layers
            b_impressions = df.at[idx, 'before_impressions'] if 'before_impressions' in df.columns else 0
            a_impressions = df.at[idx, 'after_impressions'] if 'after_impressions' in df.columns else 0
            
            # Calculate actual CPCs
            before_cpc = b_spend / b_clicks if b_clicks > 0 else 0
            after_cpc = a_spend / a_clicks if a_clicks > 0 else 0
            
            # Calculate impressions change
            imp_change_pct = (a_impressions - b_impressions) / b_impressions if b_impressions > 0 else 0
            
            # Get suggested bid from new_value
            new_value_str = str(df.at[idx, 'new_value']).strip()
            try:
                suggested_bid = float(new_value_str.replace('$', '').replace(',', ''))
            except:
                suggested_bid = 0
            
            r_before = b_sales / b_spend if b_spend > 0 else 0
            r_after = a_sales / a_spend if a_spend > 0 else 0
            
            # Get bid direction - use before_cpc as fallback when old_value is missing
            bid_direction = parse_bid_direction(df.loc[idx])
            if bid_direction == 'UNKNOWN' and before_cpc > 0 and suggested_bid > 0:
                # Fallback: compare suggested bid to before_cpc
                if suggested_bid > before_cpc * 1.05:  # >5% higher than before_cpc = UP
                    bid_direction = 'UP'
                elif suggested_bid < before_cpc * 0.95:  # <5% lower than before_cpc = DOWN
                    bid_direction = 'DOWN'
            
            # Calculate CPC change percentage
            cpc_change_pct = (after_cpc - before_cpc) / before_cpc if before_cpc > 0 else 0
            
            # LAYER 1: CPC Match (tightened to 15%)
            cpc_validated = False
            cpc_tolerance = BID_VALIDATION_CONFIG['cpc_match_threshold']  # 0.15
            
            if suggested_bid > 0 and after_cpc > 0:
                cpc_match_ratio = after_cpc / suggested_bid
                if 1 - cpc_tolerance <= cpc_match_ratio <= 1 + cpc_tolerance:
                    cpc_validated = True
            
            # LAYER 2: CPC Directional (>5% CPC change in expected direction)
            cpc_dir_threshold = BID_VALIDATION_CONFIG['cpc_directional_threshold']  # 0.05
            directional_match = None
            
            if before_cpc > 0 and after_cpc > 0:
                if bid_direction == 'DOWN' and cpc_change_pct < -cpc_dir_threshold:
                    directional_match = True
                elif bid_direction == 'UP' and cpc_change_pct > cpc_dir_threshold:
                    directional_match = True
                elif bid_direction == 'UNKNOWN':
                    directional_match = None
                else:
                    directional_match = False
            
            # LAYER 3: Volume Validated (NEW) - Impressions changed significantly
            volume_validated = False
            min_imp = BID_VALIDATION_CONFIG['min_impressions_before']  # 50
            imp_up_threshold = BID_VALIDATION_CONFIG['impressions_increase_threshold']  # 0.20
            imp_down_threshold = BID_VALIDATION_CONFIG['impressions_decrease_threshold']  # 0.15
            
            if b_impressions >= min_imp:
                if bid_direction == 'UP' and imp_change_pct >= imp_up_threshold:
                    volume_validated = True
                elif bid_direction == 'DOWN' and imp_change_pct <= -imp_down_threshold:
                    volume_validated = True
            
            # LAYER 4: Directional + Volume (NEW) - Weak signals combined
            combined_validated = False
            combined_imp_threshold = BID_VALIDATION_CONFIG['combined_impressions_threshold']  # 0.10
            
            if b_impressions >= min_imp and not cpc_validated and directional_match is not True and not volume_validated:
                if bid_direction == 'UP':
                    # Any CPC increase + moderate impressions increase
                    if cpc_change_pct > 0 and imp_change_pct > combined_imp_threshold:
                        combined_validated = True
                elif bid_direction == 'DOWN':
                    # Any CPC decrease + moderate impressions decrease
                    if cpc_change_pct < 0 and imp_change_pct < -combined_imp_threshold:
                        combined_validated = True
            
            # LAYER 5: Normalized winner (beat account baseline)
            target_roas_change = (r_after / r_before - 1) if r_before > 0 else 0
            beat_baseline = target_roas_change > baseline_roas_change
            
            # Calculate impact based on validation status
            delta_sales = a_sales - b_sales
            
            # Require minimum clicks for reliable ROAS-based impact calculation
            min_clicks_for_roas = 5
            has_enough_data = (b_clicks >= min_clicks_for_roas and a_clicks >= min_clicks_for_roas)
            
            # Include volume_validated and combined_validated in impact calculation
            is_validated = (cpc_validated or directional_match is True or volume_validated or combined_validated)
            
            if is_validated and has_enough_data:
                # Validated with sufficient data: Use ROAS-based impact, capped
                roas_impact = b_spend * (r_after - r_before)
                # Cap at 2x the actual delta_sales to prevent inflation
                max_impact = abs(delta_sales) * 2 if delta_sales != 0 else abs(roas_impact)
                impact_score = max(min(roas_impact, max_impact), -max_impact) if roas_impact != 0 else delta_sales
                df.at[idx, 'impact_score'] = impact_score
            else:
                # Not validated OR insufficient data: Use actual delta_sales (conservative)
                df.at[idx, 'impact_score'] = delta_sales
            
            # Set validation status based on layers (order matters - first match wins)
            # FIRST: Handle zero after-spend based on action intent
            if a_spend == 0:
                if b_spend > 0:
                    # Had spend before, now $0 - check if this was intended
                    if bid_direction == 'DOWN':
                        # BID_DOWN with $0 after = SUCCESS (spend eliminated)
                        df.at[idx, 'validation_status'] = '✓ Spend Eliminated'
                        df.at[idx, 'spend_avoided'] = b_spend
                        continue  # Skip rest of validation
                    else:
                        # BID_UP with $0 after = likely not implemented or target died
                        df.at[idx, 'validation_status'] = '◐ No after data'
                        continue
                else:
                    # No spend before AND after = dormant target
                    df.at[idx, 'validation_status'] = '◐ Dormant target'
                    continue
            elif a_clicks == 0:
                # Has spend but no clicks = data anomaly, mark but continue validation
                df.at[idx, 'validation_status'] = '◐ Low click volume'
                # Don't continue - let other validation layers try
            
            if cpc_validated and beat_baseline:
                df.at[idx, 'validation_status'] = '✓ CPC Match + Baseline'
            elif cpc_validated:
                df.at[idx, 'validation_status'] = '✓ CPC Validated'
            elif directional_match is True and beat_baseline:
                df.at[idx, 'validation_status'] = '✓ Directional + Baseline'
            elif directional_match is True:
                df.at[idx, 'validation_status'] = '✓ Directional match'
            elif volume_validated:
                df.at[idx, 'validation_status'] = '✓ Volume Validated'
            elif combined_validated:
                df.at[idx, 'validation_status'] = '✓ Directional + Volume'
            elif beat_baseline:
                df.at[idx, 'validation_status'] = '◐ Beat baseline only'
            else:
                df.at[idx, 'validation_status'] = '⚠️ Not validated'
        
        # RULE 4: PAUSE → Incremental loss = -before_sales (minus what you saved in spend)
        pause_mask = df['action_type'].str.contains('PAUSE', na=False)
        df.loc[pause_mask, 'after_spend'] = 0.0
        df.loc[pause_mask, 'after_sales'] = 0.0
        df.loc[pause_mask, 'delta_spend'] = -df.loc[pause_mask, 'before_spend']
        df.loc[pause_mask, 'delta_sales'] = -df.loc[pause_mask, 'before_sales']
        # For pause, impact is net incremental revenue (sales lost - spend saved)
        df.loc[pause_mask, 'impact_score'] = df.loc[pause_mask, 'delta_sales'] - df.loc[pause_mask, 'delta_spend']
        df.loc[pause_mask, 'attribution'] = 'structural_change'

        
        pause_not_impl = pause_mask & (df['observed_after_spend'] > 0)
        df.loc[pause_not_impl, 'validation_status'] = '⚠️ Still has spend'
        pause_impl = pause_mask & (df['observed_after_spend'] == 0)
        df.loc[pause_impl, 'validation_status'] = '✓ Confirmed paused'
        
        # ==========================================
        # CREDIT SYSTEM: Only count confirmed implementations
        # Zero out impact_score for actions that weren't implemented
        # Actions are still shown in table, but don't count toward totals
        # ==========================================
        not_implemented_statuses = [
            '⚠️ NOT IMPLEMENTED',
            '⚠️ Source still active',
            '⚠️ Still has spend',
            '◐ Unverified (no target data)'  # Can't confirm, don't credit
        ]
        not_impl_mask = df['validation_status'].isin(not_implemented_statuses)
        
        # Determine winners based on ABSOLUTE net impact (Sales Δ - Spend Δ > 0)
        df['is_winner'] = (df['delta_sales'] - df['delta_spend']) > 0
        
        # Store original impact for display, then zero out for totals
        df['potential_impact'] = df['impact_score'].copy()  # What it WOULD have been
        df.loc[not_impl_mask, 'impact_score'] = 0  # Zero for not implemented
        
        # ==========================================
        # DEDUPLICATION: Prevent campaign-level overcounting
        # This is CRITICAL: If 10 targets in one campaign all fall back to 
        # the same campaign-level impact, we MUST only count that impact once.
        # ==========================================
        before_count = len(df)
        
        # Key for collapsing redundant campaign-level impact
        # We group by (campaign, action_type, stats_signature)
        df['_dedup_key'] = (
            df['campaign_name'].fillna('').str.lower() + '|' +
            df['action_type'].fillna('') + '|' +
            df['before_spend'].round(2).astype(str) + '|' +
            df['before_sales'].round(2).astype(str)
        )
        
        # Keep first (favors specific target records if they exist)
        # For weekly buckets, we also deduplicate within the bucket implicitly here.
        df = df.drop_duplicates(subset='_dedup_key', keep='first')
        df = df.drop(columns=['_dedup_key'])
        
        # ==========================================
        # ADD PER-ROW DECISION METRICS TO DATAFRAME
        # (Single source of truth - frontend displays, doesn't recalculate)
        # ==========================================
        import numpy as np
        
        # Calculate decision metrics for ALL action types (not just BID)
        # This ensures HARVEST/NEGATIVE actions also have CPC, decision_impact for display
        
        # SPC Baseline: 30D rolling with window fallback (per PRD 4.7.3)
        df['spc_window'] = (
            df['before_sales'] / 
            df['before_clicks'].replace(0, np.nan)
        )
        df['spc_before'] = df['rolling_30d_spc'].fillna(df['spc_window'])
        
        # ==========================================
        # LOW-SAMPLE SPC GUARDRAIL (Critical Fix)
        # ==========================================
        # Problem: Single-click conversions create inflated SPC (e.g., 62 AED/click)
        # which causes massive negative impacts when extrapolated.
        # Solution: Cap SPC for low-sample targets to reasonable max using median + 2*std
        MIN_CLICKS_FOR_RELIABLE_SPC = 5
        low_sample_mask = df['before_clicks'] < MIN_CLICKS_FOR_RELIABLE_SPC
        
        # Calculate reasonable SPC cap from higher-sample data
        reliable_spc = df.loc[~low_sample_mask, 'spc_before'].dropna()
        if len(reliable_spc) > 3:
            spc_cap = reliable_spc.median() + 2 * reliable_spc.std()
        else:
            # Fallback: Use simple max of 10 AED per click (reasonable for most products)
            spc_cap = 10.0
        
        # Apply cap to low-sample SPC values
        df.loc[low_sample_mask, 'spc_before'] = df.loc[low_sample_mask, 'spc_before'].clip(upper=spc_cap)
        
        # CPC calculations
        df['cpc_before'] = (
            df['before_spend'] / 
            df['before_clicks'].replace(0, np.nan)
        )
        df['cpc_after'] = (
            df['observed_after_spend'] / 
            df['after_clicks'].replace(0, np.nan)
        )
        
        # CPC Change %
        df['cpc_change_pct'] = (
            (df['cpc_after'] - df['cpc_before']) / 
            df['cpc_before']
        ).fillna(0) * 100
        
        # Decision Impact Formula (PRD 4.7.2)
        df['expected_clicks'] = (
            df['observed_after_spend'] / df['cpc_before']
        )
        df['expected_sales'] = (
            df['expected_clicks'] * df['spc_before']
        )
        df['decision_impact'] = (
            df['observed_after_sales'] - df['expected_sales']
        )
        
        # ==========================================
        # CRITICAL: Zero out impact for low-sample baselines
        # ==========================================
        # Targets with <5 clicks cannot provide reliable impact estimates
        # The SPC from 1-2 clicks is statistically meaningless
        # These should NOT contribute positive or negative to total impact
        import numpy as np
        insufficient_baseline_mask = df['before_clicks'] < MIN_CLICKS_FOR_RELIABLE_SPC
        df.loc[insufficient_baseline_mask, 'decision_impact'] = 0
        df['insufficient_baseline'] = insufficient_baseline_mask
        
        # Spend Avoided (for defensive actions) - preserve any existing value
        df['spend_avoided'] = df['spend_avoided'].fillna(
            (df['before_spend'] - df['observed_after_spend']).clip(lower=0)
        )
        
        # Market Downshift flag
        df['market_downshift'] = (
            df['cpc_after'] <= 0.75 * df['cpc_before']
        )
        
        _query_cache.set(cache_key, df)
        return df

    def _empty_summary(self) -> Dict[str, Any]:
        return {
            'total_actions': 0, 'roas_before': 0, 'roas_after': 0, 'roas_lift_pct': 0,
            'incremental_revenue': 0, 'p_value': 1.0, 'is_significant': False,
            'confidence_pct': 0, 'implementation_rate': 0, 'confirmed_impact': 0,
            'pending': 0, 'not_implemented': 0, 'win_rate': 0, 'winners': 0, 'losers': 0,
            'by_action_type': {},
            # Decision Impact fields
            'decision_impact': 0, 'spend_avoided': 0,
            'pct_good': 0, 'pct_neutral': 0, 'pct_bad': 0, 'market_downshift_count': 0
        }

    def _calculate_metrics_from_df(self, df: pd.DataFrame, window_days: int, label: str = "ALL") -> Dict[str, Any]:
        """Internal helper to calculate statistics from a filtered impact dataframe."""
        import scipy.stats as scipy_stats
        import numpy as np
        
        if df.empty:
            return self._empty_summary()
            
        total_actions = len(df)
        
        # ==========================================
        # 1. ROAS ANALYTICS + DECISION IMPACT (BID_CHANGE + VISIBILITY_BOOST)
        # ==========================================
        bid_mask = df['action_type'].str.contains('BID|VISIBILITY_BOOST', na=False, regex=True)
        bid_df = df[bid_mask].copy()
        
        # We also want to include targets that had 0 spend in one period to avoid 'missing' impact
        bid_df = bid_df[(bid_df['before_spend'] > 0) | (bid_df['observed_after_spend'] > 0)]
        
        if len(bid_df) > 5:
            total_before_spend = bid_df['before_spend'].sum()
            total_after_spend = bid_df['observed_after_spend'].sum()
            total_before_sales = bid_df['before_sales'].sum()
            total_after_sales = bid_df['observed_after_sales'].sum()
            
            roas_before = total_before_sales / total_before_spend if total_before_spend > 0 else 0
            roas_after = total_after_sales / total_after_spend if total_after_spend > 0 else 0
            roas_lift_pct = ((roas_after - roas_before) / roas_before * 100) if roas_before > 0 else 0
            incremental_revenue = total_before_spend * (roas_after - roas_before)
            
            # ==========================================
            # DECISION IMPACT CALCULATIONS
            # ==========================================
            # CPC: Use old_value (bid) if available, else derive from spend/clicks
            bid_df['cpc_before'] = pd.to_numeric(bid_df['old_value'], errors='coerce').fillna(
                bid_df['before_spend'] / bid_df['before_clicks'].replace(0, np.nan)
            )
            bid_df['cpc_after'] = bid_df['observed_after_spend'] / bid_df['after_clicks'].replace(0, np.nan)
            
            # Sales per Click - Use 30D ROLLING AVERAGE for stable baseline
            # Fallback to window-based SPC if rolling not available
            bid_df['spc_window'] = bid_df['before_sales'] / bid_df['before_clicks'].replace(0, np.nan)
            bid_df['spc_before'] = bid_df['rolling_30d_spc'].fillna(bid_df['spc_window'])
            
            # ==========================================
            # LOW-SAMPLE SPC GUARDRAIL (Critical Fix)
            # ==========================================
            # Problem: Single-click conversions create inflated SPC (e.g., 62 AED/click)
            # which causes massive negative impacts when extrapolated.
            # Solution: Cap SPC for low-sample targets to reasonable max using median + 2*std
            MIN_CLICKS_FOR_RELIABLE_SPC = 5
            low_sample_mask = bid_df['before_clicks'] < MIN_CLICKS_FOR_RELIABLE_SPC
            
            # Calculate reasonable SPC cap from higher-sample data
            reliable_spc = bid_df.loc[~low_sample_mask, 'spc_before'].dropna()
            if len(reliable_spc) > 3:
                spc_cap = reliable_spc.median() + 2 * reliable_spc.std()
            else:
                # Fallback: Use simple max of 10 AED per click (reasonable for most products)
                spc_cap = 10.0
            
            # Apply cap to low-sample SPC values
            bid_df.loc[low_sample_mask, 'spc_before'] = bid_df.loc[low_sample_mask, 'spc_before'].clip(upper=spc_cap)
            
            # Counterfactual: Expected sales if we kept old CPC
            # Expected_Clicks = After_Spend / Before_CPC
            # Expected_Sales = Expected_Clicks * SPC_Baseline (30D rolling or window fallback)
            bid_df['expected_clicks'] = bid_df['observed_after_spend'] / bid_df['cpc_before']
            bid_df['expected_sales'] = bid_df['expected_clicks'] * bid_df['spc_before']
            
            # Decision Impact = Actual - Counterfactual
            bid_df['decision_impact'] = bid_df['observed_after_sales'] - bid_df['expected_sales']
            
            # Spend Changes
            bid_df['spend_change'] = bid_df['observed_after_spend'] - bid_df['before_spend']
            bid_df['spend_avoided'] = (bid_df['before_spend'] - bid_df['observed_after_spend']).clip(lower=0)
            
            # CPC Change %
            bid_df['cpc_change_pct'] = (bid_df['cpc_after'] - bid_df['cpc_before']) / bid_df['cpc_before']
            
            # ==========================================
            # GUARDRAILS
            # ==========================================
            # Guardrail 1: Market Downshift (CPC dropped 25%+)
            bid_df['market_downshift'] = bid_df['cpc_after'] <= 0.75 * bid_df['cpc_before']
            
            # Guardrail 2: Insufficient Baseline (fewer than 3 clicks before)
            # NOTE: Extended from 0 to 3 to catch more low-sample edge cases
            bid_df['insufficient_baseline'] = bid_df['before_clicks'] < 3
            
            # ==========================================
            # OUTCOME CLASSIFICATION
            # ==========================================
            def classify_outcome(row):
                # Missing data -> Neutral
                if pd.isna(row.get('decision_impact')) or row.get('insufficient_baseline', False):
                    return 'Neutral'
                
                impact = row['decision_impact']
                spend_avoided = row.get('spend_avoided', 0)
                spend_before = row.get('before_spend', 0)
                cpc_change = row.get('cpc_change_pct', 0)
                action = str(row.get('action_type', '')).upper()
                sales_before = row.get('before_sales', 0)
                
                # Thresholds
                impact_small = abs(impact) < max(0.05 * sales_before, 25) if sales_before > 0 else abs(impact) < 25
                spend_avoided_low = spend_avoided < 0.10 * spend_before if spend_before > 0 else True
                
                # HOLD classification
                if 'HOLD' in action:
                    low_vol = abs(cpc_change) < 0.10 if pd.notna(cpc_change) else True
                    return 'Good' if low_vol else 'Neutral'
                
                # BID_DOWN / PAUSE classification
                if 'DOWN' in action or 'PAUSE' in action:
                    if spend_avoided_low and impact < 0:
                        return 'Bad'
                    return 'Good' if spend_avoided > 0 else 'Neutral'
                
                # BID_UP classification
                if 'UP' in action:
                    incr_sales = row['observed_after_sales'] - row['before_sales']
                    incr_spend = row['observed_after_spend'] - row['before_spend']
                    if incr_sales > incr_spend:
                        return 'Good'
                    elif impact < 0 and not row.get('market_downshift', False):
                        return 'Bad'
                    return 'Neutral'
                
                # Default logic
                if impact > 0:
                    return 'Good'
                elif impact_small or row.get('market_downshift', False):
                    return 'Neutral'
                else:
                    return 'Bad'
            
            bid_df['outcome'] = bid_df.apply(classify_outcome, axis=1)
            
            # ==========================================
            # CRITICAL: Zero out impact for low-sample baselines
            # ==========================================
            # Targets with <5 clicks cannot provide reliable impact estimates
            # Must zero BEFORE aggregation to prevent inflated totals
            MIN_CLICKS_FOR_RELIABLE = 5
            low_sample_mask = bid_df['before_clicks'] < MIN_CLICKS_FOR_RELIABLE
            bid_df.loc[low_sample_mask, 'decision_impact'] = 0
            
            # Aggregate Decision Impact metrics (now excludes low-sample)
            valid_impacts = bid_df['decision_impact'].dropna()
            total_decision_impact = valid_impacts.sum() if len(valid_impacts) > 0 else 0
            total_spend_avoided = bid_df['spend_avoided'].sum()
            market_downshift_count = int(bid_df['market_downshift'].sum())
            
            # Outcome percentages
            outcome_counts = bid_df['outcome'].value_counts()
            n_outcomes = len(bid_df)
            pct_good = (outcome_counts.get('Good', 0) / n_outcomes * 100) if n_outcomes > 0 else 0
            pct_neutral = (outcome_counts.get('Neutral', 0) / n_outcomes * 100) if n_outcomes > 0 else 0
            pct_bad = (outcome_counts.get('Bad', 0) / n_outcomes * 100) if n_outcomes > 0 else 0
            
            # Z-Test on AGGREGATE values only (not individual actions)
            n = len(bid_df)
            if n >= 10 and roas_before > 0:
                se_before = roas_before / np.sqrt(n)
                se_after = roas_after / np.sqrt(n)
                se_diff = np.sqrt(se_before**2 + se_after**2)
                z_stat = (roas_after - roas_before) / se_diff if se_diff > 0 else 0
                from scipy.stats import norm
                p_value = 1 - norm.cdf(z_stat) if z_stat > 0 else 1.0
            else:
                p_value = 1.0
                
            is_significant = (p_value <= 0.10) and (roas_lift_pct > 0)
            confidence_pct = (1 - p_value) * 100
        else:
            roas_before, roas_after, roas_lift_pct, incremental_revenue = 0, 0, 0, 0
            p_value, is_significant, confidence_pct = 1.0, False, 0
            total_decision_impact, total_spend_avoided = 0, 0
            pct_good, pct_neutral, pct_bad = 0, 0, 0
            market_downshift_count = 0
            
        # ==========================================
        # 2. IMPLEMENTATION & WIN RATE (ALL IN DF)
        # ==========================================
        status = df['validation_status'].fillna('')
        not_implemented = status.str.contains('NOT IMPLEMENTED|Source still active|Still has spend|Not validated', na=False, regex=True)
        confirmed = status.str.contains('✓|CPC Validated|CPC Match|Directional|Normalized|Confirmed|Volume', na=False, regex=True)
        pending = status.str.contains('Unverified|Preventative|Beat baseline only|No after data', na=False, regex=True)
        
        conf_count = int(confirmed.sum())
        impl_rate = (conf_count / total_actions * 100) if total_actions > 0 else 0
        winners = int(df['is_winner'].fillna(False).sum())
        win_rate = (winners / total_actions * 100) if total_actions > 0 else 0
        
        # ==========================================
        # 3. ACTION TYPE BREAKDOWN
        # ==========================================
        by_type = {}
        for action_type in df['action_type'].unique():
            type_data = df[df['action_type'] == action_type]
            by_type[action_type] = {
                'count': len(type_data),
                'net_sales': type_data['impact_score'].fillna(0).sum(),
                'net_spend': type_data['delta_spend'].fillna(0).sum()
            }
        
        return {
            'total_actions': total_actions,
            'roas_before': round(roas_before, 2),
            'roas_after': round(roas_after, 2),
            'roas_lift_pct': round(roas_lift_pct, 1),
            'incremental_revenue': round(incremental_revenue, 2),
            'p_value': round(p_value, 4),
            'is_significant': is_significant,
            'confidence_pct': round(confidence_pct, 1),
            'implementation_rate': round(impl_rate, 1),
            'confirmed_impact': conf_count,
            'pending': int(pending.sum()),
            'not_implemented': int(not_implemented.sum()),
            'win_rate': round(win_rate, 1),
            'winners': winners,
            'losers': total_actions - winners,
            'by_action_type': by_type,
            # NEW: Decision Impact metrics
            'decision_impact': round(total_decision_impact, 2),
            'spend_avoided': round(total_spend_avoided, 2),
            'pct_good': round(pct_good, 1),
            'pct_neutral': round(pct_neutral, 1),
            'pct_bad': round(pct_bad, 1),
            'market_downshift_count': market_downshift_count,
            'period_info': {
                'before_start': df['before_date'].min() if 'before_date' in df.columns else None,
                'before_end': df['before_end_date'].max() if 'before_end_date' in df.columns else None,
                'after_start': df['after_date'].min() if 'after_date' in df.columns else None,
                'after_end': df['after_end_date'].max() if 'after_end_date' in df.columns else None
            }
        }
    
    def get_impact_summary(self, client_id: str, before_days: int = 14, after_days: int = 14) -> Dict[str, Any]:
        """
        Aggregate statistical summary of impact across all actions.
        Returns both 'all' and 'validated' summaries for synchronized UI.
        
        Uses multi-horizon measurement:
        - before_days: Fixed at 14 days (baseline period)
        - after_days: 14, 30, or 60 days (measurement horizon)
        """
        impact_df = self.get_action_impact(client_id, before_days=before_days, after_days=after_days)
        if impact_df.empty:
            return {
                'all': self._empty_summary(),
                'validated': self._empty_summary()
            }
            
        # Summary 1: ALL ACTIONS
        summary_all = self._calculate_metrics_from_df(impact_df, after_days, label="ALL ACTIONS")
        
        # Summary 2: VALIDATED ACTIONS ONLY
        # Pattern matches: ✓, CPC Validated, CPC Match, Directional, Confirmed, Normalized
        validated_mask = impact_df['validation_status'].str.contains('✓|CPC Validated|CPC Match|Directional|Confirmed|Normalized|Volume', na=False, regex=True)
        validated_df = impact_df[validated_mask].copy()
        summary_validated = self._calculate_metrics_from_df(validated_df, after_days, label="VALIDATED ONLY")
        
        return {
            'all': summary_all,
            'validated': summary_validated,
            # Common metadata
            'period_info': summary_all.get('period_info')
        }



    # ==========================================
    # REFERENCE DATA STATUS
    # ==========================================
    
    def get_reference_data_status(self) -> Dict[str, Any]:
        """Check reference data freshness for sidebar badge."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as record_count,
                            MAX(updated_at) as latest_update
                        FROM target_stats
                    """)
                    row = cursor.fetchone()
                    
                    if not row or row['record_count'] == 0:
                        return {'exists': False, 'is_stale': True, 'days_ago': None, 'record_count': 0}
                    
                    latest = row['latest_update']
                    if latest:
                        days_ago = (datetime.now() - latest).days
                        is_stale = days_ago > 14
                    else:
                        days_ago = None
                        is_stale = True
                    
                    return {
                        'exists': True,
                        'is_stale': is_stale,
                        'days_ago': days_ago,
                        'record_count': row['record_count']
                    }
        except:
            return {'exists': False, 'is_stale': True, 'days_ago': None, 'record_count': 0}

    # ==========================================
    # ACCOUNT MANAGEMENT
    # ==========================================
    
    def update_account(self, account_id: str, account_name: str, account_type: str = None, metadata: dict = None) -> bool:
        """Update an existing account."""
        import json
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if account_type and metadata:
                        cursor.execute("""
                            UPDATE accounts SET 
                                account_name = %s, 
                                account_type = %s, 
                                metadata = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE account_id = %s
                        """, (account_name, account_type, json.dumps(metadata), account_id))
                    elif account_type:
                        cursor.execute("""
                            UPDATE accounts SET 
                                account_name = %s, 
                                account_type = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE account_id = %s
                        """, (account_name, account_type, account_id))
                    else:
                        cursor.execute("""
                            UPDATE accounts SET 
                                account_name = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE account_id = %s
                        """, (account_name, account_id))
                    return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to update account: {e}")
            return False
    
    def reassign_data(self, from_account: str, to_account: str, start_date: str, end_date: str) -> int:
        """Move data between accounts for a date range."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                total_updated = 0
                
                # Update target_stats
                cursor.execute("""
                    UPDATE target_stats SET client_id = %s
                    WHERE client_id = %s AND start_date BETWEEN %s AND %s
                """, (to_account, from_account, start_date, end_date))
                total_updated += cursor.rowcount
                
                # Update weekly_stats
                cursor.execute("""
                    UPDATE weekly_stats SET client_id = %s
                    WHERE client_id = %s AND start_date BETWEEN %s AND %s
                """, (to_account, from_account, start_date, end_date))
                total_updated += cursor.rowcount
                
                # Update actions_log
                cursor.execute("""
                    UPDATE actions_log SET client_id = %s
                    WHERE client_id = %s AND DATE(action_date) BETWEEN %s AND %s
                """, (to_account, from_account, start_date, end_date))
                total_updated += cursor.rowcount
                
                return total_updated
    
    def delete_account(self, account_id: str) -> bool:
        """Delete an account and all its data."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Delete related data first
                    cursor.execute("DELETE FROM target_stats WHERE client_id = %s", (account_id,))
                    cursor.execute("DELETE FROM weekly_stats WHERE client_id = %s", (account_id,))
                    cursor.execute("DELETE FROM actions_log WHERE client_id = %s", (account_id,))
                    cursor.execute("DELETE FROM category_mappings WHERE client_id = %s", (account_id,))
                    cursor.execute("DELETE FROM advertised_product_cache WHERE client_id = %s", (account_id,))
                    cursor.execute("DELETE FROM bulk_mappings WHERE client_id = %s", (account_id,))
                    cursor.execute("DELETE FROM account_health_metrics WHERE client_id = %s", (account_id,))
                    # Delete account
                    cursor.execute("DELETE FROM accounts WHERE account_id = %s", (account_id,))
                    return True
        except Exception as e:
            print(f"Failed to delete account: {e}")
            return False
