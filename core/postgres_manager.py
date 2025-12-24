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
        # CRITICAL: Auto campaign detection
        # ==========================================
        # For auto campaigns, use "Targeting" column (close-match, loose-match, etc.)
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
        # Re-determine target_col with full logic
        if 'Targeting' in df_copy.columns:
            target_col = 'Targeting'
        elif 'Customer Search Term' in df_copy.columns:
            target_col = 'Customer Search Term'
        elif 'Keyword Text' in df_copy.columns:
            target_col = 'Keyword Text'
        else:
            return 0
        
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
        
        total_saved = 0
        
        for week_start in weeks:
            if pd.isna(week_start):
                continue
            
            week_data = df_copy[df_copy['_week_start'] == week_start]
            if week_data.empty:
                continue
            
            grouped = week_data.groupby(['_camp_norm', '_ag_norm', '_target_norm']).agg(agg_cols).reset_index()
            
            week_start_str = week_start.isoformat() if isinstance(week_start, date) else str(week_start)[:10]
            
            # Prepare records for bulk insert
            records = []
            for _, row in grouped.iterrows():
                match_type_norm = str(row.get('Match Type', '')).lower().strip()
                records.append((
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
                    int(row.get('Impressions', 0) or 0)
                ))
            
            if records:
                with self._get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        execute_values(cursor, """
                            INSERT INTO target_stats 
                            (client_id, start_date, campaign_name, ad_group_name, target_text, 
                             match_type, spend, sales, orders, clicks, impressions)
                            VALUES %s
                            ON CONFLICT (client_id, start_date, campaign_name, ad_group_name, target_text) 
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

    def get_target_stats_df(self, client_id: str = 'default_client') -> pd.DataFrame:
        with self._get_connection() as conn:
            query = """
                SELECT 
                    start_date as "Date",
                    campaign_name as "Campaign Name",
                    ad_group_name as "Ad Group Name",
                    target_text as "Targeting",
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
                return dict(cursor.fetchone())

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
        for action in actions:
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
                action.get('winner_source_campaign'),  # NEW FIELD
                action.get('new_campaign_name'),  # NEW FIELD
                action.get('before_match_type'),  # NEW FIELD
                action.get('after_match_type')  # NEW FIELD
            ))
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                execute_values(cursor, """
                    INSERT INTO actions_log 
                    (action_date, client_id, batch_id, entity_name, action_type, old_value, new_value, 
                     reason, campaign_name, ad_group_name, target_text, match_type,
                     winner_source_campaign, new_campaign_name, before_match_type, after_match_type)
                    VALUES %s
                    ON CONFLICT (client_id, action_date, target_text, action_type, campaign_name) DO NOTHING
                """, data)
        return len(actions)

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

    # ==========================================
    # IMPACT DASHBOARD METHODS
    # ==========================================
    
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
    
    def get_action_impact(self, client_id: str, window_days: int = 7) -> pd.DataFrame:
        """
        Calculate impact using rule-based expected outcomes.
        
        CLEAN FIXED-WINDOW APPROACH:
        - AFTER window: Last 7 days of available data (e.g., Dec 10-16)
        - BEFORE window: Previous 7 days (e.g., Dec 3-9)
        - Eligible actions: Actions taken BEFORE the AFTER window starts
        
        Rules:
        - NEGATIVE → After = $0 (blocked)
        - HARVEST → Source After = $0, 10% lift
        - BID_CHANGE → Use observed (can't predict)
        - PAUSE → After = $0
        """
        # Calculate intervals based on window_days
        # W=7 -> after_start = latest-6, before_end = latest-7, before_start = latest-13
        w = window_days
        w_minus_1 = w - 1
        w2 = 2 * w - 1 # This is the 13 for W=7
        
        # Single batch query with dynamic fixed windows
        query = """
            WITH date_range AS (
                -- Get latest data date and calculate windows
                SELECT 
                    MAX(start_date) as latest_date,
                    MAX(start_date) - INTERVAL '%(w_minus_1)s days' as after_start,  
                    MAX(start_date) - INTERVAL '%(w)s days' as before_end,   
                    MAX(start_date) - INTERVAL '%(w2)s days' as before_start 
                FROM target_stats 
                WHERE client_id = %(client_id)s
            ),
            before_stats AS (
                -- Aggregate performance in BEFORE window (e.g., Dec 3-9)
                SELECT 
                    LOWER(target_text) as target_lower, 
                    LOWER(campaign_name) as campaign_lower,
                    SUM(spend) as spend, 
                    SUM(sales) as sales
                FROM target_stats t
                CROSS JOIN date_range dr
                WHERE t.client_id = %(client_id)s 
                  AND t.start_date >= dr.before_start 
                  AND t.start_date <= dr.before_end
                GROUP BY LOWER(target_text), LOWER(campaign_name)
            ),
            after_stats AS (
                -- Aggregate performance in AFTER window (e.g., Dec 10-16)
                SELECT 
                    LOWER(target_text) as target_lower, 
                    LOWER(campaign_name) as campaign_lower,
                    SUM(spend) as spend, 
                    SUM(sales) as sales
                FROM target_stats t
                CROSS JOIN date_range dr
                WHERE t.client_id = %(client_id)s 
                  AND t.start_date >= dr.after_start 
                  AND t.start_date <= dr.latest_date
                GROUP BY LOWER(target_text), LOWER(campaign_name)
            ),
            before_campaign AS (
                -- Fallback: Campaign-level BEFORE stats
                SELECT 
                    LOWER(campaign_name) as campaign_lower, 
                    SUM(spend) as spend, 
                    SUM(sales) as sales
                FROM target_stats t
                CROSS JOIN date_range dr
                WHERE t.client_id = %(client_id)s 
                  AND t.start_date >= dr.before_start 
                  AND t.start_date <= dr.before_end
                GROUP BY LOWER(campaign_name)
            ),
            after_campaign AS (
                -- Fallback: Campaign-level AFTER stats
                SELECT 
                    LOWER(campaign_name) as campaign_lower, 
                    SUM(spend) as spend, 
                    SUM(sales) as sales
                FROM target_stats t
                CROSS JOIN date_range dr
                WHERE t.client_id = %(client_id)s 
                  AND t.start_date >= dr.after_start 
                  AND t.start_date <= dr.latest_date
                GROUP BY LOWER(campaign_name)
            )
            SELECT 
                a.action_date, 
                a.action_type, 
                a.target_text, 
                a.campaign_name,
                a.ad_group_name, 
                a.old_value, 
                a.new_value, 
                a.reason,
                dr.before_start as before_date,
                dr.before_end as before_end_date,
                dr.after_start as after_date,
                dr.latest_date as after_end_date,
                COALESCE(bs.spend, bc.spend, 0) as before_spend,
                COALESCE(bs.sales, bc.sales, 0) as before_sales,
                COALESCE(afs.spend, ac.spend, 0) as observed_after_spend,
                COALESCE(afs.sales, ac.sales, 0) as observed_after_sales,
                CASE WHEN bs.spend IS NOT NULL THEN 'target' ELSE 'campaign' END as match_level
            FROM actions_log a
            CROSS JOIN date_range dr
            LEFT JOIN before_stats bs 
                ON LOWER(a.target_text) = bs.target_lower 
                AND LOWER(a.campaign_name) = bs.campaign_lower
            LEFT JOIN after_stats afs 
                ON LOWER(a.target_text) = afs.target_lower 
                AND LOWER(a.campaign_name) = afs.campaign_lower
            LEFT JOIN before_campaign bc 
                ON LOWER(a.campaign_name) = bc.campaign_lower
            LEFT JOIN after_campaign ac 
                ON LOWER(a.campaign_name) = ac.campaign_lower
            WHERE a.client_id = %(client_id)s 
              AND LOWER(a.action_type) NOT IN ('hold', 'monitor', 'flagged')
              -- Only include actions taken BEFORE the after window starts
              AND DATE(a.action_date) < dr.after_start
            ORDER BY a.action_date DESC
        """
        
        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params={
                'client_id': client_id,
                'w_minus_1': w_minus_1,
                'w': w,
                'w2': w2
            })
        
        if df.empty:
            return df
        
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
        
        # RULE 1: NEGATIVE → After = $0, impact = cost saved
        neg_mask = df['action_type'].isin(['NEGATIVE', 'NEGATIVE_ADD'])
        df.loc[neg_mask, 'after_spend'] = 0.0
        df.loc[neg_mask, 'after_sales'] = 0.0
        df.loc[neg_mask, 'delta_spend'] = -df.loc[neg_mask, 'before_spend']
        df.loc[neg_mask, 'delta_sales'] = -df.loc[neg_mask, 'before_sales']
        df.loc[neg_mask, 'impact_score'] = df.loc[neg_mask, 'before_spend']  # Positive = cost saved
        df.loc[neg_mask, 'attribution'] = 'cost_avoidance'
        
        # Check if negative was actually implemented
        # Only use observed_after_spend if we have target-level match (not campaign fallback)
        has_target_match = df['match_level'] == 'target'
        
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
        
        # RULE 2: HARVEST → Source After = $0, 10% lift assumption
        harv_mask = df['action_type'] == 'HARVEST'
        df.loc[harv_mask, 'after_spend'] = 0.0
        df.loc[harv_mask, 'after_sales'] = 0.0
        df.loc[harv_mask, 'delta_spend'] = 0.0
        df.loc[harv_mask, 'delta_sales'] = df.loc[harv_mask, 'before_sales'] * 0.10
        df.loc[harv_mask, 'impact_score'] = df.loc[harv_mask, 'delta_sales']
        df.loc[harv_mask, 'attribution'] = 'harvest'
        
        harv_not_impl = harv_mask & (df['observed_after_spend'] > 0)
        df.loc[harv_not_impl, 'validation_status'] = '⚠️ Source still active'
        harv_impl = harv_mask & (df['observed_after_spend'] == 0)
        df.loc[harv_impl, 'validation_status'] = '✓ Harvested to exact'
        
        # RULE 3: BID_CHANGE → Incremental Revenue = before_spend * (roas_after - roas_before)
        bid_mask = df['action_type'].str.contains('BID', na=False)
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
            
            r_before = b_sales / b_spend if b_spend > 0 else 0
            r_after = a_sales / a_spend if a_spend > 0 else 0
            
            # Impact = incremental revenue at baseline spend
            df.at[idx, 'impact_score'] = b_spend * (r_after - r_before)
            
            # LAYER 2: Check directional match (using spend change as CPC proxy)
            bid_direction = parse_bid_direction(df.loc[idx])
            spend_change = (a_spend / b_spend - 1) if b_spend > 0 else 0
            
            # Determine directional match only if we have direction info
            if bid_direction == 'DOWN' and spend_change < 0:
                directional_match = True
            elif bid_direction == 'UP' and spend_change > 0:
                directional_match = True
            elif bid_direction == 'UNKNOWN':
                directional_match = None  # Can't determine, fallback to Layer 3 only
            else:
                directional_match = False
            
            # LAYER 3: Normalized winner (beat account baseline)
            target_roas_change = (r_after / r_before - 1) if r_before > 0 else 0
            beat_baseline = target_roas_change > baseline_roas_change
            
            # Set validation status based on layers
            if a_spend == 0:
                df.at[idx, 'validation_status'] = '◐ No after data'
            elif directional_match is True and beat_baseline:
                df.at[idx, 'validation_status'] = '✓ Directional + Normalized'
            elif directional_match is True:
                df.at[idx, 'validation_status'] = '✓ Directional match'
            elif directional_match is None and beat_baseline:
                # Fallback: No direction info but beat baseline
                df.at[idx, 'validation_status'] = '✓ Beat baseline'
            elif beat_baseline:
                df.at[idx, 'validation_status'] = '◐ Beat baseline only'
            else:
                df.at[idx, 'validation_status'] = '⚠️ No validation match'
                df.at[idx, 'validation_status'] = '⚠️ No validation match'
        
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
        # This is decoupled from 'impact_score' which uses the incremental formula
        df['is_winner'] = (df['delta_sales'] - df['delta_spend']) > 0
        
        # Store original impact for display, then zero out for totals
        df['potential_impact'] = df['impact_score'].copy()  # What it WOULD have been
        df.loc[not_impl_mask, 'impact_score'] = 0  # Zero for not implemented

        
        # ==========================================
        # DEDUPLICATION: Prevent counting same impact multiple times
        # Multiple search terms can map to same ASIN/product in same campaign.
        # Group by (campaign, action_type, before_spend, before_sales) and keep first.
        # ==========================================
        before_count = len(df)
        
        # Create dedup key from campaign + action_type + stats (rounded to avoid float issues)
        df['_dedup_key'] = (
            df['campaign_name'].fillna('').str.lower() + '|' +
            df['action_type'].fillna('') + '|' +
            df['before_spend'].round(2).astype(str) + '|' +
            df['before_sales'].round(2).astype(str)
        )
        
        # Keep first occurrence of each dedup key (preserves one representative action)
        df = df.drop_duplicates(subset='_dedup_key', keep='first')
        df = df.drop(columns=['_dedup_key'])
        
        after_count = len(df)
        if before_count > after_count:
            print(f"Deduplicated: {before_count} → {after_count} actions (removed {before_count - after_count} duplicates)")
        
        return df
    
    def get_impact_summary(self, client_id: str, window_days: int = 7) -> Dict[str, Any]:
        """Get aggregated impact metrics including ROAS analytics for a client."""
        import numpy as np
        from scipy import stats as scipy_stats
        
        impact_df = self.get_action_impact(client_id, window_days=window_days)
        
        if impact_df.empty:
            return {
                'total_actions': 0,
                'roas_before': 0, 'roas_after': 0, 'roas_lift_pct': 0,
                'incremental_revenue': 0,
                'p_value': 1.0, 'is_significant': False, 'confidence_pct': 0,
                'implementation_rate': 0,
                'confirmed_impact': 0, 'pending': 0,
                'win_rate': 0, 'winners': 0, 'losers': 0,
                'by_action_type': {}
            }
        
        total_actions = len(impact_df)
        
        # ==========================================
        # ROAS ANALYTICS (for BID_CHANGE actions with valid data)
        # ==========================================
        bid_df = impact_df[impact_df['action_type'].str.contains('BID', na=False)].copy()
        bid_df = bid_df[(bid_df['before_spend'] > 0) & (bid_df['before_sales'] > 0)]
        bid_df = bid_df[bid_df['observed_after_spend'] > 0]
        
        if len(bid_df) > 5:
            # Aggregate ROAS (weighted by spend)
            total_before_spend = bid_df['before_spend'].sum()
            total_after_spend = bid_df['observed_after_spend'].sum()
            total_before_sales = bid_df['before_sales'].sum()
            total_after_sales = bid_df['observed_after_sales'].sum()
            
            roas_before = total_before_sales / total_before_spend if total_before_spend > 0 else 0
            roas_after = total_after_sales / total_after_spend if total_after_spend > 0 else 0
            roas_lift_pct = ((roas_after - roas_before) / roas_before * 100) if roas_before > 0 else 0
            
            # Incremental revenue = what you'd gain at same spend with new efficiency
            incremental_revenue = roas_lift_pct / 100 * total_before_spend * roas_before
            
            # Statistical significance (t-test on per-action ROAS % change)
            bid_df['roas_change_pct'] = (bid_df['observed_after_sales'] / bid_df['observed_after_spend'] - 
                                         bid_df['before_sales'] / bid_df['before_spend']) / \
                                        (bid_df['before_sales'] / bid_df['before_spend'])
            # Remove extreme outliers
            valid = bid_df[(bid_df['roas_change_pct'] > -0.95) & (bid_df['roas_change_pct'] < 10)]['roas_change_pct']
            
            if len(valid) > 2:
                t_stat, p_value = scipy_stats.ttest_1samp(valid, 0)
                p_value = float(p_value) if not np.isnan(p_value) else 1.0
            else:
                p_value = 1.0
            
            is_significant = p_value < 0.05
            confidence_pct = (1 - p_value) * 100
        else:
            roas_before, roas_after, roas_lift_pct = 0, 0, 0
            incremental_revenue = 0
            p_value, is_significant, confidence_pct = 1.0, False, 0
        
        # ==========================================
        # IMPLEMENTATION RATE
        # ==========================================
        # Updated patterns to recognize new layered validation statuses
        not_implemented = impact_df['validation_status'].str.contains('NOT IMPLEMENTED|Source still active|Still has spend|No validation match', na=False, regex=True)
        confirmed = impact_df['validation_status'].str.contains('✓|Directional|Normalized|Observed', na=False, regex=True)
        pending = impact_df['validation_status'].str.contains('Unverified|Preventative|Beat baseline only|No after data', na=False, regex=True)
        
        confirmed_count = confirmed.sum()
        not_impl_count = not_implemented.sum()
        pending_count = pending.sum()
        
        impl_rate = (confirmed_count / (confirmed_count + not_impl_count) * 100) if (confirmed_count + not_impl_count) > 0 else 0
        
        # ==========================================
        # WIN RATE & BY ACTION TYPE
        # ==========================================
        winners = impact_df['is_winner'].fillna(False).sum()
        losers = total_actions - winners
        win_rate = (winners / total_actions * 100) if total_actions > 0 else 0
        
        # ==========================================
        # REVENUE IMPACT (Sum of individual incremental impacts)
        # ==========================================
        # Individual actions now have impact_score = incremental revenue
        revenue_impact = impact_df['impact_score'].fillna(0).sum()
        
        # ==========================================
        # BY ACTION TYPE -> Incremental Revenue Breakdown
        # ==========================================
        by_action_type = {}
        for action_type in impact_df['action_type'].unique():
            type_data = impact_df[impact_df['action_type'] == action_type]
            by_action_type[action_type] = {
                'count': len(type_data),
                'net_sales': type_data['impact_score'].fillna(0).sum(), # This is incremental $
                'net_spend': type_data['delta_spend'].fillna(0).sum()
            }
        
        return {
            'total_actions': total_actions,
            # ROAS Analytics
            'roas_before': round(roas_before, 2),
            'roas_after': round(roas_after, 2),
            'roas_lift_pct': round(roas_lift_pct, 1), # Will be labeled "ROAS Change" in UI
            'incremental_revenue': round(revenue_impact, 2), # Primary metric
            # Statistical Significance
            'p_value': round(p_value, 4),
            'is_significant': is_significant,
            'confidence_pct': round(confidence_pct, 1),
            # Implementation
            'implementation_rate': round(impl_rate, 1),
            'confirmed_impact': int(confirmed_count),
            'pending': int(pending_count),
            'not_implemented': int(not_impl_count),
            # Legacy/Secondary
            'win_rate': round(win_rate, 1),
            'winners': int(winners),
            'losers': int(losers),
            'by_action_type': by_action_type
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
