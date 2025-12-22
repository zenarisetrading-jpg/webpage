"""
PostgreSQL Database Manager for Supabase Integration.

Implements the same interface as DatabaseManager but uses psycopg2 and PostgreSQL syntax.
Handles 'ON CONFLICT' for upserts instead of 'INSERT OR REPLACE'.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from typing import Optional, List, Dict, Any, Union
from datetime import date, datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import uuid

class PostgresManager:
    """
    PostgreSQL persistence for Supabase / Cloud Postgres.
    """
    
    def __init__(self, db_url: str):
        """
        Initialize Postgres manager.
        
        Args:
            db_url: Postgres connection string (postgres://user:pass@host:port/db)
        """
        self.db_url = db_url
        self._init_schema()
    
    @property
    def placeholder(self) -> str:
        """SQL parameter placeholder for Postgres."""
        return "%s"
    
    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections."""
        conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
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
            with conn.cursor() as cursor:
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
            with conn.cursor() as cursor:
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
        if df is None or df.empty:
            return 0
        
        # NOTE: Reusing the same logic from DB Manager for dataframe processing
        # Identifying columns
        target_col = None
        for col in ['Customer Search Term', 'Targeting', 'Keyword Text']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            return 0
            
        required = ['Campaign Name', 'Ad Group Name']
        if not all(col in df.columns for col in required):
            return 0
            
        # Standard aggregation maps
        agg_cols = {}
        if 'Spend' in df.columns: agg_cols['Spend'] = 'sum'
        if 'Sales' in df.columns: agg_cols['Sales'] = 'sum'
        if 'Clicks' in df.columns: agg_cols['Clicks'] = 'sum'
        if 'Impressions' in df.columns: agg_cols['Impressions'] = 'sum'
        if 'Orders' in df.columns: agg_cols['Orders'] = 'sum'
        if 'Match Type' in df.columns: agg_cols['Match Type'] = 'first'
        
        if not agg_cols:
            return 0
        
        df_copy = df.copy()
        
        # Date Logic
        date_col = None
        for col in ['Date', 'Start Date', 'Report Date', 'date', 'start_date']:
            if col in df_copy.columns:
                date_col = col
                break
        
        if date_col:
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
        
        df_copy['_camp_norm'] = df_copy['Campaign Name'].astype(str).str.lower().str.strip()
        df_copy['_ag_norm'] = df_copy['Ad Group Name'].astype(str).str.lower().str.strip()
        df_copy['_target_norm'] = df_copy[target_col].astype(str).str.lower().str.strip()
        
        total_saved = 0
        
        for week_start in weeks:
            if pd.isna(week_start): continue
            
            week_data = df_copy[df_copy['_week_start'] == week_start]
            if week_data.empty: continue
            
            grouped = week_data.groupby(['_camp_norm', '_ag_norm', '_target_norm']).agg(agg_cols).reset_index()
            
            week_start_str = week_start.isoformat() if isinstance(week_start, date) else str(week_start)[:10]
            
            # Prepare data for bulk insert
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
                    with conn.cursor() as cursor:
                        execute_values(cursor, """
                            INSERT INTO target_stats 
                            (client_id, start_date, campaign_name, ad_group_name, target_text, match_type, spend, sales, orders, clicks, impressions, updated_at)
                            VALUES %s
                            ON CONFLICT (client_id, start_date, campaign_name, ad_group_name, target_text) DO UPDATE SET
                                spend = EXCLUDED.spend,
                                sales = EXCLUDED.sales,
                                orders = EXCLUDED.orders,
                                clicks = EXCLUDED.clicks,
                                impressions = EXCLUDED.impressions,
                                updated_at = CURRENT_TIMESTAMP
                        """, records)
                        
                total_saved += len(records)
                
        return total_saved

    def get_all_weekly_stats(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM weekly_stats ORDER BY start_date DESC")
                return cursor.fetchall()
    
    def get_stats_by_client(self, client_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
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
            with conn.cursor() as cursor:
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
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT client_id FROM weekly_stats ORDER BY client_id")
                return [row['client_id'] for row in cursor.fetchall()]

    def delete_stats_by_client(self, client_id: str) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM weekly_stats WHERE client_id = %s", (client_id,))
                rows = cursor.rowcount
                cursor.execute("DELETE FROM target_stats WHERE client_id = %s", (client_id,))
                rows += cursor.rowcount
                cursor.execute("DELETE FROM actions_log WHERE client_id = %s", (client_id,))
                rows += cursor.rowcount
                return rows

    def clear_all_stats(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM weekly_stats")
                return cursor.rowcount

    def get_connection_status(self) -> tuple[str, str]:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return "Connected (Postgres)", "green"
        except Exception as e:
            return f"Error: {str(e)}", "red"

    def get_stats_summary(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
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
            with conn.cursor() as cursor:
                execute_values(cursor, """
                    INSERT INTO category_mappings (client_id, sku, category, sub_category, updated_at)
                    VALUES %s
                    ON CONFLICT (client_id, sku) DO UPDATE SET
                        category = EXCLUDED.category,
                        sub_category = EXCLUDED.sub_category,
                        updated_at = CURRENT_TIMESTAMP
                """, data)
        return len(data)

    def get_category_mappings(self, client_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql("SELECT sku as SKU, category as Category, sub_category as 'Sub-Category' FROM category_mappings WHERE client_id = %s", conn, params=(client_id,))

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
            with conn.cursor() as cursor:
                execute_values(cursor, """
                    INSERT INTO advertised_product_cache 
                    (client_id, campaign_name, ad_group_name, sku, asin, updated_at)
                    VALUES %s
                    ON CONFLICT (client_id, campaign_name, ad_group_name, sku) DO UPDATE SET
                        asin = EXCLUDED.asin,
                        updated_at = CURRENT_TIMESTAMP
                """, data)
        return len(data)

    def get_advertised_product_map(self, client_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql("SELECT campaign_name as 'Campaign Name', ad_group_name as 'Ad Group Name', sku as SKU, asin as ASIN FROM advertised_product_cache WHERE client_id = %s", conn, params=(client_id,))

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
            with conn.cursor() as cursor:
                execute_values(cursor, """
                    INSERT INTO bulk_mappings 
                    (client_id, campaign_name, campaign_id, ad_group_name, ad_group_id, 
                        keyword_text, keyword_id, targeting_expression, targeting_id, sku, match_type, updated_at)
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
                action.get('match_type', '')
            ))
            
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                execute_values(cursor, """
                    INSERT INTO actions_log 
                    (action_date, client_id, batch_id, entity_name, action_type, old_value, new_value, 
                        reason, campaign_name, ad_group_name, target_text, match_type)
                    VALUES %s
                    ON CONFLICT (client_id, action_date, target_text, action_type, campaign_name) DO NOTHING
                """, data)
        return len(actions)

    def create_account(self, account_id: str, account_name: str, account_type: str = 'brand', metadata: dict = None) -> bool:
        import json
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    metadata_json = json.dumps(metadata) if metadata else '{}'
                    cursor.execute("""
                        INSERT INTO accounts (account_id, account_name, account_type, metadata)
                        VALUES (%s, %s, %s, %s)
                    """, (account_id, account_name, account_type, metadata_json))
                    return True
        except psycopg2.IntegrityError:
            return False

    def get_all_accounts(self) -> List[tuple]:
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT account_id, account_name, account_type FROM accounts ORDER BY account_name")
                return [(row['account_id'], row['account_name'], row['account_type']) for row in cursor.fetchall()]

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        import json
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
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
                with conn.cursor() as cursor:
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
                        metrics.get('health_score', 0),
                        metrics.get('roas_score', 0),
                        metrics.get('efficiency_score', metrics.get('waste_score', 0)),
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
        except Exception as e:
            print(f"Failed to save account health: {e}")
            return False
    
    def get_account_health(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get account health metrics from database."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM account_health_metrics WHERE client_id = %s",
                    (client_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    # ==========================================
    # IMPACT DASHBOARD METHODS
    # ==========================================
    
    def get_available_dates(self, client_id: str) -> List[str]:
        """Get list of unique action dates for a client."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT DATE(action_date) as action_date 
                    FROM actions_log 
                    WHERE client_id = %s 
                    ORDER BY action_date DESC
                """, (client_id,))
                return [str(row['action_date']) for row in cursor.fetchall()]
    
    def get_action_impact(self, client_id: str) -> pd.DataFrame:
        """
        Get actions with before/after performance data for impact analysis.
        Joins actions_log with target_stats to compare performance.
        """
        with self._get_connection() as conn:
            query = """
                WITH action_weeks AS (
                    SELECT 
                        a.action_date,
                        a.action_type,
                        a.target_text,
                        a.campaign_name,
                        a.ad_group_name,
                        a.old_value,
                        a.new_value,
                        a.reason,
                        DATE(a.action_date) as action_week
                    FROM actions_log a
                    WHERE a.client_id = %s
                ),
                before_stats AS (
                    SELECT 
                        t.target_text,
                        t.campaign_name,
                        t.start_date,
                        t.spend as before_spend,
                        t.sales as before_sales,
                        t.orders as before_orders,
                        t.clicks as before_clicks
                    FROM target_stats t
                    WHERE t.client_id = %s
                ),
                after_stats AS (
                    SELECT 
                        t.target_text,
                        t.campaign_name,
                        t.start_date,
                        t.spend as after_spend,
                        t.sales as after_sales,
                        t.orders as after_orders,
                        t.clicks as after_clicks
                    FROM target_stats t
                    WHERE t.client_id = %s
                )
                SELECT 
                    aw.action_date,
                    aw.action_type,
                    aw.target_text,
                    aw.campaign_name,
                    aw.ad_group_name,
                    aw.old_value,
                    aw.new_value,
                    aw.reason,
                    COALESCE(bs.before_spend, 0) as before_spend,
                    COALESCE(bs.before_sales, 0) as before_sales,
                    COALESCE(afs.after_spend, 0) as after_spend,
                    COALESCE(afs.after_sales, 0) as after_sales,
                    COALESCE(afs.after_sales, 0) - COALESCE(bs.before_sales, 0) as delta_sales,
                    COALESCE(afs.after_spend, 0) - COALESCE(bs.before_spend, 0) as delta_spend,
                    COALESCE(afs.after_sales, 0) - COALESCE(afs.after_spend, 0) - 
                        (COALESCE(bs.before_sales, 0) - COALESCE(bs.before_spend, 0)) as impact_score,
                    CASE WHEN COALESCE(afs.after_sales, 0) > COALESCE(bs.before_sales, 0) 
                         THEN true ELSE false END as is_winner
                FROM action_weeks aw
                LEFT JOIN before_stats bs ON 
                    LOWER(aw.target_text) = LOWER(bs.target_text) AND
                    LOWER(aw.campaign_name) = LOWER(bs.campaign_name) AND
                    bs.start_date < aw.action_week
                LEFT JOIN after_stats afs ON 
                    LOWER(aw.target_text) = LOWER(afs.target_text) AND
                    LOWER(aw.campaign_name) = LOWER(afs.campaign_name) AND
                    afs.start_date >= aw.action_week
                ORDER BY aw.action_date DESC
            """
            return pd.read_sql(query, conn, params=(client_id, client_id, client_id))
    
    def get_impact_summary(self, client_id: str) -> Dict[str, Any]:
        """Get aggregated impact metrics for a client."""
        impact_df = self.get_action_impact(client_id)
        
        if impact_df.empty:
            return {
                'total_actions': 0,
                'net_sales_impact': 0,
                'net_spend_change': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'roi': 0,
                'by_action_type': {}
            }
        
        # Calculate summary
        total_actions = len(impact_df)
        net_sales = impact_df['delta_sales'].fillna(0).sum()
        net_spend = impact_df['delta_spend'].fillna(0).sum()
        winners = impact_df['is_winner'].fillna(False).sum()
        losers = total_actions - winners
        win_rate = (winners / total_actions * 100) if total_actions > 0 else 0
        roi = net_sales / net_spend if net_spend != 0 else 0
        
        # By action type
        by_action_type = {}
        for action_type in impact_df['action_type'].unique():
            type_data = impact_df[impact_df['action_type'] == action_type]
            by_action_type[action_type] = {
                'count': len(type_data),
                'net_sales': type_data['delta_sales'].fillna(0).sum(),
                'net_spend': type_data['delta_spend'].fillna(0).sum()
            }
        
        return {
            'total_actions': total_actions,
            'net_sales_impact': net_sales,
            'net_spend_change': net_spend,
            'winners': int(winners),
            'losers': int(losers),
            'win_rate': win_rate,
            'roi': roi,
            'by_action_type': by_action_type
        }

    # ==========================================
    # REFERENCE DATA STATUS
    # ==========================================
    
    def get_reference_data_status(self) -> Dict[str, Any]:
        """Check reference data freshness for sidebar badge."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
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
                with conn.cursor() as cursor:
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
    
    def delete_account(self, account_id: str) -> bool:
        """Delete an account and all its data."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
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
