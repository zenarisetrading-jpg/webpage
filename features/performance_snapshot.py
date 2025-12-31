"""
Performance Snapshot Module

Comprehensive dashboard for visualizing campaign performance.
Features:
- Executive Dashboard (High-level KPIs with trends)
- Campaign Trend Analysis (Interactive Charts)
- Performance Breakdown (By Match Type, Category, etc.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from features._base import BaseFeature
from features.constants import classify_match_type
from core.data_hub import DataHub
from core.data_loader import SmartMapper, safe_numeric
from utils.metrics import calculate_ppc_metrics, ensure_numeric_columns

class PerformanceSnapshotModule(BaseFeature):
    """Performance Snapshot Dashboard."""
    
    def render_ui(self):
        """Render the dashboard UI."""
        # 1. Load Data FIRST
        from core.db_manager import get_db_manager
        
        db = get_db_manager(st.session_state.get('test_mode', False))
        
        # USE ACTIVE ACCOUNT from session state
        client_id = st.session_state.get('active_account_id', 'default_client')
        
        hide_header = st.session_state.get('active_perf_tab') is not None
        
        if not client_id:
            if not hide_header:
                st.title("📊 Account Overview")
            st.error("⚠️ No account selected! Please select an account in the sidebar.")
            return
        
        # Fetch persistent data for this account
        db_data = db.get_target_stats_df(client_id)
        
        if db_data.empty:
            if not hide_header:
                st.title("📊 Account Overview")
            st.warning(f"⚠️ No data found for account '{st.session_state.get('active_account_name', client_id)}'. Please upload a Search Term Report in the Data Hub.")
            return

        # 2. Enrichment Logic
        hub = DataHub()
        self.data = db_data
        
        # ===== MERGE SKU INFO - PREFER DATABASE =====
        # Session state may have stale/mismatched data, so try DB first
        sku_merge_success = False
        
        # STEP 1: Try DB advertised product cache FIRST (most reliable)
        try:
            adv_cache = db.get_advertised_product_map(client_id)
            # Find SKU column case-insensitively (DB returns 'sku' lowercase)
            sku_col = next((c for c in adv_cache.columns if c.lower() == 'sku'), None) if adv_cache is not None else None
            
            if adv_cache is not None and not adv_cache.empty and sku_col:
                print(f"📦 Merging SKU from DB cache: {len(adv_cache)} rows, column '{sku_col}'")
                
                # Normalize for merge
                self.data['Camp_Norm'] = self.data['Campaign Name'].astype(str).str.strip().str.lower()
                self.data['AG_Norm'] = self.data['Ad Group Name'].astype(str).str.strip().str.lower()
                adv_cache['Camp_Norm'] = adv_cache['Campaign Name'].astype(str).str.strip().str.lower()
                adv_cache['AG_Norm'] = adv_cache['Ad Group Name'].astype(str).str.strip().str.lower()
                
                sku_lookup = adv_cache.groupby(['Camp_Norm', 'AG_Norm'])[sku_col].apply(
                    lambda x: ', '.join(str(s) for s in x.dropna().unique() if str(s).strip())
                ).reset_index()
                sku_lookup.columns = ['Camp_Norm', 'AG_Norm', 'SKU_advertised']
                
                # Debug: Check overlap
                data_camps = set(self.data['Camp_Norm'].unique())
                adv_camps = set(sku_lookup['Camp_Norm'].unique())
                overlap = len(data_camps & adv_camps)
                print(f"📦 Campaign overlap: {overlap}/{len(data_camps)} data campaigns match adv cache")
                
                self.data = self.data.merge(sku_lookup, on=['Camp_Norm', 'AG_Norm'], how='left')
                self.data.drop(columns=['Camp_Norm', 'AG_Norm'], inplace=True, errors='ignore')
                
                # Check success
                if 'SKU_advertised' in self.data.columns:
                    coverage = self.data['SKU_advertised'].notna().mean()
                    print(f"📦 SKU coverage after DB merge: {coverage:.1%}")
                    if coverage > 0.05:
                        sku_merge_success = True
        except Exception as e:
            print(f"⚠️ DB advertised product merge failed: {e}")
        
        # STEP 2: Fallback to session state only if DB failed
        if not sku_merge_success:
            if hub.is_loaded('advertised_product_report'):
                adv_report = hub.get_data('advertised_product_report')
                if adv_report is not None:
                    # Case-insensitive SKU column lookup
                    sku_col = next((c for c in adv_report.columns if c.lower() == 'sku'), None)
                    
                    if sku_col and 'Campaign Name' in adv_report.columns and 'Ad Group Name' in adv_report.columns:
                        print(f"📦 Fallback: Merging SKU from session state: {len(adv_report)} rows")
                        
                        self.data['Camp_Norm'] = self.data['Campaign Name'].astype(str).str.strip().str.lower()
                        self.data['AG_Norm'] = self.data['Ad Group Name'].astype(str).str.strip().str.lower()
                        
                        adv_report = adv_report.copy()
                        adv_report['Camp_Norm'] = adv_report['Campaign Name'].astype(str).str.strip().str.lower()
                        adv_report['AG_Norm'] = adv_report['Ad Group Name'].astype(str).str.strip().str.lower()
                        
                        sku_lookup = adv_report.groupby(['Camp_Norm', 'AG_Norm'])[sku_col].apply(
                            lambda x: ', '.join(str(s) for s in x.dropna().unique() if str(s).strip())
                        ).reset_index()
                        sku_lookup.columns = ['Camp_Norm', 'AG_Norm', 'SKU_advertised']
                        
                        # If SKU_advertised column already exists but empty, drop it first
                        if 'SKU_advertised' in self.data.columns:
                            self.data.drop(columns=['SKU_advertised'], inplace=True)
                        
                        self.data = self.data.merge(sku_lookup, on=['Camp_Norm', 'AG_Norm'], how='left')
                        self.data.drop(columns=['Camp_Norm', 'AG_Norm'], inplace=True, errors='ignore')

        
        # --- BRIDGE FALLBACK: Try Bulk File if SKU still missing ---
        # If 'SKU_advertised' is missing or largely empty, try to link via Bulk File (Campaign -> SKU)
        sku_exists = 'SKU_advertised' in self.data.columns
        sku_coverage = self.data['SKU_advertised'].notna().mean() if sku_exists else 0
        
        if not sku_exists or sku_coverage < 0.1: # Less than 10% matched
             bulk_map = hub.get_data('bulk_id_mapping')
             if bulk_map is not None and 'Campaign Name' in bulk_map.columns:
                  # Find SKU column
                  sku_candidates = [c for c in bulk_map.columns if c.lower() in ['sku', 'msku', 'vendor sku', 'vendor_sku']]
                  
                  if sku_candidates:
                       sku_col = sku_candidates[0]
                       
                       # Normalize & Aggregate SKUs by Campaign
                       # We map Campaign -> comma-separated SKUs
                       bulk_map['Camp_Norm'] = bulk_map['Campaign Name'].astype(str).str.strip().str.lower()
                       bridge = bulk_map.groupby('Camp_Norm')[sku_col].apply(
                           lambda x: ', '.join(x.dropna().unique().astype(str))
                       ).reset_index()
                       bridge.columns = ['Camp_Norm', 'SKU_From_Bulk']
                       
                       # Prepare Main Data
                       self.data['Camp_Norm'] = self.data['Campaign Name'].astype(str).str.strip().str.lower()
                       
                       # Merge
                       self.data = self.data.merge(bridge, on='Camp_Norm', how='left')
                       
                       # Apply to SKU_advertised
                       if 'SKU_advertised' not in self.data.columns:
                           self.data['SKU_advertised'] = self.data['SKU_From_Bulk']
                       else:
                           self.data['SKU_advertised'] = self.data['SKU_advertised'].fillna(self.data['SKU_From_Bulk'])
                           
                       # Cleanup
                       self.data.drop(columns=['Camp_Norm', 'SKU_From_Bulk'], inplace=True, errors='ignore')
        
        
        # Merge Category Mapping - ALWAYS prefer fresh DB data
        # Session state can have stale data from previous uploads that doesn't match current account
        cat_map = None
        
        # First try DB (most reliable, tied to current account)
        try:
            cat_map = db.get_category_mappings(client_id)
            if cat_map is not None and not cat_map.empty:
                print(f"🔍 Category Map loaded from DB: {len(cat_map)} rows for {client_id}")
        except Exception as e:
            print(f"⚠️ Failed to load category map from DB: {e}")
        
        # Fallback to session state only if DB failed
        if cat_map is None or cat_map.empty:
            cat_map = hub.get_data('category_mapping')
            if cat_map is not None and not cat_map.empty:
                print(f"🔍 Category Map loaded from session: {len(cat_map)} rows")
        
        if cat_map is not None and not cat_map.empty:

            
            # Check which product ID column we have
            has_sku = 'SKU_advertised' in self.data.columns
            has_asin = 'ASIN_advertised' in self.data.columns
            
            # Determine which column to use (prefer SKU, fallback to ASIN)
            product_id_col = None
            if has_sku and self.data['SKU_advertised'].notna().sum() > 0:
                product_id_col = 'SKU_advertised'
            elif has_asin and self.data['ASIN_advertised'].notna().sum() > 0:
                product_id_col = 'ASIN_advertised'
            
            print(f"🔍 Using product ID column: {product_id_col}")
            if product_id_col:
                print(f"🔍 Sample values: {self.data[product_id_col].dropna().head(3).tolist()}")
            
            if cat_map is not None and product_id_col is not None:
                # Find product ID column in category map - be more flexible
                cat_id_candidates = [c for c in cat_map.columns if any(s in c.lower() for s in ['sku', 'asin', 'product'])]
                cat_id_col = cat_id_candidates[0] if cat_id_candidates else None
                
                print(f"🔍 Found product ID column in category map: {cat_id_col}")
                print(f"🔍 Category map columns: {list(cat_map.columns)}")
                if cat_id_col:
                    print(f"🔍 Category map sample values for '{cat_id_col}': {cat_map[cat_id_col].head(3).tolist()}")
                
                if cat_id_col:
                    # Normalize the product ID column (SKU or ASIN)
                    # Handle comma-separated IDs by creating multiple rows
                    # IMPORTANT: Filter out NaN BEFORE converting to string to avoid 'nan' strings
                    self.data['ID_List'] = self.data[product_id_col].apply(
                        lambda x: [s.strip() for s in str(x).split(',')] if pd.notna(x) and str(x).lower() != 'nan' else []
                    )
                    
                    # Explode to handle multi-ID rows (empty lists will create NaN rows)
                    exploded = self.data.explode('ID_List')
                    
                    # Filter out empty/NaN rows from explosion
                    exploded = exploded[exploded['ID_List'].notna() & (exploded['ID_List'] != '')]
                    
                    # AGGRESSIVE normalization for matching
                    # Remove all non-alphanumeric characters, lowercase
                    exploded['ID_Clean'] = (
                        exploded['ID_List']
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r'[^a-z0-9]', '', regex=True)
                    )
                    
                    # Filter out 'nan' strings that might have slipped through
                    exploded = exploded[exploded['ID_Clean'] != 'nan']
                    exploded = exploded[exploded['ID_Clean'] != '']
                    
                    # Normalize category map IDs with same aggressive normalization
                    cat_map = cat_map.copy()
                    cat_map['ID_Clean'] = (
                        cat_map[cat_id_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r'[^a-z0-9]', '', regex=True)
                    )
                    
                    # DEBUG: Log sample IDs for troubleshooting
                    sample_data_ids = exploded['ID_Clean'].dropna().head(5).tolist()
                    sample_cat_ids = cat_map['ID_Clean'].dropna().head(5).tolist()
                    print(f"🔍 Category Mapping - Sample Data IDs: {sample_data_ids}")
                    print(f"🔍 Category Mapping - Sample Category Map IDs: {sample_cat_ids}")
                    
                    # Find category columns - CASE INSENSITIVE (DB returns lowercase)
                    cat_cols_to_merge = ['ID_Clean']
                    cat_col = next((c for c in cat_map.columns if c.lower() == 'category'), None)
                    subcat_col = next((c for c in cat_map.columns if c.lower() == 'sub-category'), None)
                    
                    if cat_col:
                        cat_cols_to_merge.append(cat_col)
                    if subcat_col:
                        cat_cols_to_merge.append(subcat_col)
                    
                    print(f"🔍 Category columns to merge: {cat_cols_to_merge}")

                    
                    # Merge
                    merged = exploded.merge(
                        cat_map[cat_cols_to_merge], 
                        on='ID_Clean', 
                        how='left'
                    )
                    
                    # Check match rate - use actual column name (case-insensitive)
                    if cat_col and cat_col in merged.columns:
                        match_rate = merged[cat_col].notna().sum() / len(merged) if len(merged) > 0 else 0
                        print(f"📊 Category Mapping Match Rate: {match_rate:.1%}")
                    else:
                        print(f"⚠️ Category column '{cat_col}' not found in merged data")
                    
                    # De-duplicate back (take first match per original row)
                    # Group by original index and take first
                    first_match = merged.groupby(level=0).first()
                    
                    # Update self.data with category info - use actual column names and normalize to standard names
                    if cat_col and cat_col in first_match.columns:
                        self.data['Category'] = first_match[cat_col]  # Store as standard 'Category'
                    if subcat_col and subcat_col in first_match.columns:
                        self.data['Sub-Category'] = first_match[subcat_col]  # Store as standard 'Sub-Category'
                    
                    # Cleanup
                    self.data.drop(columns=['ID_List'], inplace=True, errors='ignore')


        # ---------------------
        
        if not hide_header:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                        border: 1px solid rgba(91, 85, 111, 0.2); 
                        border-radius: 8px; 
                        padding: 12px 16px; 
                        margin-bottom: 32px;
                        display: flex; 
                        align-items: center; 
                        justify-content: space-between;">
                <div style="display: flex; align-items: center;">
                    {overview_icon}
                    <span style="color: #F5F5F7; font-size: 1.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">Account Overview</span>
                </div>
                <div style="color: #8F8CA3; font-size: 0.8rem; font-weight: 600;">
                    Snapshot: {st.session_state.get('active_account_name', client_id)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        self.date_filter = None
        if not self.data.empty:
            date_col_name = next((c for c in ['Date', 'date'] if c in self.data.columns), None)
            if date_col_name:
                try:
                    dates = pd.to_datetime(self.data[date_col_name], errors='coerce').dropna()
                    if not dates.empty:
                        min_d, max_d = dates.min().date(), dates.max().date()
                        
                        # --- Unified Control Bar (Fixed 7-Day Comparison) ---
                        comp_days = 7  # Fixed to 7-day comparison
                        
                        c1, c2 = st.columns([5, 3])
                        with c1:
                            st.markdown(f'<div style="margin-top: 8px; color: #8F8CA3; font-size: 0.9rem;">Showing <b>Last 7 Days</b> vs. <b>Previous 7 Days</b></div>', unsafe_allow_html=True)
                        with c2:
                            self.date_filter = st.date_input(
                                "Date Range",
                                value=(min_d, max_d),
                                min_value=min_d,
                                max_value=max_d,
                                label_visibility="collapsed",
                                key="overview_date_picker"
                            )
                            st.markdown('<div style="text-align: right; margin-top: -10px; color: #8F8CA3; font-size: 0.75rem;">📅 Period Analysis</div>', unsafe_allow_html=True)
                        
                        st.session_state['overview_comp_days'] = comp_days
                except Exception:
                    pass

    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate required columns."""
        required = ['Spend', 'Sales', 'Impressions', 'Clicks']
        missing = [col for col in required if col not in data.columns]
        if missing:
            # Try mapping
            mapped = SmartMapper.map_columns(data)
            missing_mapped = [col for col in required if col not in mapped]
            if missing_mapped:
                return False, f"Missing columns: {missing_mapped}"
        return True, ""

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for visualization."""
        df = data.copy()

        # Ensure numeric (using shared utility)
        df = ensure_numeric_columns(df, inplace=True)

        # Create derived metrics (using shared utility)
        # performance_snapshot.py uses percentage format: 5.0 = 5%
        df = calculate_ppc_metrics(df, percentage_format='percentage', inplace=True)
        
        # Handle Date for Trends
        # Try to find a date column
        date_col = None
        for col in ['Date', 'Start Date', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Apply Global Date Filter if set in render_ui
            if getattr(self, 'date_filter', None):
                try:
                    dates = self.date_filter
                    # Handle single date or range tuple
                    if isinstance(dates, (tuple, list)):
                        start = pd.Timestamp(dates[0])
                        end = pd.Timestamp(dates[1]) if len(dates) > 1 else start
                        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
                except Exception:
                    pass

        # ---------------------------------------------------------
        # Unified Match Type Logic
        # ---------------------------------------------------------
        # Create a new column 'Refined Match Type'
        # Start with original Match Type
        df['Refined Match Type'] = df['Match Type'].fillna('-').astype(str)

        # Apply refinement using shared classify_match_type from features.constants
        # Ensure Targeting column exists
        if 'Targeting' in df.columns and not df.empty:
            df['Refined Match Type'] = df.apply(classify_match_type, axis=1)
        
        return {
            'data': df,
            'date_col': date_col
        }

    def _calculate_comparison_metrics(self, df: pd.DataFrame, days: int = 7) -> Dict[str, Any]:
        """Calculate metrics for the latest N days vs. the previous N days."""
        if df.empty or 'Date' not in df.columns:
            return {}

        df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        max_date = df['Date_Parsed'].max()
        if pd.isna(max_date):
            return {}

        # Define Periods
        period_end = max_date
        period_start = max_date - pd.Timedelta(days=days-1)
        prev_period_end = period_start - pd.Timedelta(days=1)
        prev_period_start = prev_period_end - pd.Timedelta(days=days-1)

        # Current Period Data
        curr_df = df[(df['Date_Parsed'] >= period_start) & (df['Date_Parsed'] <= period_end)]
        # Previous Period Data
        prev_df = df[(df['Date_Parsed'] >= prev_period_start) & (df['Date_Parsed'] <= prev_period_end)]

        def get_kpis(data_df):
            spend = data_df['Spend'].sum()
            sales = data_df['Sales'].sum()
            orders = data_df['Orders'].sum()
            clicks = data_df['Clicks'].sum() if 'Clicks' in data_df.columns else 0
            roas = sales / spend if spend > 0 else 0
            acos = (spend / sales * 100) if sales > 0 else 0
            cvr = (orders / clicks * 100) if clicks > 0 else 0
            return {'spend': spend, 'sales': sales, 'orders': orders, 'roas': roas, 'acos': acos, 'cvr': cvr}

        curr_stats = get_kpis(curr_df)
        prev_stats = get_kpis(prev_df)

        deltas = {}
        for key in curr_stats:
            curr_v = curr_stats[key]
            prev_v = prev_stats[key]
            
            if prev_v > 0:
                # For ACOS, decrease is positive (+) delta in common parlance, but let's keep it literal (-)
                change_pct = ((curr_v / prev_v) - 1) * 100
                deltas[key] = f"{change_pct:+.1f}%"
            else:
                deltas[key] = None
        
        return deltas

    def display_results(self, results: Dict[str, Any]):
        """Display the dashboard."""
        df = results['data']
        date_col = results['date_col']
        
        # ==========================================
        # 1. Executive Dashboard (KPI Cards with Trends)
        # ==========================================
        
        # Calculate comparison
        # Comparison Period from unified bar
        comp_days = st.session_state.get('overview_comp_days', 7)
        
        deltas = self._calculate_comparison_metrics(df, comp_days)
        
        # Calculate Totals
        total_spend = df['Spend'].sum()
        total_sales = df['Sales'].sum()
        total_orders = df['Orders'].sum()
        total_clicks = df['Clicks'].sum()
        total_impr = df['Impressions'].sum()
        
        # Weighted Averages
        total_roas = total_sales / total_spend if total_spend > 0 else 0
        total_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        total_ctr = (total_clicks / total_impr * 100) if total_impr > 0 else 0
        total_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        total_cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
        
        # Layout metrics with HTML
        from ui.components import metric_card
        from utils.formatters import get_account_currency
        currency = get_account_currency()
        
        # === PRIMARY METRICS (4 cards with 7-day deltas) ===
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Spend", f"{currency} {total_spend:,.0f}", icon_name="spend", delta=deltas.get('spend'))
        with c2: metric_card("Revenue", f"{currency} {total_sales:,.0f}", icon_name="revenue", delta=deltas.get('sales'))
        with c3: metric_card("ROAS", f"{total_roas:.2f}x", icon_name="roas", delta=deltas.get('roas'))
        with c4: metric_card("CVR", f"{total_cvr:.2f}%", icon_name="roas", delta=deltas.get('cvr') if 'cvr' in deltas else None)
        
        # === SECONDARY METRICS (Expandable Section) ===
        with st.expander("View More Metrics", expanded=False):
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            with sc1: metric_card("Orders", f"{total_orders:,.0f}", icon_name="orders", delta=deltas.get('orders'))
            with sc2: metric_card("ACOS", f"{total_acos:.2f}%", icon_name="acos", delta=deltas.get('acos'))
            with sc3: metric_card("Impressions", f"{total_impr:,.0f}", icon_name="impressions")
            with sc4: metric_card("Clicks", f"{total_clicks:,.0f}", icon_name="clicks")
            with sc5: metric_card("CPC", f"{currency} {total_cpc:.2f}", icon_name="spend")
        
        st.markdown("---")

        # ==========================================
        # 2. Trend Analysis
        # ==========================================
        if date_col and df['Date'].notna().any():
            st.markdown("### 📈 Campaign Trend Analysis")
            
            # Layout: Trend Chart (Left) + Scatter Plot (Right)
            t_col1, t_col2 = st.columns([2, 1])
            
            with t_col1:
                # Controls (Inside the column)
                c1, c2, c3 = st.columns([1, 1, 1])
                time_frame = c1.selectbox("Timeframe", ["Weekly", "Monthly", "Quarterly", "Yearly"], index=0)
                metric_bar = c2.selectbox("Bar Metric", ["Sales", "Spend", "Orders", "Clicks", "Impressions"], index=0)
                metric_line = c3.selectbox("Line Metric", ["ACOS", "ROAS", "CPC", "CTR", "CVR"], index=0)
                
                # Resample Data based on selection
                trend_df = df.set_index('Date').sort_index()
                
                # Resampling Rules: W=Weekly, M=Monthly, Q=Quarterly, Y=Yearly
                if time_frame == "Weekly":
                    rule = 'W'
                elif time_frame == "Monthly":
                    rule = 'M'
                elif time_frame == "Quarterly":
                    rule = 'Q'
                else:
                    rule = 'Y'
                
                resampled = trend_df.resample(rule).agg({
                    'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum', 
                    'Clicks': 'sum', 'Impressions': 'sum'
                }).reset_index()
                
                # Recalculate rates
                resampled['ROAS'] = np.where(resampled['Spend'] > 0, resampled['Sales'] / resampled['Spend'], 0)
                resampled['ACOS'] = np.where(resampled['Sales'] > 0, resampled['Spend'] / resampled['Sales'] * 100, 0)
                resampled['CPC'] = np.where(resampled['Clicks'] > 0, resampled['Spend'] / resampled['Clicks'], 0)
                resampled['CTR'] = np.where(resampled['Impressions'] > 0, resampled['Clicks'] / resampled['Impressions'] * 100, 0)
                resampled['CVR'] = np.where(resampled['Clicks'] > 0, resampled['Orders'] / resampled['Clicks'] * 100, 0)
                
                # Plot Trend
                fig = go.Figure()
                
                # Bar Chart
                fig.add_trace(go.Bar(
                    x=resampled['Date'], 
                    y=resampled[metric_bar], 
                    name=metric_bar,
                    marker_color='#5B556F', # Brand Purple
                    marker_line_width=0,
                    opacity=0.9
                ))
                
                # Line Chart (Dual Axis)
                fig.add_trace(go.Scatter(
                    x=resampled['Date'], 
                    y=resampled[metric_line], 
                    name=metric_line,
                    yaxis='y2',
                    line=dict(color='#22d3ee', width=3) # Accent Cyan
                ))
                
                # Get dynamic template
                from ui.theme import ThemeManager
                chart_template = ThemeManager.get_chart_template()
                is_dark = st.session_state.get('theme_mode', 'dark') == 'dark'
                bg_color = 'rgba(0,0,0,0)' 
                text_color = '#f3f4f6' if is_dark else '#1f2937'
                
                fig.update_layout(
                    title=dict(text=f"{metric_bar} vs {metric_line} ({time_frame})", font=dict(color=text_color)),
                    yaxis=dict(title=dict(text=metric_bar, font=dict(color=text_color)), tickfont=dict(color=text_color), showgrid=False),
                    yaxis2=dict(title=dict(text=metric_line, font=dict(color=text_color)), overlaying='y', side='right', tickfont=dict(color=text_color), showgrid=False),
                    hovermode='x unified',
                    template=chart_template,
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color),
                    height=450,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(font=dict(color=text_color))
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with t_col2:
                # ------------------------------------
                # BUBBLE CHART: ROAS vs CVR (Size = Orders)
                # ------------------------------------
                # Spacers to align with the chart on the left (pushing it down)
                st.write("")
                st.write("")
                
                st.markdown(f"**Campaign Performance Quadrants**")
                
                # Determine Data Source for Bubble Chart (Prioritize Uploaded File for "Current Snapshot")
                bubble_df = df
                from core.data_hub import DataHub
                hub = DataHub()
                if hub.is_loaded('search_term_report'):
                     uploaded = hub.get_enriched_data()
                     if uploaded is None:
                         uploaded = hub.get_data('search_term_report')
                     
                     if uploaded is not None:
                         bubble_df = uploaded.copy()
                         # Ensure numeric for aggregation
                         params = ['Spend', 'Sales', 'Orders', 'Clicks']
                         for p in params:
                             if p in bubble_df.columns:
                                 bubble_df[p] = pd.to_numeric(bubble_df[p], errors='coerce').fillna(0)
                
                # Group by Campaign
                camp_agg = bubble_df.groupby('Campaign Name').agg({
                    'Sales': 'sum', 
                    'Spend': 'sum',
                    'Orders': 'sum',
                    'Clicks': 'sum'
                }).reset_index()
                
                # Calc Metrics
                camp_agg['ROAS'] = np.where(camp_agg['Spend'] > 0, camp_agg['Sales'] / camp_agg['Spend'], 0)
                camp_agg['CVR'] = np.where(camp_agg['Clicks'] > 0, camp_agg['Orders'] / camp_agg['Clicks'] * 100, 0)
                
                # Calculate Medians
                median_cvr = camp_agg['CVR'].median()
                median_roas = camp_agg['ROAS'].median()
                
                scatter_fig = go.Figure()
                
                scatter_fig.add_trace(go.Scatter(
                    x=camp_agg['CVR'],
                    y=camp_agg['ROAS'],
                    mode='markers',
                    text=camp_agg['Campaign Name'],
                    marker=dict(
                        size=camp_agg['Orders'],
                        sizemode='area',
                        sizeref=2. * max(camp_agg['Orders']) / (40.**2) if not camp_agg.empty and max(camp_agg['Orders']) > 0 else 1, # Scaling
                        sizemin=4,
                        color=camp_agg['ROAS'],
                        colorscale=[[0, '#5B556F'], [1, '#22d3ee']], # Brand Purple to Cyan
                        showscale=False,
                        line=dict(color='rgba(255,255,255,0.2)', width=1) if chart_template == 'plotly_dark' else dict(color='black', width=1)
                    ),
                    hovertemplate="<b>%{text}</b><br>CVR: %{x:.2f}%<br>ROAS: %{y:.2f}x<br>Orders: %{marker.size}<extra></extra>"
                ))
                
                # Add Colored Dotted Lines (Medians)
                # Horizontal: Median ROAS (Red)
                scatter_fig.add_hline(
                    y=median_roas, 
                    line_dash="dot", 
                    line_color="#ef4444", # Red
                    line_width=2,
                    annotation_text="Med ROAS", 
                    annotation_position="top left",
                    annotation_font=dict(color="#ef4444")
                )
                
                # Vertical: Median CVR (Green)
                scatter_fig.add_vline(
                    x=median_cvr, 
                    line_dash="dot", 
                    line_color="#22c55e", # Green
                    line_width=2,
                    annotation_text="Med CVR", 
                    annotation_position="top right",
                    annotation_font=dict(color="#22c55e")
                )
                
                scatter_fig.update_layout(
                    title=dict(text="ROAS vs CVR (Size = Orders)", font=dict(color=text_color)),
                    xaxis=dict(title=dict(text="Conversion Rate (%)", font=dict(color=text_color)), showgrid=False, zeroline=True, tickfont=dict(color=text_color)),
                    yaxis=dict(title=dict(text="ROAS", font=dict(color=text_color)), showgrid=False, zeroline=True, tickfont=dict(color=text_color)),
                    template=chart_template,
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color),
                    height=450,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(scatter_fig, use_container_width=True)
            
        else:
            st.info("ℹ️ No 'Date' column found in report. Trend analysis unavailable.")

        st.markdown("---")

        # ==========================================
        # 3. Performance Breakdown
        # ==========================================
        st.markdown("### 📋 Performance Breakdown")
        
        view_by = st.selectbox("View By:", ["Match Type", "Campaign Name", "Category Breakdown", "Portfolio name", "Ad Group Name"])
        
        # Clean column name for grouping
        group_col = view_by
        
        # ---------------------------
        # CATEGORY DRILL DOWN LOGIC
        # ---------------------------
        if view_by == "Category Breakdown":
            # Check if we have category data
            cat_col = 'Category' if 'Category' in df.columns else None
            sub_col = 'Sub-Category' if 'Sub-Category' in df.columns else None
            sku_col = 'SKU_advertised' if 'SKU_advertised' in df.columns else ('SKU' if 'SKU' in df.columns else None)
            
            if not cat_col:
                # Check what's actually missing - session state OR database
                has_cat_in_session = st.session_state.get('unified_data', {}).get('upload_status', {}).get('category_mapping', False)
                has_cat_in_db = False
                has_adv_in_db = False
                
                try:
                    cat_from_db = db.get_category_mappings(client_id)
                    has_cat_in_db = cat_from_db is not None and not cat_from_db.empty
                    adv_from_db = db.get_advertised_product_map(client_id)
                    has_adv_in_db = adv_from_db is not None and not adv_from_db.empty
                except Exception:
                    pass
                
                # Smart Error Message
                if has_cat_in_session or has_cat_in_db:
                    if not has_adv_in_db:
                        st.warning("⚠️ **Missing Link**: 'Category Mapping' is active, but we can't link Campaigns to SKUs. Please upload the **Advertised Product Report**.")
                    else:
                        st.warning("⚠️ **Match Failure**: Category mapping and Advertised Product data are both loaded, but SKU matching failed. Check if SKUs in Category Map match the Advertised Product Report.")
                else:
                    st.warning("⚠️ 'Category' column not found. Please upload 'Category Mapping' file in Data Hub.")
                group_col = "Match Type" # Fallback
            else:
                # Drill Down Filters
                c1, c2, c3 = st.columns(3)
                
                # Level 1: Category Filter
                cats = ['All'] + sorted(df[cat_col].dropna().unique().tolist())
                sel_cat = c1.selectbox("Filter Category", cats)
                
                # Filter data based on selection
                if sel_cat != 'All':
                    df = df[df[cat_col] == sel_cat]
                    
                    # Level 2: Sub-Category Filter (dynamic options)
                    subcats = ['All'] + sorted(df[sub_col].dropna().unique().tolist()) if sub_col else ['All']
                    sel_sub = c2.selectbox("Filter Sub-Category", subcats)
                    
                    if sel_sub != 'All':
                        df = df[df[sub_col] == sel_sub]
                        
                        # Level 3: SKU Filter
                        skus = ['All'] + sorted(df[sku_col].dropna().unique().tolist()) if sku_col else ['All']
                        sel_sku = c3.selectbox("Filter SKU", skus)
                        
                        if sel_sku != 'All':
                            # Level 4: Show Campaign Performance
                            df = df[df[sku_col] == sel_sku]
                            group_col = "Campaign Name"
                            st.info(f"Showing Campaigns for SKU: {sel_sku}")
                        else:
                            # Show SKUs
                            group_col = sku_col if sku_col else "Campaign Name"
                            st.info(f"Showing SKUs in {sel_sub}")
                    else:
                        # Show Sub-Categories
                        group_col = sub_col if sub_col else "Campaign Name"
                        st.info(f"Showing Sub-Categories in {sel_cat}")
                else:
                    # Show Categories
                    group_col = cat_col
                    c2.selectbox("Filter Sub-Category", ["All"], disabled=True)
                    c3.selectbox("Filter SKU", ["All"], disabled=True)
        
        # Smart Switching: If 'Match Type' is selected, use our new 'Refined Match Type' column
        elif view_by == "Match Type" and "Refined Match Type" in df.columns:
            group_col = "Refined Match Type"

        if group_col not in df.columns:
            # Try to help user if column is missing (e.g. Portfolio might be missing)
            st.warning(f"⚠️ Column '{group_col}' not found in data. Switching to 'Match Type'.")
            group_col = "Match Type"
            
        # Group Data
        agg_cols = {
            'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum', 
            'Clicks': 'sum', 'Impressions': 'sum'
        }
        
        grouped = df.groupby(group_col).agg(agg_cols).reset_index()
        
        # Calc Metrics
        grouped['ACOS'] = np.where(grouped['Sales'] > 0, grouped['Spend'] / grouped['Sales'] * 100, 0)
        grouped['ROAS'] = np.where(grouped['Spend'] > 0, grouped['Sales'] / grouped['Spend'], 0)
        grouped['CTR'] = np.where(grouped['Impressions'] > 0, grouped['Clicks'] / grouped['Impressions'] * 100, 0)
        grouped['CVR'] = np.where(grouped['Clicks'] > 0, grouped['Orders'] / grouped['Clicks'] * 100, 0)
        grouped['CPC'] = np.where(grouped['Clicks'] > 0, grouped['Spend'] / grouped['Clicks'], 0)
        
        # Sort by Spend desc
        grouped = grouped.sort_values('Spend', ascending=False)
        
        # Formatting for Display
        display_df = grouped.copy()
        
        # Layout: Donut Chart (Left) + Table (Right)
        
        d_col1, d_col2 = st.columns([1, 2])
        
        with d_col1:
            # DONUT CHART
            if 'Sales' in grouped.columns and grouped['Sales'].sum() > 0:
                # Get dynamic template
                from ui.theme import ThemeManager
                chart_template = ThemeManager.get_chart_template()
                
                donut_fig = px.pie(
                    grouped, 
                    values='Sales', 
                    names=group_col, 
                    title=f"Sales by {view_by}",
                    hole=0.4,
                    color_discrete_sequence=['#5B556F', '#8F8CA3', '#22d3ee', '#334155', '#475569']
                )
                
                is_dark = st.session_state.get('theme_mode', 'dark') == 'dark'
                bg_color = 'rgba(0,0,0,0)'
                text_color = '#f3f4f6' if is_dark else '#1f2937'
                
                donut_fig.update_layout(
                    template=chart_template,
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color),
                    title=dict(font=dict(color=text_color)),
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False # Cleaner look, labels usually enough or hover
                )
                # Enable text info inside
                donut_fig.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(donut_fig, use_container_width=True)
            else:
                st.caption("No sales data available for chart.")
        
        with d_col2:
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    'Spend': st.column_config.NumberColumn(format="AED %.2f"),
                    'Sales': st.column_config.NumberColumn(format="AED %.2f"),
                    'CPC': st.column_config.NumberColumn(format="AED %.2f"),
                    'ACOS': st.column_config.NumberColumn(format="%.2f%%"),
                    'ROAS': st.column_config.NumberColumn(format="%.2fx"),
                    'CTR': st.column_config.NumberColumn(format="%.2f%%"),
                    'CVR': st.column_config.NumberColumn(format="%.2f%%"),
                },
                hide_index=True
            )
