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
from core.data_hub import DataHub
from core.data_loader import SmartMapper, safe_numeric

class PerformanceSnapshotModule(BaseFeature):
    """Performance Snapshot Dashboard."""
    
    def render_ui(self):
        """Render the dashboard UI."""
        # 1. Load Data FIRST
        from core.db_manager import get_db_manager
        
        db = get_db_manager(st.session_state.get('test_mode', False))
        
        # USE ACTIVE ACCOUNT from session state
        client_id = st.session_state.get('active_account_id', 'default_client')
        
        if not client_id:
            st.title("📊 Account Overview")
            st.error("⚠️ No account selected! Please select an account in the sidebar.")
            return
        
        # Fetch persistent data for this account
        db_data = db.get_target_stats_df(client_id)
        
        if db_data.empty:
            st.title("📊 Account Overview")
            st.warning(f"⚠️ No data found for account '{st.session_state.get('active_account_name', client_id)}'. Please upload a Search Term Report in the Data Hub.")
            return

        # 2. Enrichment Logic
        hub = DataHub()
        self.data = db_data
        
        # Merge SKU info if Advertised Product Report is loaded
        if hub.is_loaded('advertised_product_report'):
             adv_report = hub.get_data('advertised_product_report')
             if adv_report is not None:
                 agg_dict = {}
                 if 'SKU' in adv_report.columns:
                     agg_dict['SKU'] = lambda x: ', '.join(x.dropna().unique())
                 
                 if agg_dict and 'Campaign Name' in adv_report.columns and 'Ad Group Name' in adv_report.columns:
                     # 1. Normalize Keys for Robust Merge
                     self.data['Camp_Norm'] = self.data['Campaign Name'].astype(str).str.strip().str.lower()
                     self.data['AG_Norm'] = self.data['Ad Group Name'].astype(str).str.strip().str.lower()
                     
                     adv_report = adv_report.copy()
                     adv_report['Camp_Norm'] = adv_report['Campaign Name'].astype(str).str.strip().str.lower()
                     adv_report['AG_Norm'] = adv_report['Ad Group Name'].astype(str).str.strip().str.lower()
                     
                     # 2. Aggregation
                     sku_lookup = adv_report.groupby(['Camp_Norm', 'AG_Norm']).agg(agg_dict).reset_index()
                     sku_lookup.columns = ['Camp_Norm', 'AG_Norm', 'SKU_advertised']
                     
                     # 3. Merge
                     self.data = self.data.merge(
                         sku_lookup,
                         on=['Camp_Norm', 'AG_Norm'],
                         how='left'
                     )
                     
                     # Cleanup
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
        
        # Merge Category Mapping if loaded
        if hub.is_loaded('category_mapping'):
            cat_map = hub.get_data('category_mapping')
            print(f"🔍 Category Map loaded: {cat_map is not None}, rows: {len(cat_map) if cat_map is not None else 0}")
            
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
                    self.data['ID_List'] = self.data[product_id_col].astype(str).str.split(',')
                    
                    # Explode to handle multi-ID rows
                    exploded = self.data.explode('ID_List')
                    
                    # AGGRESSIVE normalization for matching
                    # Remove all non-alphanumeric characters, lowercase
                    exploded['ID_Clean'] = (
                        exploded['ID_List']
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r'[^a-z0-9]', '', regex=True)
                    )
                    
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
                    
                    # Find category columns
                    cat_cols_to_merge = ['ID_Clean']
                    if 'Category' in cat_map.columns:
                        cat_cols_to_merge.append('Category')
                    if 'Sub-Category' in cat_map.columns:
                        cat_cols_to_merge.append('Sub-Category')
                    
                    # Merge
                    merged = exploded.merge(
                        cat_map[cat_cols_to_merge], 
                        on='ID_Clean', 
                        how='left'
                    )
                    
                    # Check match rate
                    match_rate = merged['Category'].notna().sum() / len(merged) if 'Category' in merged.columns else 0
                    print(f"📊 Category Mapping Match Rate: {match_rate:.1%}")
                    
                    # De-duplicate back (take first match per original row)
                    # Group by original index and take first
                    first_match = merged.groupby(level=0).first()
                    
                    # Update self.data with category info
                    if 'Category' in first_match.columns:
                        self.data['Category'] = first_match['Category']
                    if 'Sub-Category' in first_match.columns:
                        self.data['Sub-Category'] = first_match['Sub-Category']
                    
                    # Cleanup
                    self.data.drop(columns=['ID_List'], inplace=True, errors='ignore')

        # --- DEBUG SECTION ---
        # help user diagnose missing data
        with st.expander("🛠 Debug Data Merge Stats"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total Rows", len(self.data))
            
            sku_found = self.data['SKU_advertised'].notna().sum() if 'SKU_advertised' in self.data.columns else 0
            with c2: st.metric("Rows with SKU Linked", f"{sku_found} ({sku_found/len(self.data)*100:.1f}%)")
            
            cat_found = self.data['Category'].notna().sum() if 'Category' in self.data.columns else 0
            with c3: st.metric("Rows with Category Linked", f"{cat_found} ({cat_found/len(self.data)*100:.1f}%)")
            
            # Detailed Bridge Debugging
            st.divider()
            st.markdown("#### 🕵️ Bridge Diagnostics")
            
            # 1. Advertised Product Report Status
            adv_loaded = hub.is_loaded('advertised_product_report')
            st.write(f"**Advertised Product Report Loaded:** {adv_loaded}")
            if adv_loaded:
                adv_df = hub.get_data('advertised_product_report')
                st.write(f"- Columns: {list(adv_df.columns)}")
                st.write(f"- Rows: {len(adv_df)}")
            
            # 2. Bulk File Status
            bulk_loaded = hub.is_loaded('bulk_id_mapping')
            st.write(f"**Bulk ID Mapping Loaded:** {bulk_loaded}")
            if bulk_loaded:
                bulk_df = hub.get_data('bulk_id_mapping')
                st.write(f"- Columns: {list(bulk_df.columns)}")
                st.write(f"- Rows: {len(bulk_df)}")
                
                # Check SKU Column detection
                sku_candidates = [c for c in bulk_df.columns if c.lower() in ['sku', 'msku', 'vendor sku', 'vendor_sku']]
                st.write(f"- Detected SKU Candidates: {sku_candidates}")
                
            # 3. Merge Sample
            st.write("**Merge Match Debugging**")
            if 'Campaign Name' in self.data.columns:
                unique_camps = self.data['Campaign Name'].dropna().unique()[:5]
                st.write(f"- Sample Campaigns in Report: {list(unique_camps)}")
            else:
                st.error("❌ 'Campaign Name' column MISSING in Main Data!")
                
            if 'SKU_advertised' in self.data.columns:
                 filled = self.data['SKU_advertised'].notna().mean()
                 st.write(f"- Final 'SKU_advertised' Fill Rate: {filled:.2%}")
            else:
                 st.write("- 'SKU_advertised' column NOT CREATED.")
            
            if cat_found == 0 and hub.is_loaded('category_mapping'):
                st.error("Merge Failed: No categories matched. Check if SKU column in Category Map matches SKUs in Advertised Product Report.")
                st.write("Sample Data SKUs:", self.data['SKU_advertised'].dropna().head(3).tolist() if 'SKU_advertised' in self.data.columns else "No SKUs")
        # ---------------------
        
        # 3. Compact Header Layout (Title + Date Picker)
        col_header, col_date = st.columns([8, 3])
        
        with col_header:
            import base64
            try:
                with open("assets/icons/dashboard.png", "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                st.markdown(f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
                    <h1 style="margin: 0; padding: 0; line-height: 1.2;">Account Overview</h1>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.title("Account Overview")
            

            
        self.date_filter = None
        if not self.data.empty:
            date_col_name = next((c for c in ['Date', 'date'] if c in self.data.columns), None)
            if date_col_name:
                try:
                    dates = pd.to_datetime(self.data[date_col_name], errors='coerce').dropna()
                    if not dates.empty:
                        min_d, max_d = dates.min().date(), dates.max().date()
                        
                        with col_date:
                            # Top margin to align with Title
                            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                            self.date_filter = st.date_input(
                                "📅 Date Range",
                                value=(min_d, max_d),
                                min_value=min_d,
                                max_value=max_d,
                                label_visibility="collapsed"
                            )
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
        
        # Ensure numeric
        numeric_cols = ['Spend', 'Sales', 'Impressions', 'Clicks', 'Orders']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = safe_numeric(df[col])
            else:
                df[col] = 0.0
                
        # Create derived metrics
        df['ROAS'] = np.where(df['Spend'] > 0, df['Sales'] / df['Spend'], 0)
        df['CPC'] = np.where(df['Clicks'] > 0, df['Spend'] / df['Clicks'], 0)
        df['CTR'] = np.where(df['Impressions'] > 0, df['Clicks'] / df['Impressions'] * 100, 0)
        df['CVR'] = np.where(df['Clicks'] > 0, df['Orders'] / df['Clicks'] * 100, 0)
        df['ACOS'] = np.where(df['Sales'] > 0, df['Spend'] / df['Sales'] * 100, 0)
        
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

        # Helper to classify based on Targeting
        def refine_match_type(row):
            mt = str(row['Refined Match Type']).upper()
            # Check Targeting AND TargetingExpression if available
            targeting = str(row.get('Targeting', '')).lower()
            # If Targeting starts with 'asin=' or 'category=', use that
            
            # 1. Trust Strong Types
            if mt in ['EXACT', 'BROAD', 'PHRASE', 'PT', 'CATEGORY', 'AUTO']:
                return mt
            
            # 2. Heuristics on Targeting Text
            if 'asin=' in targeting or (len(targeting) == 10 and targeting.startswith('b0')):
                return 'PT'
            if 'category=' in targeting:
                return 'CATEGORY'
            
            # 3. Auto targeting keywords
            auto_keywords = ['close-match', 'loose-match', 'substitutes', 'complements', '*']
            if any(k in targeting for k in auto_keywords):
                return 'AUTO'
            
            # 4. Fallback
            return 'OTHER' if mt in ['-', 'NAN', 'NONE'] else mt

        # Apply refinement
        # Ensure Targeting column exists
        if 'Targeting' in df.columns and not df.empty:
            df['Refined Match Type'] = df.apply(refine_match_type, axis=1)
        
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
            roas = sales / spend if spend > 0 else 0
            acos = (spend / sales * 100) if sales > 0 else 0
            return {'spend': spend, 'sales': sales, 'orders': orders, 'roas': roas, 'acos': acos}

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
        
        # Calculate comparison deltas (Default to 7-day change)
        # Check if user wants 7D or 14D
        comp_days = st.radio("Trend Comparison Period", [7, 14], index=0, horizontal=True, label_visibility="collapsed")
        st.caption(f"Showing comparison for Last {comp_days} Days vs. Previous {comp_days} Days")
        
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
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("Spend", f"AED {total_spend:,.0f}", delta=deltas.get('spend'))
        with c2: metric_card("Revenue", f"AED {total_sales:,.0f}", delta=deltas.get('sales'))
        with c3: metric_card("ACOS", f"{total_acos:.2f}%", delta=deltas.get('acos'))
        with c4: metric_card("ROAS", f"{total_roas:.2f}x", delta=deltas.get('roas'))
        with c5: metric_card("Orders", f"{total_orders:,.0f}", delta=deltas.get('orders'))
        
        c6, c7, c8, c9, c10 = st.columns(5)
        with c6: metric_card("Impressions", f"{total_impr:,.0f}")
        with c7: metric_card("Clicks", f"{total_clicks:,.0f}")
        with c8: metric_card("CTR", f"{total_ctr:.2f}%")
        with c9: metric_card("CPC", f"AED {total_cpc:.2f}")
        with c10: metric_card("Conv. Rate", f"{total_cvr:.2f}%")
        
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
                    marker_color='#6366f1'  # Indigo
                ))
                
                # Line Chart (Dual Axis)
                fig.add_trace(go.Scatter(
                    x=resampled['Date'], 
                    y=resampled[metric_line], 
                    name=metric_line,
                    yaxis='y2',
                    line=dict(color='#f97316', width=3) # Orange
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
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(color='white', width=1) if chart_template == 'plotly_dark' else dict(color='black', width=1)
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
                # Smart Error Message
                if st.session_state.get('unified_data', {}).get('upload_status', {}).get('category_mapping', False):
                     st.warning("⚠️ **Missing Link**: 'Category Mapping' is active, but we can't link Campaigns to SKUs. Please upload the **Advertised Product Report**.")
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
                    color_discrete_sequence=px.colors.qualitative.Prism
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
