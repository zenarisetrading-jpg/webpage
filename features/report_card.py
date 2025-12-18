
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Core imports
from features._base import BaseFeature
from core.data_hub import DataHub
from ui.components import metric_card
from utils.formatters import format_currency, format_percentage

class ReportCardModule(BaseFeature):
    """
    Modern, minimal 'Report Card' view summarizing optimization health.
    Features:
    - 3-4 gauges for health metrics
    - Action/Result counters
    - AI Summary (Isolated)
    - PDF Export
    """
    
    
    def render_ui(self):
        """Render the feature's user interface."""
        st.title("Optimization Report Card")
        
    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data has required columns."""
        required = ['Spend', 'Sales', 'Orders']
        missing = [c for c in required if c not in data.columns]
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        return True, ""
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and compute metrics."""
        return self._compute_metrics(data)

    def display_results(self, metrics: Dict[str, Any]):
        """Render the Report Card view."""
        # 3. Render UI Sections
        self._render_section_1_health(metrics)
        st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        self._render_section_2_actions(metrics)
        st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        self._render_section_3_ai_summary(metrics)
        
        st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # Print Mode Instructions
        st.info("📸 **To export:** Press `Cmd+P` (Mac) or `Ctrl+P` (Windows) → Save as PDF. For best results, print in **landscape mode**.")

    
    def run(self):
        """Custom run to handle data loading explicitly if needed, or rely on BaseFeature."""
        # We override run to fetch data from DataHub explicitly first, then call super's logic if we wanted, 
        # BUT BaseFeature expects self.data to be set.
        
        hub = DataHub()
        df = hub.get_enriched_data()
        if df is None:
             df = hub.get_data("search_term_report")
             
        if df is None:
            self.render_ui()
            st.warning("⚠️ No data available. Please upload a Search Term Report first.")
            return

        self.data = df
        
        # Now we can just call the manual steps or rely on BaseFeature's orchestration logic if we copied it.
        # However, to be safe and simple, I will just call the methods directly as my original run did, 
        # but now I have the abstract methods implemented to satisfy the ABC check in tests.
        
        self.render_ui()
        is_valid, msg = self.validate_data(self.data)
        if not is_valid:
            st.error(msg)
            return
            
        metrics = self.analyze(self.data)
        self.display_results(metrics)
        
    
    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute all report card metrics from data."""
        
        # 1. Performance Health
        total_spend = df['Spend'].sum()
        total_sales = df['Sales'].sum()
        actual_roas = total_spend / total_sales if total_sales > 0 else 0
        # Invert for ROAS (Sales/Spend) as commonly used in Amazon PPC, user code used Spend/Sales? 
        # Wait, usually ROAS = Sales / Spend. ACOS = Spend / Sales.
        # User's previous code: "actual_roas = total_sales / total_spend if total_spend > 0 else 0"
        # Let's stick to standard ROAS = Sales/Spend.
        actual_roas = total_sales / total_spend if total_spend > 0 else 0
        
        target_roas = st.session_state.get('target_roas', 3.0) # Default target
        
        # Spend Quality (Spend on terms with > 0 orders)
        if 'Orders' in df.columns:
            converting_spend = df[df['Orders'] > 0]['Spend'].sum()
        elif 'Sales' in df.columns:
            converting_spend = df[df['Sales'] > 0]['Spend'].sum()
        else:
            converting_spend = 0
            
        spend_quality_score = (converting_spend / total_spend * 100) if total_spend > 0 else 0
        
        # Efficiency Health (ROAS vs Target)
        efficiency_health = (actual_roas / target_roas * 100) if target_roas > 0 else 0
        
        # Optimization Coverage Health (% of eligible targets adjusted)
        # Will be computed after we have action counts - placeholder for now
        optimization_coverage = 0.0  # Will be updated below after actions are counted
        
        # 2. Optimization Actions (Counts)
        # We need to fetch the latest optimizer run results if available
        actions = {'bid_increases': 0, 'bid_decreases': 0, 'negatives': 0, 'harvests': 0}
        
        if 'latest_optimizer_run' in st.session_state:
            res = st.session_state['latest_optimizer_run']
            
            # --- Bids ---
            direct_bids = res.get('direct_bids', pd.DataFrame())
            agg_bids = res.get('agg_bids', pd.DataFrame())
            
            if direct_bids.empty and agg_bids.empty:
                 all_bids = pd.DataFrame()
            else:
                 all_bids = pd.concat([direct_bids, agg_bids], ignore_index=True)
            
            bid_removed_val = 0
            bid_added_val = 0
            
            if not all_bids.empty and 'New Bid' in all_bids.columns:
                bid_col = 'Current Bid' if 'Current Bid' in all_bids.columns else 'CPC'
                if bid_col not in all_bids.columns: all_bids[bid_col] = 0

                # Ensure numeric
                all_bids['New Bid'] = pd.to_numeric(all_bids['New Bid'], errors='coerce').fillna(0)
                all_bids[bid_col] = pd.to_numeric(all_bids[bid_col], errors='coerce').fillna(0)
                if 'Clicks' not in all_bids.columns: all_bids['Clicks'] = 0
                all_bids['Clicks'] = pd.to_numeric(all_bids['Clicks'], errors='coerce').fillna(0)
                
                if 'Sales' not in all_bids.columns: all_bids['Sales'] = 0
                all_bids['Sales'] = pd.to_numeric(all_bids['Sales'], errors='coerce').fillna(0)
                
                if 'Spend' not in all_bids.columns: all_bids['Spend'] = 0
                all_bids['Spend'] = pd.to_numeric(all_bids['Spend'], errors='coerce').fillna(0)

                # Backfill Sales from ROAS if Sales is 0 (Fix for zero revenue impact)
                if 'ROAS' in all_bids.columns:
                     all_bids['ROAS'] = pd.to_numeric(all_bids['ROAS'], errors='coerce').fillna(0)
                     mask = (all_bids['Sales'] == 0) & (all_bids['ROAS'] > 0)
                     if mask.any():
                         all_bids.loc[mask, 'Sales'] = all_bids.loc[mask, 'ROAS'] * all_bids.loc[mask, 'Spend']
                
                # Determine ROAS for efficiency filter
                # If ROAS col exists use it, else calc
                if 'ROAS' not in all_bids.columns:
                    if 'Sales' in all_bids.columns and 'Spend' in all_bids.columns:
                         all_bids['ROAS'] = all_bids['Sales'] / all_bids['Spend'].replace(0, 1)
                    else:
                         all_bids['ROAS'] = 0.0 # Unknown fallback
                
                # Increases (High Efficiency Only) -> ADDED
                # High Eff: ROAS > Target * 1.2
                high_eff_threshold = target_roas * 1.2
                # Increases (High Efficiency Only) -> ADDED
                # High Eff: ROAS > Target * 1.2
                high_eff_threshold = target_roas * 1.2
                row_increases = all_bids[
                    (all_bids['New Bid'] > all_bids[bid_col]) & 
                    (all_bids['ROAS'] > high_eff_threshold)
                ].copy()
                
                # Revenue Impact Proxy: Current Sales * ((New Bid / Current Bid) - 1) * 0.8 (conservative elasticity)
                row_increases['RevImpact'] = row_increases['Sales'] * ((row_increases['New Bid'] / row_increases[bid_col].replace(0, 1)) - 1) * 0.8
                row_increases['Investment'] = (row_increases['New Bid'] - row_increases[bid_col]) * row_increases['Clicks']
                
                bid_added_val = row_increases['Investment'].sum()
                actions['bid_increases'] = len(all_bids[all_bids['New Bid'] > all_bids[bid_col]])

                # Decreases (Low Efficiency Only) -> REMOVED
                # Low Eff: ROAS < Target * 0.8
                low_eff_threshold = target_roas * 0.8
                row_decreases = all_bids[
                    (all_bids['New Bid'] < all_bids[bid_col]) & 
                    (all_bids['ROAS'] < low_eff_threshold)
                ].copy()
                row_decreases['Savings'] = (row_decreases[bid_col] - row_decreases['New Bid']) * row_decreases['Clicks']
                bid_removed_val = row_decreases['Savings'].sum()
                actions['bid_decreases'] = len(all_bids[all_bids['New Bid'] < all_bids[bid_col]])

            else:
                 actions['bid_increases'] = 0
                 actions['bid_decreases'] = 0
                 row_increases = pd.DataFrame()
                 row_decreases = pd.DataFrame()

            # --- Negatives (All are assumed Low Efficiency/Waste) ---
            neg_kw = res.get('neg_kw', pd.DataFrame())
            neg_pt = res.get('neg_pt', pd.DataFrame())
            actions['negatives'] = len(neg_kw) + len(neg_pt)
            
            neg_spend_val = 0
            neg_items = []
            
            def get_camp(r):
                return r.get('Campaign Name', r.get('Campaign', 'Unknown Campaign'))
                
            if not neg_kw.empty and 'Spend' in neg_kw.columns: 
                neg_spend_val += neg_kw['Spend'].sum()
                for _, row in neg_kw.iterrows():
                    neg_items.append({
                        'name': f"{row.get('Customer Search Term', 'Target')}",
                        'camp': get_camp(row),
                        'val': row['Spend'],
                        'type': 'Negative'
                    })
            if not neg_pt.empty and 'Spend' in neg_pt.columns: 
                neg_spend_val += neg_pt['Spend'].sum()
                for _, row in neg_pt.iterrows():
                    neg_items.append({
                        'name': f"{row.get('Targeting', 'Target')}",
                        'camp': get_camp(row),
                        'val': row['Spend'],
                        'type': 'Negative'
                    })
            
            # --- Harvests (Assumed High Efficiency/Growth) ---
            harvest = res.get('harvest', pd.DataFrame())
            actions['harvests'] = len(harvest)
            harvest_added_val = actions['harvests'] * 50.0 # Spend assumption kept for total metric
            
            harvest_items = []
            if not harvest.empty:
                for _, row in harvest.iterrows():
                    # Estimate Revenue for new keywords: Target ROAS * Est Spend (assume 50 AED)
                    est_rev = 50.0 * target_roas
                    harvest_items.append({
                        'name': f"{row.get('Customer Search Term', 'Keyword')}",
                        'camp': 'New Campaign', # Usually harvested to new or existing
                        'val': est_rev,
                        'type': 'Harvest'
                    })

        else:
            bid_removed_val = 0
            bid_added_val = 0
            neg_spend_val = 0
            harvest_added_val = 0
            row_increases = pd.DataFrame()
            row_decreases = pd.DataFrame()
            neg_items = []
            harvest_items = []

        # --- Top Details Construction ---
        # Removed: Negatives + Bid Decreases (Savings)
        removed_list = neg_items
        if not row_decreases.empty:
            for _, row in row_decreases.iterrows():
                removed_list.append({
                    'name': f"{row.get('Targeting', 'Target')}",
                    'camp': get_camp(row),
                    'val': row['Savings'],
                    'type': 'Bid Decrease'
                })
        
        # Sort Removed (Descending Savings)
        removed_list = sorted(removed_list, key=lambda x: x['val'], reverse=True)[:5]

        # Added: Bid Increases ONLY (User requested to exclude static harvests)
        added_list = []
        if not row_increases.empty:
            for _, row in row_increases.iterrows():
                added_list.append({
                    'name': f"{row.get('Targeting', 'Target')}",
                    'camp': get_camp(row),
                    'val': row['RevImpact'],
                    'type': 'Bid Increase'
                })
        
        # Sort Added (Descending Revenue Impact)
        added_list = sorted(added_list, key=lambda x: x['val'], reverse=True)[:5]

        # Total metrics
        total_removed = bid_removed_val + neg_spend_val
        total_added = bid_added_val + harvest_added_val
        
        # Financials (Legacy for metric display)
        est_savings = total_removed
        
        # Reallocation Percentages
        # Denominator: Total Spend Previous Cycle
        realloc_denom = total_spend if total_spend > 0 else 1
        
        removed_pct = (total_removed / realloc_denom) * 100
        added_pct = (total_added / realloc_denom) * 100
        
        # 3. Budget Allocation Buckets (Legacy calc kept for safety but unused in new chart)
        # Logic: 
        # Low = ROAS < 0.8 * Target (or 0 Orders)
        # Mid = 0.8 * Target <= ROAS <= 1.2 * Target
        # High = ROAS > 1.2 * Target
        
        def classify_spend(row):
            s = row['Spend']
            r = row['Sales'] / s if s > 0 else 0
            if r == 0 or r < (target_roas * 0.8):
                return 'Low'
            elif r >= (target_roas * 0.8) and r <= (target_roas * 1.2):
                return 'Mid'
            else:
                return 'High'
        
        # Apply classification
        # We need a copy to not mutate original df
        temp_df = df.copy()
        if 'Spend' in temp_df.columns and 'Sales' in temp_df.columns:
            temp_df['Bucket'] = temp_df.apply(classify_spend, axis=1)
            bucket_spend = temp_df.groupby('Bucket')['Spend'].sum().to_dict()
        else:
            bucket_spend = {'Low': 0, 'Mid': 0, 'High': 0}

        low_spend = bucket_spend.get('Low', 0)
        mid_spend = bucket_spend.get('Mid', 0)
        high_spend = bucket_spend.get('High', 0)
        
        # Projection (After Optimization)
        # Est Savings = Negatives Spend + Bid Reduction Savings
        est_savings_old = neg_spend_val + bid_removed_val # Renamed to avoid conflict
        
        est_growth = actions['harvests'] * 50.0   # Avg spend growth per harvest
        
        # Shift logic:
        # After Low = Low - Savings (min 0)
        # After Mid = Mid (assume stable)
        # After High = High + Growth
        
        low_spend_after = max(0, low_spend - est_savings_old)
        mid_spend_after = mid_spend 
        high_spend_after = high_spend + est_growth
        
        # Normalize to %
        total_before = low_spend + mid_spend + high_spend
        total_after = low_spend_after + mid_spend_after + high_spend_after
        
        def safe_pct(val, tot):
            return (val / tot * 100) if tot > 0 else 0
        
        allocation = {
            'before': {
                'Low': safe_pct(low_spend, total_before),
                'Mid': safe_pct(mid_spend, total_before),
                'High': safe_pct(high_spend, total_before)
            },
            'after': {
                'Low': safe_pct(low_spend_after, total_after),
                'Mid': safe_pct(mid_spend_after, total_after),
                'High': safe_pct(high_spend_after, total_after)
            }
        }

        
        # Calculate totals for percentages (Moved up for dependencies)
        total_targets = df['Targeting'].nunique() if 'Targeting' in df.columns else len(df)
        total_terms = df['Customer Search Term'].nunique() if 'Customer Search Term' in df.columns else len(df)

        # Optimization Coverage Health (% of eligible targets adjusted this cycle)
        # Adjusted = unique targets with at least one action (bid change, pause, or promotion)
        # For simplicity, we sum unique actions since each action is on a unique target
        adjusted_targets = actions['bid_increases'] + actions['bid_decreases'] + actions['negatives'] + actions['harvests']
        
        # Eligible targets = total unique targets in the data
        eligible_targets = total_targets if total_targets > 0 else 1
        
        # Calculate optimization coverage (clamped 0-100)
        optimization_coverage = min(100, max(0, (adjusted_targets / eligible_targets) * 100))
        
        return {
            "roas": actual_roas,
            "target_roas": target_roas,
            "spend_quality": spend_quality_score,
            "efficiency_health": efficiency_health,
            "optimization_coverage": optimization_coverage,
            "actions": actions,
            "counts": {
                "targets": total_targets,
                "terms": total_terms
            },
            "reallocation": {
                "removed_pct": removed_pct,
                "added_pct": added_pct
            },
            "details": {
                "removed": removed_list,
                "added": added_list
            },
            "financials": {
                "savings": est_savings,
                "growth": harvest_added_val
            },
            "total_spend": total_spend,
            "total_sales": total_sales
        }


    def _create_gauge(self, value: float, title: str, min_val=0, max_val=100, suffix="%", target_val=None) -> Any:
        import plotly.graph_objects as go
        
        mode = "gauge+number"
        delta_config = {}
        
        if target_val is not None: # Changed from !== to is not None
            mode = "gauge+number+delta"
            delta_config = {'reference': target_val, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}}

        # Traffic Light Colors
        # Standard: Red -> Yellow -> Green
        c_low = "#ef4444"   # Red
        c_mid = "#eab308"   # Yellow
        c_high = "#22c55e"  # Green
        
        # Invert for Risk metrics (where High is Bad)
        if "Risk" in title:
            c_low = "#22c55e"   # Green (Low Risk)
            c_mid = "#eab308"   # Yellow
            c_high = "#ef4444"  # Red (High Risk)

        steps_config = [
            {'range': [min_val, min_val + (max_val-min_val)*0.4], 'color': c_low},       # 0-40%
            {'range': [min_val + (max_val-min_val)*0.4, min_val + (max_val-min_val)*0.75], 'color': c_mid}, # 40-75%
            {'range': [min_val + (max_val-min_val)*0.75, max_val], 'color': c_high}     # 75-100%
        ]

        fig = go.Figure(go.Indicator(
            mode = mode,
            value = value,
            delta = delta_config,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': suffix, 'font': {'size': 20}},
            title = {'text': title, 'font': {'size': 14, 'color': "gray"}},
            gauge = {
                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"}, # Transparent bar, rely on background steps or needle
                # Note: Plotly gauge 'bar' is the progress bar. If we want standard sections, we usually make bar transparent or black needle.
                # However, user asked for traffic colors. Usually that means the background bands are colored.
                # Let's use the steps for bands and a dark needle/bar indicator.
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': steps_config,
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        # Add a needle using a workaround or just keep the 'bar' (progress) as black line?
        # Plotly 'bar' covers the steps. If we want to see steps, bar should be thin or transparent.
        # But 'value' needs to be shown.
        # Let's stick to standard gauge where 'steps' are the background track and 'bar' is the fill.
        # Wait, traffic light usually implies the TRACK is colored R/Y/G and the NEEDLE points to it.
        # Or the FILL matches the color of the zone it's in.
        # Simpler interpretation: The background segments are R/Y/G. The Bar is dark or transparent.
        # Let's make the Bar black to act as a pointer/fill.
        
        fig.update_traces(gauge_bar_color='rgba(30, 41, 59, 0.7)') # Slate-800 semi-transparent for the "Value" fill

        fig.update_layout(
            height=130, 
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor = "rgba(0,0,0,0)",
            font={'family': "Inter, sans-serif"}
        )
        return fig

    def _render_section_1_health(self, metrics: Dict[str, Any]):
        """Render top section with semi-circular gauges."""
        st.markdown("<h3 style='font-family: Inter, sans-serif; font-weight: 600; margin-bottom: 5px;'>Performance Snapshot</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 0; margin-bottom: 10px; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        
        def render_gauge_block(col, value, title, min_v, max_v, suffix, micro_label, tooltip_text, target=None):
            with col:
                fig = self._create_gauge(value, title, min_v, max_v, suffix, target_val=target)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Micro-label
                st.markdown(
                    f"<div style='text-align: center; font-size: 13px; font-weight: 600; color: #cbd5e1; margin-top: -10px;'>{micro_label}</div>",
                    unsafe_allow_html=True,
                    help=tooltip_text
                )

        render_gauge_block(
            c1, metrics['roas'], "ROAS vs Target", 0, metrics['target_roas']*2, "x",
            "Actual ROAS compared to target threshold",
            f"Target: {metrics['target_roas']}x. Return on Ad Spend (ROAS) compared to your strategic target."
        )
        
        render_gauge_block(
            c2, metrics['spend_quality'], "Spend Efficiency", 0, 100, "%",
            "% of spend on high-efficiency targets",
            "Percentage of total spend allocated to search terms that have generated at least 1 order."
        )
        
        # Optimization Coverage Health - Custom color zones
        # 0-3% red, 3-8% yellow, 8-15% green, >15% red
        coverage_val = metrics['optimization_coverage']
        with c3:
            fig = self._create_gauge(coverage_val, "Coverage Health", 0, 20, "%")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Determine color based on coverage zones
            if coverage_val <= 3:
                zone_color = "#ef4444"  # Red - Under-active
            elif coverage_val <= 8:
                zone_color = "#eab308"  # Yellow - Selective/healthy
            elif coverage_val <= 15:
                zone_color = "#22c55e"  # Green - Active but controlled
            else:
                zone_color = "#ef4444"  # Red - Over-tuning risk
            
            st.markdown(
                f"<div style='text-align: center; font-size: 13px; font-weight: 600; color: {zone_color}; margin-top: -10px;'>% of eligible targets adjusted this cycle</div>",
                unsafe_allow_html=True,
                help="Shows how actively the system adjusted the account. Calculated as % of eligible targets that received at least one action (bid change, pause, or promotion).\n\nColor Zones:\n🔴 0–3% → Under-active\n🟡 3–8% → Selective / healthy\n🟢 8–15% → Active but controlled\n🔴 >15% → Over-tuning risk"
            )
            
        render_gauge_block(
            c4, 100 - metrics['spend_quality'], "Spend Risk", 0, 100, "%",
            "% of spend below efficiency threshold",
            "Percentage of spend going to terms with 0 orders (Bleeders)."
        )
    

    def _create_reallocation_chart(self, reallocation: Dict[str, float]):
        import plotly.graph_objects as go
        
        removed = reallocation['removed_pct']
        added = reallocation['added_pct']
        
        fig = go.Figure()
        
        # Max range for dynamic scaling
        max_val = max(5, removed, added) * 1.3
        
        # --- Left Side (Amber) ---
        # Bar
        fig.add_trace(go.Bar(
            name='Reduced',
            y=['Spend Flow'], 
            x=[-removed],
            orientation='h',
            marker=dict(
                color='rgba(234, 179, 8, 0.8)', # Amber-500 with opacity
                line=dict(color='rgba(234, 179, 8, 1.0)', width=2)
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Arrow Shape (Left) - Pointing Left (Removal)
        # We can't easily put shapes inside bars dynamically without complex coord math, 
        # but we can place an annotation with an arrow.
        
        # Label Badge (Left)
        if removed > 0:
            fig.add_annotation(
                x=-removed/2 if removed > 2 else -removed, 
                y=0, 
                yshift=45, # Shifted higher to be clearly above
                text=f"-{removed:.1f}%",
                showarrow=False,
                bgcolor="#1e293b", # Slate-800
                bordercolor="rgba(234, 179, 8, 0.5)",
                borderwidth=1,
                font=dict(color="#fbbf24", size=20, family="Inter, sans-serif", weight="bold"), # Larger modern font
                height=35,
                width=90
            )

        # --- Right Side (Green) ---
        # Bar
        fig.add_trace(go.Bar(
            name='Added',
            y=['Spend Flow'], 
            x=[added],
            orientation='h',
            marker=dict(
                color='rgba(34, 197, 94, 0.8)', # Green-500
                line=dict(color='rgba(34, 197, 94, 1.0)', width=2)
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Label Badge (Right)
        if added > 0:
            fig.add_annotation(
                x=added/2 if added > 2 else added,
                y=0,
                yshift=45, # Shifted higher
                text=f"+{added:.1f}%",
                showarrow=False,
                bgcolor="#1e293b",
                bordercolor="rgba(34, 197, 94, 0.5)",
                borderwidth=1,
                font=dict(color="#4ade80", size=20, family="Inter, sans-serif", weight="bold"), 
                height=35,
                width=90
            )

        # --- Center Line & Axis ---
        fig.add_vline(x=0, line_width=1, line_color="#94a3b8", opacity=0.5) # Slate-400
        
        # Bottom Annotations (Context)
        fig.add_annotation(
            x=-max_val/2, y=-0.8, # Positioned below bar
            text="Inefficient Spend Removed",
            showarrow=False,
            font=dict(color="#fbbf24", size=12, family="Inter, sans-serif")
        )
        fig.add_annotation(
            x=max_val/2, y=-0.8,
            text="Invested in Growth",
            showarrow=False,
            font=dict(color="#4ade80", size=12, family="Inter, sans-serif")
        )
        
        # Central "0"
        fig.add_annotation(
            x=0, y=-0.5,
            text="0",
            showarrow=False,
            font=dict(color="white", size=14, family="Inter, sans-serif"),
            bgcolor="#0f172a" # Match bg to hide line overlap if needed
        )

        fig.update_layout(
            barmode='relative',
            height=200, # Increased height to accommodate higher badges
            margin=dict(l=10, r=10, t=60, b=40), # Increased top margin
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-max_val, max_val]
            ),
            yaxis=dict(showgrid=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(family="Inter, sans-serif")
        )
        return fig

    def _render_section_2_actions(self, metrics: Dict[str, Any]):
        """Render middle section: Actions & Results with visual charts."""
        st.markdown("<h3 style='font-family: Inter, sans-serif; font-weight: 600; margin-bottom: 5px;'>Actions & Results</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 0; margin-bottom: 10px; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        actions = metrics['actions']
        counts = metrics['counts']
        realloc = metrics['reallocation']
        details = metrics.get('details', {'removed': [], 'added': []})
        fin = metrics['financials']
        
        # Helper to format string
        def fmt_stat(val, total, context):
            pct = (val / total * 100) if total > 0 else 0
            return f"**{val}** <span style='color: #94a3b8; font-weight: 400; font-size: 14px;'>({pct:.1f}% of {context})</span>"
            
        # Row 1: Stats & Chart
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family: Inter, sans-serif; font-size: 17px; font-weight: 700; color: #cbd5e1; margin-bottom: 10px;'>System Executed Adjustments</div>", unsafe_allow_html=True)
            st.markdown(f"⬆️ Bid Increases: &nbsp; {fmt_stat(actions['bid_increases'], counts['targets'], 'targets')}", unsafe_allow_html=True)
            st.markdown(f"⬇️ Bid Decreases: &nbsp; {fmt_stat(actions['bid_decreases'], counts['targets'], 'targets')}", unsafe_allow_html=True)
            st.markdown(f"⏸️ Paused Targets: &nbsp; {fmt_stat(actions['negatives'], counts['terms'], 'terms')}", unsafe_allow_html=True)
            st.markdown(f"⭐ Promoted Keywords: &nbsp; {fmt_stat(actions['harvests'], counts['terms'], 'terms')}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div style='font-family: Inter, sans-serif; font-size: 17px; font-weight: 700; color: #cbd5e1; margin-bottom: 5px;'>Net Spend Reallocation (This Cycle)</div>", unsafe_allow_html=True)
            st.caption("Directional spend movement driven by optimization actions")
            fig = self._create_reallocation_chart(realloc)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown("---")
        
        # Row 2: Financials & Lists
        r2_c1, r2_c2 = st.columns([1, 1.5])
        
        with r2_c1:
            # Financial Impact aligned with lists
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family: Inter, sans-serif; font-size: 17px; font-weight: 700; color: #cbd5e1; margin-bottom: 5px;'>💰 Spend Preserved</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-family: Inter, sans-serif; font-size: 28px; font-weight: 700; color: #22c55e;'>AED {fin['savings']:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12px; color: #4ade80;'>↑ Annualized Savings</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with r2_c2:
             # Top Contributors Lists
            k1, k2 = st.columns(2)
            
            # Styles
            item_style = "font-family: 'Inter', sans-serif; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px;"
            camp_style = "display: block; font-size: 11px; color: #64748b; margin-top: 2px;"
            name_style = "font-weight: 500; color: #e2e8f0;"
            
            with k1:
                st.markdown("<div style='font-family: Inter, sans-serif; font-size: 17px; font-weight: 700; color: #cbd5e1; margin-bottom: 10px;'>Top Sources of Waste Removed</div>", unsafe_allow_html=True)
                if details['removed']:
                    for item in details['removed']:
                        val_str = f"AED {item['val']:,.0f}"
                        icon = "⛔" if item['type'] == 'Negative' else "⬇️"
                        
                        html = f"""
                        <div style="{item_style}">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="{name_style}">{icon} {item['name']}</span>
                                <span style="color: #fbbf24; font-weight: 600;">{val_str}</span>
                            </div>
                            <span style="{camp_style}">{item['camp']}</span>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                else:
                    st.caption("No significant removal actions.")

            with k2:
                st.markdown("<div style='font-family: Inter, sans-serif; font-size: 17px; font-weight: 700; color: #cbd5e1; margin-bottom: 10px;'>Top Investments in Growth</div>", unsafe_allow_html=True)
                if details['added']:
                    for item in details['added']:
                        val_str = f"AED {item['val']:,.0f}" # Revenue Potential
                        icon = "✨" if item['type'] == 'Harvest' else "⬆️"
                        
                        html = f"""
                        <div style="{item_style}">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="{name_style}">{icon} {item['name']}</span>
                                <span style="color: #4ade80; font-weight: 600;">Est. Rev: {val_str}</span>
                            </div>
                             <span style="{camp_style}">{item['camp']}</span>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                else:
                    st.caption("No significant investment actions.")
            
            st.caption("Representative contributors to this cycle’s reallocation")

    def _render_section_3_ai_summary(self, metrics: Dict[str, Any]):
        """Render bottom section: Isolated AI Summary."""
        
        st.markdown("<h3 style='font-family: Inter, sans-serif; font-weight: 600; margin-bottom: 5px;'>🧠 Zenny's Insight Summary</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 0; margin-bottom: 10px; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # Check for existing summary in session state to avoid re-generating
        if 'report_card_ai_summary' not in st.session_state:
            st.info("Click to get Zenny's interpretation of these results.")
            if st.button("✨ Zenny's POV"):
                with st.spinner("Zenny is analyzing optimization patterns..."):
                    summary = self._generate_ai_insight(metrics)
                    st.session_state['report_card_ai_summary'] = summary
                    st.rerun()
        else:
            st.markdown(st.session_state['report_card_ai_summary'])
            if st.button("Regenerate"):
                del st.session_state['report_card_ai_summary']
                st.rerun()
            
    def _generate_ai_insight(self, metrics: Dict[str, Any]) -> str:
        """
        Isolated AI generation. 
        Strictly summarizes metrics. DOES NOT execute tools.
        """
        try:
            import requests
            import json
            
            # Fetch API Key
            api_key = None
            if hasattr(st, "secrets"):
                try: api_key = st.secrets["OPENAI_API_KEY"]
                except: pass
                
            if not api_key:
                return "⚠️ AI Configuration Missing: API Key not found."

            # Construct Prompt
            system_prompt = """
            You are Zenny, a sharp PPC analyst who speaks like a trusted advisor, not a textbook.

            Generate exactly 3 insights. Each insight should be:
            - 2-3 sentences max (40-60 words)
            - Start with a bold header like "**ROAS Performance:**"
            - First sentence: State what the data shows with a specific number
            - Second sentence: What it means for the business (the "so what")
            
            Tone:
            - Confident and conversational, like a colleague in a meeting
            - No jargon, no filler phrases like "This indicates that..."
            - Direct and punchy, every word earns its place
            
            Example style:
            "**Spend Efficiency:** Only 28% of spend is going to converting targets—most budget is bleeding on zero-order terms. Tightening targeting or pausing underperformers could recover significant waste."
            """
            
            user_content = f"""
            Metrics:
            - ROAS: {metrics['roas']:.2f} (Target: {metrics['target_roas']})
            - Efficiency Score: {metrics['efficiency_health']:.1f}%
            - Spend Quality: {metrics['spend_quality']:.1f}%
            - Optimization Coverage: {metrics['optimization_coverage']:.1f}% (% of targets adjusted)
            - Spend Removed: {metrics['reallocation']['removed_pct']:.1f}%
            - Spend Added: {metrics['reallocation']['added_pct']:.1f}%
            - Actions: {metrics['actions']}
            - Total Spend: {metrics['total_spend']}
            - Total Sales: {metrics['total_sales']}
            """
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.4,
                "max_tokens": 700
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" 
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"⚠️ AI Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"⚠️ Could not generate insight: {str(e)}"

    def _render_download_button(self, metrics: Dict[str, Any], insight_text: str = ""):
        """Render the PDF download button."""
        try:
            pdf_bytes = self._generate_pdf(metrics, insight_text)
            
            # Filename
            account_name = "Account" # Placeholder, ideally fetch from session
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"Optimization_Report_{account_name}_{date_str}.pdf"
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key="report_card_download"
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")

    def _generate_pdf(self, metrics: Dict[str, Any], insight_text: str) -> bytes:
        """Generate professionally styled PDF report using fpdf2."""
        from fpdf import FPDF
        
        # Colors (RGB)
        DARK_BG = (15, 23, 42)      # Slate-900
        CARD_BG = (30, 41, 59)      # Slate-800
        TEXT_LIGHT = (226, 232, 240)  # Slate-200
        TEXT_MUTED = (148, 163, 184)  # Slate-400
        GREEN = (34, 197, 94)       # Green-500
        AMBER = (251, 191, 36)      # Amber-400
        
        class ReportPDF(FPDF):
            def header(self):
                # Dark header bar
                self.set_fill_color(*DARK_BG)
                self.rect(0, 0, 210, 35, 'F')
                
                self.set_text_color(*TEXT_LIGHT)
                self.set_font('Helvetica', 'B', 22)
                self.set_xy(15, 10)
                self.cell(0, 10, 'Optimization Report Card', align='L')
                
                self.set_font('Helvetica', '', 10)
                self.set_text_color(*TEXT_MUTED)
                self.set_xy(15, 22)
                self.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='L')
                self.ln(25)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(*TEXT_MUTED)
                self.cell(0, 10, 'Saddle PPC Optimizer | AI-powered optimization insights', align='C')
                
            def section_header(self, title):
                self.set_font('Helvetica', 'B', 14)
                self.set_text_color(*TEXT_LIGHT)
                self.set_fill_color(*CARD_BG)
                self.cell(0, 10, title, fill=True, new_x="LMARGIN", new_y="NEXT")
                self.ln(3)

        pdf = ReportPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        pdf.set_fill_color(*DARK_BG)
        pdf.rect(0, 0, 210, 297, 'F')  # Full page dark background
        
        # Section 1: Performance Snapshot
        pdf.section_header('Performance Snapshot')
        
        # Metrics in a grid
        pdf.set_font('Helvetica', '', 11)
        col_w = 45
        
        def metric_box(label, value, x, y):
            pdf.set_xy(x, y)
            pdf.set_fill_color(*CARD_BG)
            pdf.rect(x, y, col_w, 22, 'F')
            pdf.set_xy(x + 2, y + 3)
            pdf.set_font('Helvetica', 'B', 16)
            pdf.set_text_color(*GREEN)
            pdf.cell(col_w - 4, 8, str(value), align='C')
            pdf.set_xy(x + 2, y + 12)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(*TEXT_MUTED)
            pdf.cell(col_w - 4, 6, label, align='C')
        
        y_pos = pdf.get_y() + 5
        metric_box('ROAS', f"{metrics['roas']:.2f}x", 15, y_pos)
        metric_box('Spend Efficiency', f"{metrics['spend_quality']:.0f}%", 65, y_pos)
        metric_box('Coverage', f"{metrics['optimization_coverage']:.1f}%", 115, y_pos)
        metric_box('Spend Risk', f"{100 - metrics['spend_quality']:.0f}%", 165, y_pos)
        
        pdf.set_y(y_pos + 30)
        
        # Section 2: Actions & Results
        pdf.section_header('Actions & Results')
        
        actions = metrics['actions']
        realloc = metrics['reallocation']
        fin = metrics['financials']
        
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(*TEXT_LIGHT)
        
        # Two columns
        col1_x, col2_x = 15, 110
        y_start = pdf.get_y() + 3
        
        pdf.set_xy(col1_x, y_start)
        pdf.cell(0, 6, f"Bid Increases: {actions['bid_increases']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(col1_x)
        pdf.cell(0, 6, f"Bid Decreases: {actions['bid_decreases']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(col1_x)
        pdf.cell(0, 6, f"Paused Targets: {actions['negatives']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(col1_x)
        pdf.cell(0, 6, f"Promoted Keywords: {actions['harvests']}", new_x="LMARGIN", new_y="NEXT")
        
        # Reallocation summary
        pdf.ln(5)
        pdf.set_x(col1_x)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*AMBER)
        pdf.cell(40, 8, f"-{realloc['removed_pct']:.1f}% Removed")
        pdf.set_text_color(*GREEN)
        pdf.cell(40, 8, f"+{realloc['added_pct']:.1f}% Added")
        pdf.set_text_color(*TEXT_LIGHT)
        pdf.cell(0, 8, f"| Spend Preserved: AED {fin['savings']:,.0f}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(8)
        
        # Section 3: AI Summary
        if insight_text:
            pdf.section_header("Zenny's Insight Summary")
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(*TEXT_LIGHT)
            # Clean markdown bold markers
            clean_text = insight_text.replace('**', '')
            pdf.multi_cell(0, 5, clean_text)
        
        return pdf.output()

    def _generate_html_report(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML report with CSS gauges matching the UI."""
        ai_summary = st.session_state.get('report_card_ai_summary', 'No AI summary generated yet.')
        actions = metrics['actions']
        fin = metrics['financials']
        realloc = metrics['reallocation']
        
        # Helper to generate gauge SVG
        def gauge_svg(value, max_val, label, color_low, color_mid, color_high):
            # Normalize value to 0-180 degrees
            pct = min(value / max_val, 1.0) if max_val > 0 else 0
            angle = pct * 180
            
            # Determine color based on value
            if pct < 0.4:
                fill_color = color_low
            elif pct < 0.75:
                fill_color = color_mid
            else:
                fill_color = color_high
            
            # SVG arc calculation
            import math
            cx, cy, r = 60, 60, 50
            start_angle = 180  # Start from left
            end_angle = 180 - angle
            
            # Convert to radians
            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)
            
            x1 = cx + r * math.cos(start_rad)
            y1 = cy - r * math.sin(start_rad)
            x2 = cx + r * math.cos(end_rad)
            y2 = cy - r * math.sin(end_rad)
            
            large_arc = 1 if angle > 180 else 0
            
            return f'''
            <svg width="120" height="80" viewBox="0 0 120 80">
                <!-- Background arc (gray) -->
                <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="#334155" stroke-width="10" stroke-linecap="round"/>
                <!-- Value arc (colored) -->
                <path d="M {x1} {y1} A 50 50 0 {large_arc} 0 {x2} {y2}" fill="none" stroke="{fill_color}" stroke-width="10" stroke-linecap="round"/>
                <!-- Value text -->
                <text x="60" y="55" text-anchor="middle" fill="{fill_color}" font-size="18" font-weight="bold" font-family="Inter, sans-serif">{value:.0f}{"%" if max_val == 100 else "x" if "ROAS" in label else ""}</text>
                <!-- Label -->
                <text x="60" y="75" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="Inter, sans-serif">{label}</text>
            </svg>
            '''
        
        # Generate gauge SVGs
        roas_gauge = gauge_svg(metrics['roas'], metrics['target_roas'] * 2, "ROAS vs Target", "#ef4444", "#eab308", "#22c55e")
        efficiency_gauge = gauge_svg(metrics['spend_quality'], 100, "Spend Efficiency", "#ef4444", "#eab308", "#22c55e")
        
        # Coverage gauge with custom color zones: 0-3 red, 3-8 yellow, 8-15 green, >15 red
        cov_val = metrics['optimization_coverage']
        if cov_val <= 3:
            cov_color = "#ef4444"
        elif cov_val <= 8:
            cov_color = "#eab308"
        elif cov_val <= 15:
            cov_color = "#22c55e"
        else:
            cov_color = "#ef4444"
        coverage_gauge = gauge_svg(cov_val, 20, "Coverage Health", cov_color, cov_color, cov_color)
        
        risk_gauge = gauge_svg(100 - metrics['spend_quality'], 100, "Spend Risk", "#22c55e", "#eab308", "#ef4444")  # Inverted colors
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Optimization Report Card</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
                body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #e2e8f0; padding: 30px; margin: 0; }}
                h1 {{ color: #f8fafc; margin-bottom: 5px; font-size: 24px; }}
                h2 {{ color: #cbd5e1; font-size: 17px; font-weight: 700; margin: 25px 0 15px; }}
                .meta {{ color: #64748b; font-size: 11px; margin-bottom: 20px; }}
                .gauges {{ display: flex; justify-content: space-between; gap: 15px; margin-bottom: 25px; }}
                .gauge-card {{ background: #1e293b; border-radius: 8px; padding: 15px 10px; text-align: center; flex: 1; }}
                .gauge-label {{ font-size: 11px; color: #cbd5e1; margin-top: 5px; }}
                .actions-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 20px; }}
                .action-item {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #1e293b; border-radius: 6px; }}
                .action-item .icon {{ margin-right: 8px; }}
                .action-item .count {{ font-weight: 600; color: #38bdf8; font-size: 16px; }}
                .realloc-bar {{ display: flex; justify-content: center; align-items: center; gap: 30px; margin: 20px 0; padding: 20px; background: #1e293b; border-radius: 8px; }}
                .realloc-item {{ text-align: center; }}
                .realloc-value {{ font-size: 28px; font-weight: 700; }}
                .realloc-value.removed {{ color: #fbbf24; }}
                .realloc-value.added {{ color: #4ade80; }}
                .realloc-label {{ font-size: 11px; color: #64748b; margin-top: 5px; }}
                .spend-preserved {{ font-size: 15px; margin: 15px 0; }}
                .spend-preserved .value {{ color: #22c55e; font-weight: 700; font-size: 24px; }}
                .ai-summary {{ background: #1e293b; padding: 20px; border-radius: 8px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; }}
                .footer {{ text-align: center; font-size: 10px; color: #475569; margin-top: 25px; padding-top: 15px; border-top: 1px solid #334155; }}
            </style>
        </head>
        <body>
            <h1>Optimization Report Card</h1>
            <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>Performance Snapshot</h2>
            <div class="gauges">
                <div class="gauge-card">{roas_gauge}<div class="gauge-label">Actual ROAS compared to target</div></div>
                <div class="gauge-card">{efficiency_gauge}<div class="gauge-label">% spend on high-efficiency targets</div></div>
                <div class="gauge-card">{coverage_gauge}<div class="gauge-label">% of eligible targets adjusted</div></div>
                <div class="gauge-card">{risk_gauge}<div class="gauge-label">% spend below efficiency threshold</div></div>
            </div>
            
            <h2>Actions & Results</h2>
            <div class="actions-grid">
                <div class="action-item"><span><span class="icon">⬆️</span>Bid Increases</span><span class="count">{actions['bid_increases']}</span></div>
                <div class="action-item"><span><span class="icon">⬇️</span>Bid Decreases</span><span class="count">{actions['bid_decreases']}</span></div>
                <div class="action-item"><span><span class="icon">⏸️</span>Paused Targets</span><span class="count">{actions['negatives']}</span></div>
                <div class="action-item"><span><span class="icon">⭐</span>Promoted Keywords</span><span class="count">{actions['harvests']}</span></div>
            </div>
            
            <h2>Net Spend Reallocation</h2>
            <div class="realloc-bar">
                <div class="realloc-item">
                    <div class="realloc-value removed">-{realloc['removed_pct']:.1f}%</div>
                    <div class="realloc-label">Inefficient Spend Removed</div>
                </div>
                <div style="color: #475569; font-size: 28px;">→</div>
                <div class="realloc-item">
                    <div class="realloc-value added">+{realloc['added_pct']:.1f}%</div>
                    <div class="realloc-label">Invested in Growth</div>
                </div>
            </div>
            
            <div class="spend-preserved">
                💰 <strong>Spend Preserved:</strong> <span class="value">AED {fin['savings']:,.0f}</span>
            </div>
            
            <h2>🧠 Zenny's Insight Summary</h2>
            <div class="ai-summary">{ai_summary}</div>
            
            <p class="footer">Saddle PPC Optimizer | AI-powered optimization insights</p>
        </body>
        </html>
        """
        return html

    def _generate_image_report(self, metrics: Dict[str, Any]) -> bytes:
        """Generate PNG image of the report using html2image."""
        from html2image import Html2Image
        import tempfile
        import os
        
        # Generate styled HTML
        html_content = self._generate_html_report(metrics)
        
        # Create temp directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            hti = Html2Image(output_path=tmpdir, size=(1200, 900))
            
            # Generate image
            output_file = "report.png"
            hti.screenshot(html_str=html_content, save_as=output_file)
            
            # Read the generated image
            image_path = os.path.join(tmpdir, output_file)
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        
        return image_bytes
