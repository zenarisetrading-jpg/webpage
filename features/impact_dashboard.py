"""
Impact Dashboard Module

Sleek before/after analysis dashboard showing the ROI of optimization actions.
Features:
- Hero tiles with key metrics
- Waterfall chart by action type
- Winners/Losers bar chart
- Detailed drill-down table
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
from datetime import datetime, timedelta




def render_impact_dashboard():
    """Main render function for Impact Dashboard."""
    
    # Header Layout with Toggle
    col_header, col_toggle = st.columns([3, 1])
    
    with col_header:
        import base64
        try:
            with open("assets/icons/impact.png", "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
                <h1 style="margin: 0; padding: 0; line-height: 1.2;">Impact Analyzer</h1>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.title("Impact Analyzer")

    with col_toggle:
        st.write("") # Spacer
        # Use radio buttons with horizontal layout for time frame selection
        time_frame = st.radio(
            "Time Frame",
            options=["7D", "14D", "30D", "60D", "90D"],
            index=2,  # Default to 30D (index 2)
            horizontal=True,
            label_visibility="collapsed",
            key="impact_time_frame"
        )
        # Ensure time_frame is never None
        if time_frame is None:
            time_frame = "30D"

    
    # Dark theme compatible CSS
    st.markdown("""
    <style>
    /* Dark theme buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-1px);
    }
    /* Data table dark theme compatibility */
    .stDataFrame {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for database manager
    db_manager = st.session_state.get('db_manager')
    if db_manager is None:
        st.warning("⚠️ Database not initialized. Please ensure you're in the main app.")
        return
    
    # USE ACTIVE ACCOUNT from session state
    selected_client = st.session_state.get('active_account_id', 'default_client')
    
    if not selected_client:
        st.error("⚠️ No account selected! Please select an account in the sidebar.")
        return
    
    # Get available dates for selected account
    available_dates = db_manager.get_available_dates(selected_client)
    
    if not available_dates:
        st.warning(f"⚠️ No action data found for account '{st.session_state.get('active_account_name', selected_client)}'. "\
                   "Run the optimizer to log actions.")
        return
    
    # Sidebar info - show active account
    with st.sidebar:
        # Just show account info, removed comparison settings
        st.info(f"**Account:** {st.session_state.get('active_account_name', selected_client)}")
        st.caption(f"📅 Data available: {len(available_dates)} weeks")
    
    # Get impact data using auto time-lag matching (no date params needed)
    with st.spinner("Calculating impact using auto time-lag matching..."):
        # We always fetch full data first, then filter in memory
        full_summary = db_manager.get_impact_summary(selected_client)
        impact_df = db_manager.get_action_impact(selected_client)
    
    if full_summary['total_actions'] == 0:
        st.info("📊 No actions with matching 'next week' performance data found. This means either:\n"
                "- Actions were logged but no performance data for the following week exists yet.\n"
                "- Upload next week's Search Term Report and run the optimizer to see impact.")
        return
        
    # APPLY FILTER: TIME FRAME BASED (7D, 14D, 30D, 60D, 90D)
    # Parse time frame to days
    time_frame_days = {
        "7D": 7,
        "14D": 14,
        "30D": 30,
        "60D": 60,
        "90D": 90
    }
    days = time_frame_days.get(time_frame, 30)
    filter_label = f"Last {days} Days"
    
    if not impact_df.empty and 'action_date' in impact_df.columns:
        # Convert to datetime for filtering
        impact_df['action_date_dt'] = pd.to_datetime(impact_df['action_date'], errors='coerce')
        
        # Calculate cutoff date relative to the horizon of performance data, not just actions
        # This makes 14D vs 30D more meaningful by filtering based on the 'current' data window
        latest_data_date = pd.to_datetime(available_dates[0]) if available_dates else impact_df['action_date_dt'].max()
        cutoff_date = latest_data_date - timedelta(days=days)
        
        # Filter to only actions within the time frame
        impact_df = impact_df[impact_df['action_date_dt'] >= cutoff_date].copy()
    
    # ==========================================
    # DATE RANGE CALLOUT
    # ==========================================
    if not impact_df.empty and 'action_date' in impact_df.columns:
        action_dates = pd.to_datetime(impact_df['action_date'], errors='coerce').dropna()
        if not action_dates.empty:
            min_date = action_dates.min().strftime('%b %d, %Y')
            max_date = action_dates.max().strftime('%b %d, %Y')
            unique_weeks = impact_df['action_date'].nunique()
            
            # Handle single date vs range
            if min_date == max_date:
                date_text = f"from <strong>{min_date}</strong>"
            else:
                date_text = f"from <strong>{min_date}</strong> to <strong>{max_date}</strong>"
            
            # Dynamic Title based on filter
            title_text = f"{time_frame} Impact Summary"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.05) 100%);
                        border: 1px solid rgba(59, 130, 246, 0.3);
                        border-radius: 8px;
                        padding: 12px 16px;
                        margin-bottom: 16px;">
                📅 <strong>{title_text}</strong> — Showing data {date_text} 
                ({len(impact_df)} actions across {unique_weeks} optimization run{'s' if unique_weeks > 1 else ''})
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================
    # SPLIT: ACTIVE vs DORMANT ACTIONS
    # ==========================================
    if not impact_df.empty:
        # Active: rows where (before_spend + after_spend) > 0
        active_mask = (impact_df['before_spend'].fillna(0) + impact_df['after_spend'].fillna(0)) > 0
        active_df = impact_df[active_mask].copy()
        dormant_df = impact_df[~active_mask].copy()
        
        # Add migration tracking indicator for HARVEST with before_spend > 0
        if 'action_type' in active_df.columns:
            active_df['is_migration'] = (
                (active_df['action_type'] == 'HARVEST') & 
                (active_df['before_spend'].fillna(0) > 0)
            )
    else:
        active_df = impact_df
        dormant_df = pd.DataFrame()
    
    # Recalculate summary for ACTIVE ONLY to ensure consistency
    active_count = len(active_df)
    dormant_count = len(dormant_df)
    
    # Calculate active-only summary metrics
    if not active_df.empty:
        delta_sales = active_df['delta_sales'].fillna(0)
        delta_spend = active_df['delta_spend'].fillna(0)
        is_winner = active_df['is_winner'].fillna(False)
        
        # Build by_action_type with correct nested structure
        by_action_type = {}
        for action_type in active_df['action_type'].unique():
            action_data = active_df[active_df['action_type'] == action_type]
            by_action_type[action_type] = {
                'net_sales': action_data['delta_sales'].fillna(0).sum(),
                'net_spend': action_data['delta_spend'].fillna(0).sum()
            }
        
        active_summary = {
            'total_actions': active_count,
            'net_sales_impact': delta_sales.sum(),
            'net_spend_change': delta_spend.sum(),
            'winners': is_winner.sum(),
            'losers': (~is_winner).sum() if 'is_winner' in active_df.columns else 0,
            'win_rate': (is_winner.sum() / active_count * 100) if active_count > 0 else 0,
            'roi': delta_sales.sum() / delta_spend.sum() if delta_spend.sum() != 0 else 0,
            'by_action_type': by_action_type
        }
    else:
        # Fallback if active_df is empty
        if time_frame:  # Always apply time frame filter
             # If filtering and no active data, show Zeros (don't fallback to lifetime)
             active_summary = {
                'total_actions': 0,
                'net_sales_impact': 0,
                'net_spend_change': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'roi': 0,
                'by_action_type': {}
            }
        else:
             active_summary = full_summary
    
    # ==========================================
    # HERO TILES (Based on Active Only)
    # ==========================================
    _render_hero_tiles(active_summary, active_count, dormant_count)
    
    st.divider()
    
    # ==========================================
    # ACTIVE vs DORMANT TABS
    # ==========================================
    tab_active, tab_dormant = st.tabs([
        f"💰 Active Impact ({active_count})", 
        f"😴 Dormant Optimization ({dormant_count})"
    ])
    
    with tab_active:
        if active_df.empty:
            st.info("No active impact data (all actions have $0 spend)")
        else:
            # Charts for active data - USE ACTIVE SUMMARY
            col1, col2 = st.columns(2)
            with col1:
                _render_waterfall_chart(active_summary)
            with col2:
                _render_winners_losers_chart(active_df)
            
            st.divider()
            
            # Drill-down table with migration badges
            _render_drill_down_table(active_df, show_migration_badge=True)
    
    with tab_dormant:
        if dormant_df.empty:
            st.success("✨ No dormant actions - all your optimizations have activity!")
        else:
            st.info("💤 **Dormant Optimization** — These actions were applied to keywords/targets "
                   "with $0 spend in both periods. The optimization is ready and waiting for traffic.")
            
            # Simple table for dormant
            _render_dormant_table(dormant_df)


def _render_empty_state():
    """Render empty state when no data exists."""
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <div style="font-size: 64px; margin-bottom: 20px;">📊</div>
        <h2 style="color: #666;">No Impact Data Yet</h2>
        <p style="color: #888; max-width: 400px; margin: 0 auto;">
            Run the optimizer and download the report to start tracking actions. 
            Then upload next week's data to see the impact.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### How to use Impact Analysis:
    
    1. **Week 1**: Upload Search Term Report → Run Optimizer → Download Full Report
    2. **Week 2**: Upload new Search Term Report → Come here to see before/after comparison
    """)


def _render_hero_tiles(summary: Dict[str, Any], active_count: int = 0, dormant_count: int = 0):
    """Render the hero metric tiles with glassmorphism style."""
    
    # Custom CSS for glassmorphism tiles
    st.markdown("""
    <style>
    .hero-tile {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .hero-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .hero-label {
        font-size: 0.9rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .positive { color: #10B981; }
    .negative { color: #EF4444; }
    .neutral { color: #6B7280; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Net Sales Impact
    net_sales = summary['net_sales_impact']
    sales_class = 'positive' if net_sales > 0 else 'negative' if net_sales < 0 else 'neutral'
    sales_prefix = '+' if net_sales > 0 else ''
    
    with col1:
        st.markdown(f"""
        <div class="hero-tile">
            <div class="hero-value {sales_class}">{sales_prefix}${net_sales:,.0f}</div>
            <div class="hero-label" title="Difference in total sales between the active period and the baseline period across all optimized targets.">💰 Net Sales Impact ℹ️</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Spend Change
    net_spend = summary['net_spend_change']
    spend_class = 'positive' if net_spend < 0 else 'negative' if net_spend > 0 else 'neutral'
    spend_prefix = '+' if net_spend > 0 else ''
    
    with col2:
        st.markdown(f"""
        <div class="hero-tile">
            <div class="hero-value {spend_class}">{spend_prefix}${net_spend:,.0f}</div>
            <div class="hero-label" title="Difference in total ad spend. Negative (Green) means cost savings; Positive (Red) means increased investment.">📉 Spend Change ℹ️</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Win Rate
    win_rate = summary['win_rate']
    win_class = 'positive' if win_rate >= 60 else 'negative' if win_rate < 40 else 'neutral'
    
    with col3:
        st.markdown(f"""
        <div class="hero-tile">
            <div class="hero-value {win_class}">{win_rate:.0f}%</div>
            <div class="hero-label" title="Percentage of actions that resulted in a positive outcome (Sales Increase OR Cost Savings). Target > 50%.">🎯 Win Rate ℹ️</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Profit Impact (Replaces ambiguous ROI)
    profit_impact = net_sales - net_spend
    profit_class = 'positive' if profit_impact > 0 else 'negative' if profit_impact < 0 else 'neutral'
    profit_prefix = '+' if profit_impact > 0 else ''
    
    with col4:
        st.markdown(f"""
        <div class="hero-tile">
            <div class="hero-value {profit_class}">{profit_prefix}${profit_impact:,.0f}</div>
            <div class="hero-label" title="(Net Sales Change) - (Spend Change). The actual net impact on your wallet. Green means you made more profit.">💸 Profit Impact ℹ️</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary stats below tiles - now shows active/dormant split
    st.markdown(f"""
    <div style="text-align: center; margin-top: 16px; opacity: 0.7;">
        <span style="margin: 0 16px;">✅ {summary['winners']} Winners</span>
        <span style="margin: 0 16px;">❌ {summary['losers']} Losers</span>
        <span style="margin: 0 16px;">💰 {active_count} Active</span>
        <span style="margin: 0 16px;">😴 {dormant_count} Dormant</span>
    </div>
    """, unsafe_allow_html=True)


def _render_waterfall_chart(summary: Dict[str, Any]):
    """Render waterfall chart showing impact by action type."""
    
    st.subheader("📊 Impact by Action Type")
    
    by_type = summary.get('by_action_type', {})
    
    if not by_type:
        st.info("No action type breakdown available")
        return
    
    # Prepare data
    action_types = list(by_type.keys())
    net_impacts = [by_type[t]['net_sales'] - by_type[t]['net_spend'] for t in action_types]
    
    # Sort by impact
    sorted_data = sorted(zip(action_types, net_impacts), key=lambda x: x[1], reverse=True)
    action_types = [x[0] for x in sorted_data]
    net_impacts = [x[1] for x in sorted_data]
    
    # Create waterfall chart
    colors = ['#10B981' if v > 0 else '#EF4444' for v in net_impacts]
    
    fig = go.Figure(go.Waterfall(
        name="Impact",
        orientation="v",
        measure=["relative"] * len(net_impacts) + ["total"],
        x=action_types + ['Total'],
        y=net_impacts + [sum(net_impacts)],
        connector={"line": {"color": "rgba(63, 63, 63, 0.3)"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}},
        totals={"marker": {"color": "#6366F1"}},
        textposition="outside",
        text=[f"${v:+,.0f}" for v in net_impacts] + [f"${sum(net_impacts):+,.0f}"]
    ))
    
    fig.update_layout(
        showlegend=False,
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        xaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_winners_losers_chart(impact_df: pd.DataFrame):
    """Render top winners and losers bar chart."""
    
    st.subheader("🏆 Top Winners & Losers")
    
    if impact_df.empty or 'impact_score' not in impact_df.columns:
        st.info("No impact data available")
        return
    
    # Filter to rows with impact data
    has_data = impact_df['impact_score'].notna()
    df = impact_df[has_data].copy()
    
    if df.empty:
        st.info("No matched actions found")
        return
    
    # Get top 5 winners and losers
    df_sorted = df.sort_values('impact_score', ascending=False)
    winners = df_sorted.head(5)
    losers = df_sorted.tail(5).sort_values('impact_score', ascending=True)
    
    # Combine for chart
    chart_df = pd.concat([winners, losers])
    chart_df['target_short'] = chart_df['target_text'].str[:25] + '...'
    chart_df['color'] = chart_df['impact_score'].apply(lambda x: '#10B981' if x > 0 else '#EF4444')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=chart_df['target_short'],
        x=chart_df['impact_score'],
        orientation='h',
        marker_color=chart_df['color'],
        text=[f"${v:+,.0f}" for v in chart_df['impact_score']],
        textposition='outside'
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(128,128,128,0.5)'),
        yaxis=dict(showgrid=False, autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_drill_down_table(impact_df: pd.DataFrame, show_migration_badge: bool = False):
    """Render detailed drill-down table with optional migration badges."""
    
    with st.expander("📋 Detailed Action Log", expanded=False):
        if impact_df.empty:
            st.info("No actions to display")
            return
        
        # Create display dataframe
        display_df = impact_df.copy()
        
        # Add migration badge for HARVEST with before_spend > 0
        if show_migration_badge and 'is_migration' in display_df.columns:
            # Create a formatted action column with badge
            display_df['action_display'] = display_df.apply(
                lambda r: f"🔄 {r['action_type']}" if r.get('is_migration', False) else r['action_type'],
                axis=1
            )
        else:
            display_df['action_display'] = display_df['action_type']
        
        # Select columns for display
        display_cols = [
            'action_display', 'target_text', 'reason',
            'before_spend', 'after_spend', 'delta_spend',
            'before_sales', 'after_sales', 'delta_sales',
            'impact_score', 'is_winner'
        ]
        
        available_cols = [c for c in display_cols if c in display_df.columns]
        display_df = display_df[available_cols].copy()
        
        # Format numeric columns
        for col in ['before_spend', 'after_spend', 'delta_spend', 'before_sales', 'after_sales', 'delta_sales', 'impact_score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
                )
        
        # Format is_winner
        if 'is_winner' in display_df.columns:
            display_df['is_winner'] = display_df['is_winner'].apply(
                lambda x: "✅ Winner" if x else "❌ Loser" if pd.notna(x) else "-"
            )
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'action_display': 'Action',
            'target_text': 'Target',
            'reason': 'Reason',
            'before_spend': 'Before Spend',
            'after_spend': 'After Spend',
            'delta_spend': 'Δ Spend',
            'before_sales': 'Before Sales',
            'after_sales': 'After Sales',
            'delta_sales': 'Δ Sales',
            'impact_score': 'Impact',
            'is_winner': 'Result'
        })
        
        # Show migration legend if applicable
        if show_migration_badge and 'is_migration' in impact_df.columns and impact_df['is_migration'].any():
            st.caption("🔄 = **Migration Tracking**: This search term's performance is tracked across ALL campaigns "
                      "(e.g., Auto → Exact). Shows efficiency gain from your harvest.")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = impact_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Data (CSV)",
            csv,
            "impact_analysis.csv",
            "text/csv"
        )


def _render_dormant_table(dormant_df: pd.DataFrame):
    """Render simple table for dormant actions ($0 spend in both periods)."""
    
    if dormant_df.empty:
        return
    
    # Simplified view for dormant
    display_cols = ['action_type', 'target_text', 'old_value', 'new_value', 'reason']
    available_cols = [c for c in display_cols if c in dormant_df.columns]
    display_df = dormant_df[available_cols].copy()
    
    display_df = display_df.rename(columns={
        'action_type': 'Action',
        'target_text': 'Target',
        'old_value': 'Old Value',
        'new_value': 'New Value',
        'reason': 'Reason'
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.caption(f"💡 These {len(dormant_df)} optimizations are in place but haven't received traffic yet. "
              "They'll start impacting performance once the keywords/targets get impressions.")


def render_reference_data_badge():
    """Render reference data status badge for sidebar."""
    
    db_manager = st.session_state.get('db_manager')
    if db_manager is None:
        return
    
    try:
        status = db_manager.get_reference_data_status()
        
        if not status['exists']:
            st.markdown("""
            <div style="padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border-left: 3px solid #EF4444;">
                <span style="font-size: 0.85rem;">❌ <strong>No Reference Data</strong></span>
            </div>
            """, unsafe_allow_html=True)
        elif status['is_stale']:
            days = status['days_ago']
            st.markdown(f"""
            <div style="padding: 8px 12px; background: rgba(245, 158, 11, 0.1); border-radius: 8px; border-left: 3px solid #F59E0B;">
                <span style="font-size: 0.85rem;">⚠️ <strong>Data Stale</strong> ({days} days ago)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            days = status['days_ago']
            count = status['record_count']
            st.markdown(f"""
            <div style="padding: 8px 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 3px solid #10B981;">
                <span style="font-size: 0.85rem;">✅ <strong>Data Loaded</strong> ({days}d ago, {count:,} records)</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        pass  # Silently handle errors
