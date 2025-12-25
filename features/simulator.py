import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from ui.components import metric_card
from utils.formatters import get_account_currency

from features._base import BaseFeature

class SimulatorModule(BaseFeature):
    """Standalone module for Bid Change Simulation and Forecasting."""
    
    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data - Simulator relies on Optimizer state, not direct input."""
        if 'latest_optimizer_run' in st.session_state:
            return True, ""
        return False, "Optimizer validation needed"

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data - Simulator uses pre-computed optimization data."""
        return {}

    def render_ui(self):
        """Render the Simulator UI."""
        # Main execution logic
        self._run_logic()

    def run(self):
        """Main execution method for the Simulator module."""
        self.render_ui()

    def _run_logic(self):
        """Internal logic for the simulator UI components."""
        
        # Dependency Check
        if 'latest_optimizer_run' not in st.session_state:
            self._render_empty_state()
            return
            
        # Retrieve optimization data
        r = st.session_state['latest_optimizer_run']
        sim = r.get("simulation")
        date_info = r.get("date_info", {})
        
        # Check if simulation data exists within the run
        if sim is None:
            st.warning("⚠️ Simulation data not found in the latest optimization run.")
            st.info("Go to **Actions Review**, ensure 'Include Simulation' is checked in Settings, and click 'Run Optimization'.")
            if st.button("Go to Actions Review", type="primary"):
                st.session_state['current_module'] = 'optimizer'
                st.rerun()
            return
            
        self._display_simulation(sim, date_info)

    def _render_empty_state(self):
        """Render prompt when no data is available."""
        st.warning("⚠️ No optimization data available.")
        st.markdown("""
        The Simulator requires an active optimization run to forecast changes.
        
        **Steps to Activate:**
        1. Go to **Actions Review**
        2. Configure your settings (Bids, Harvest, Negatives)
        3. Ensure **"Include Simulation"** is checked
        4. Click **"Run Optimization"**
        """)
        
        if st.button("Go to Actions Review", type="primary"):
            st.session_state['current_module'] = 'optimizer'
            st.rerun()

    def _display_simulation(self, sim: dict, date_info: dict):
        """Display advanced simulation results with premium UI."""
        
        icon_color = "#8F8CA3"
        forecast_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 2v10l4.5 4.5"></path><circle cx="12" cy="12" r="10"></circle></svg>'
        table_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line><line x1="9" y1="3" x2="9" y2="21"></line><line x1="15" y1="3" x2="15" y2="21"></line></svg>'
        trend_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
        alert_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'

        st.markdown(f'<div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">{forecast_icon}<span style="color: #F5F5F7; font-size: 1.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">What-If Forecast (Monthly Estimate)</span></div>', unsafe_allow_html=True)
        
        st.info(f"📅 **Data Period:** {date_info.get('label', 'Unknown')} — Forecasted impact is scaled to monthly estimates (4.33x weekly).")
        
        scenarios = sim.get("scenarios", {})
        current = scenarios.get("current", {})
        expected = scenarios.get("expected", {})
        
        weekly_to_monthly = 4.33
        
        current_monthly = {
            "spend": current.get("spend", 0) * weekly_to_monthly,
            "sales": current.get("sales", 0) * weekly_to_monthly,
            "orders": current.get("orders", 0) * weekly_to_monthly,
            "roas": current.get("roas", 0)
        }
        expected_monthly = {
            "spend": expected.get("spend", 0) * weekly_to_monthly,
            "sales": expected.get("sales", 0) * weekly_to_monthly,
            "orders": expected.get("orders", 0) * weekly_to_monthly,
            "roas": expected.get("roas", 0)
        }
        
        def pct_change(new, old):
            return ((new - old) / old * 100) if old > 0 else 0

        # Row 1: Headers (Aligned to halves)
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("<p style='color: #8F8CA3; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 12px;'>Baseline (Current)</p>", unsafe_allow_html=True)
        with h2:
            st.markdown("<p style='color: #8F8CA3; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 12px;'>Expected Forecast</p>", unsafe_allow_html=True)

        # Row 2: Spend & Sales (4 columns)
        c1, c2, c3, c4 = st.columns(4)
        currency = get_account_currency()
        with c1: metric_card("Monthly Spend", f"{currency} {current_monthly['spend']:,.0f}", "shield")
        with c2: metric_card("Monthly Sales", f"{currency} {current_monthly['sales']:,.0f}", "trending_up")
        
        spend_chg = pct_change(expected_monthly["spend"], current_monthly["spend"])
        sales_chg = pct_change(expected_monthly["sales"], current_monthly["sales"])
        with c3: metric_card("Monthly Spend", f"{currency} {expected_monthly['spend']:,.0f}", "shield", delta=f"{spend_chg:+.1f}%")
        with c4: metric_card("Monthly Sales", f"{currency} {expected_monthly['sales']:,.0f}", "trending_up", delta=f"{sales_chg:+.1f}%")

        # Row 3: ROAS & Orders (4 columns)
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Baseline ROAS", f"{current_monthly['roas']:.2f}x", "layers")
        with m2: metric_card("Monthly Orders", f"{current_monthly['orders']:.0f}", "check")
        
        roas_chg = pct_change(expected_monthly["roas"], current_monthly["roas"])
        orders_chg = pct_change(expected_monthly["orders"], current_monthly["orders"])
        with m3: metric_card("Forecasted ROAS", f"{expected_monthly['roas']:.2f}x", "layers", delta=f"{roas_chg:+.1f}%")
        with m4: metric_card("Monthly Orders", f"{expected_monthly['orders']:.0f}", "check", delta=f"{orders_chg:+.1f}%")
            
        st.markdown(f"<p style='color: #8F8CA3; font-size: 0.75rem; text-align: right; margin-top: 5px;'>Confidence: 70% probability | Based on {date_info.get('weeks', 1):.1f} weeks of data</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 1.5 NEW: Risk & Diagnostics Row
        er1, er2 = st.columns(2)
        with er1:
            st.markdown(f"<div style='color: #F5F5F7; font-size: 1rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center;'>{alert_icon}Strategic Risk Analysis</div>", unsafe_allow_html=True)
            risk = sim.get("risk_analysis", {})
            sumry = risk.get("summary", {})
            
            # Stylized Risk Sections
            rc1, rc2, rc3 = st.columns(3)
            with rc1: metric_card("High Risk", f"{sumry.get('high_risk_count', 0)}", "shield", color="#f87171")
            with rc2: metric_card("Med Risk", f"{sumry.get('medium_risk_count', 0)}", "shield", color="#fbbf24")
            with rc3: metric_card("Low Risk", f"{sumry.get('low_risk_count', 0)}", "check", color="#4ade80")
            
            if risk.get("high_risk"):
                with st.expander("Review High Risk Items", expanded=False):
                    st.dataframe(pd.DataFrame(risk["high_risk"]), use_container_width=True, hide_index=True)

        with er2:
            st.markdown(f"<div style='color: #F5F5F7; font-size: 1rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center;'>{forecast_icon}Forecast Diagnostics</div>", unsafe_allow_html=True)
            diag = sim.get("diagnostics", {}) if "diagnostics" in sim else r.get("diagnostics", {})
            
            # Stylized Diagnostic Info (Single Line)
            diag_box = "background: rgba(91, 85, 111, 0.03); border-radius: 8px; padding: 12px; border: 1px solid rgba(143, 140, 163, 0.05);"
            st.markdown(f'<div style="{diag_box}"><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="color: #8F8CA3; font-size: 0.85rem;">Execution Confidence</span><span style="color: #F5F5F7; font-size: 0.85rem; font-weight: 600;">70% Probability</span></div><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="color: #8F8CA3; font-size: 0.85rem;">Active Adjustments</span><span style="color: #F5F5F7; font-size: 0.85rem; font-weight: 600;">{diag.get("actual_changes", 0)} Targets</span></div><div style="display: flex; justify-content: space-between;"><span style="color: #8F8CA3; font-size: 0.85rem;">Data Horizon</span><span style="color: #F5F5F7; font-size: 0.85rem; font-weight: 600;">{date_info.get("days", 0)} Days Benchmarked</span></div></div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Scenario Analysis Chiclet Header
        st.markdown(f'<div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">{table_icon}<span style="color: #F5F5F7; font-size: 1.25rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">Scenario Analysis (Monthly Estimates)</span></div>', unsafe_allow_html=True)
        
        conservative = scenarios.get("conservative", {})
        aggressive = scenarios.get("aggressive", {})
        
        scenario_df = pd.DataFrame({
            "Scenario": ["Current", "Conservative (15%)", "Expected (70%)", "Aggressive (15%)"],
            f"Spend ({currency})": [
                current.get("spend", 0) * weekly_to_monthly,
                conservative.get("spend", 0) * weekly_to_monthly,
                expected.get("spend", 0) * weekly_to_monthly,
                aggressive.get("spend", 0) * weekly_to_monthly
            ],
            f"Sales ({currency})": [
                current.get("sales", 0) * weekly_to_monthly,
                conservative.get("sales", 0) * weekly_to_monthly,
                expected.get("sales", 0) * weekly_to_monthly,
                aggressive.get("sales", 0) * weekly_to_monthly
            ],
            "ROAS": [current.get("roas", 0), conservative.get("roas", 0), expected.get("roas", 0), aggressive.get("roas", 0)],
            "Orders": [
                current.get("orders", 0) * weekly_to_monthly,
                conservative.get("orders", 0) * weekly_to_monthly,
                expected.get("orders", 0) * weekly_to_monthly,
                aggressive.get("orders", 0) * weekly_to_monthly
            ],
            "ACoS": [current.get("acos", 0), conservative.get("acos", 0), expected.get("acos", 0), aggressive.get("acos", 0)]
        })
        
        st.dataframe(
            scenario_df.style.format({
                f"Spend ({currency})": "{:,.0f}",
                f"Sales ({currency})": "{:,.0f}",
                "ROAS": "{:.2f}x",
                "Orders": "{:.0f}",
                "ACoS": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("<p style='color: #8F8CA3; font-size: 0.85rem; margin-top: 10px;'>💡 <strong>Expected scenario</strong> has the highest probability (70%) and represents typical market conditions.</p>", unsafe_allow_html=True)
        
        st.divider()
        
        # Sensitivity Analysis
        sensitivity_df = sim.get("sensitivity", pd.DataFrame())
        if not sensitivity_df.empty:
            st.markdown(f'<div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">{trend_icon}<span style="color: #F5F5F7; font-size: 1.25rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">Bid Sensitivity Analysis</span></div>', unsafe_allow_html=True)
            
            st.markdown("<p style='color: #B6B4C2; font-size: 0.9rem; margin-bottom: 20px;'>See how different global bid adjustment levels would impact account-wide performance trade-offs.</p>", unsafe_allow_html=True)
            
            # Premium Sensitivity Chart (Plotly)
            fig = go.Figure()
            
            # Efficiency Curve Trace
            fig.add_trace(go.Scatter(
                x=sensitivity_df["Spend"],
                y=sensitivity_df["Sales"],
                mode="lines+markers",
                name="Sales vs Spend",
                line=dict(color="#7C3AED", width=3),
                marker=dict(size=8, color="#5B556F", line=dict(width=2, color="white")),
                text=sensitivity_df["Bid_Adjustment"],
                hovertemplate=f"<b>%{{text}}</b><br>Projected Spend: {currency} %{{x:,.0f}}<br>Projected Sales: {currency} %{{y:,.0f}}<extra></extra>"
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=400,
                xaxis=dict(
                    title=f"Avg Weekly Spend ({currency})",
                    gridcolor='rgba(143, 140, 163, 0.1)',
                    zerolinecolor='rgba(143, 140, 163, 0.2)',
                    tickfont=dict(color='#8F8CA3')
                ),
                yaxis=dict(
                    title=f"Avg Weekly Sales ({currency})",
                    gridcolor='rgba(143, 140, 163, 0.1)',
                    zerolinecolor='rgba(143, 140, 163, 0.2)',
                    tickfont=dict(color='#8F8CA3')
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#F5F5F7')
                ),
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.caption("💡 **Tip:** Look for the elbow in the curve where sales growth starts to slow relative to spend increases.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.dataframe(
                sensitivity_df.style.format({
                    "Spend": "{:,.0f}",
                    "Sales": "{:,.0f}",
                    "ROAS": "{:.2f}x",
                    "Orders": "{:.0f}",
                    "ACoS": "{:.1f}%"
                }),
                use_container_width=True,
                hide_index=True
            )
