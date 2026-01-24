"""
Knowledge Guide - Saddle AdPulse
"""

import streamlit as st

def render_readme():
    """Render the knowledge guide."""
    
    # Icon styling
    icon_color = "#94a3b8"
    book_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>'
    
    st.markdown(f"""
    <h1 style="font-family: Inter, sans-serif; font-weight: 700; display: flex; align-items: center; gap: 12px;">
        {book_icon}
        <span>Knowledge Guide</span>
    </h1>
    """, unsafe_allow_html=True)
    st.caption("How-tos, models, and strategy guides")
    
    tab_howto, tab_models, tab_faq = st.tabs(["How-To Guides", "Models & Math", "FAQ"])
    
    # =========================================================================
    # TAB 1: HOW-TO GUIDES
    # =========================================================================
    with tab_howto:
        
        with st.expander("**Getting Started**", expanded=True):
            st.markdown("""
            **Step 1: Upload Your Data**
            - Go to **Data Hub** → Upload your Search Term Report
            - Optional: Add Bulk File for campaign IDs, Advertised Products for SKU mapping
            
            **Step 2: Run the Optimizer**
            - Click **Run Optimizer** → Review the recommendations
            - Approve bids, harvests, and negatives
            
            **Step 3: Export & Apply**
            - Download the bulk file → Upload to Amazon/Noon Ads Console
            """)
        
        with st.expander("**Finding Wasted Spend (Bleeders)**"):
            st.markdown("""
            Bleeders are terms eating your budget without returns.
            
            **Where to Find:**
            - **Optimizer → Negatives Tab** — Shows all zero-order terms
            - **AI Strategist** — Ask "Where am I losing money?"
            
            **What to Do:**
            1. Review the negative recommendations
            2. Add them as Negative Exact in your campaigns
            3. Re-run monthly to catch new bleeders
            """)
        
        with st.expander("**Harvesting Winners**"):
            st.markdown("""
            Harvesting moves winning search terms to Exact Match campaigns for control.
            
            **Why Harvest?**
            - Lock in high-performing terms
            - Set specific bids instead of Auto/Broad defaults
            - Improve tracking and attribution
            
            **Process:**
            1. **Optimizer → Harvest Tab** — See qualified terms
            2. Click **Open Campaign Creator**
            3. Generate bulk file with new Exact campaigns
            4. Upload to Ads Console
            """)
        
        with st.expander("**Understanding Bid Recommendations**"):
            st.markdown("""
            The optimizer calculates bids based on performance vs your account baseline.
            
            **Bid Goes UP when:**
            - Term ROAS > Account Median ROAS
            - High conversion rate
            - Strong sales velocity
            
            **Bid Goes DOWN when:**
            - Term ROAS < Account Median ROAS  
            - Low or no conversions
            - Spending without returns
            
            **Tip:** Use the Simulation tab to preview impact before applying.
            """)
        
        with st.expander("**Working with the AI Strategist**"):
            st.markdown("""
            The AI knows your data. Ask strategic questions, not just data lookups.
            
            **Access:** Click the 💬 **Chat Bubble** in the bottom-right corner, or use **"Ask Zenny"** in the sidebar.
            
            **Good Questions:**
            - "Why is my ACOS increasing this month?"
            - "What are my biggest opportunities right now?"
            - "Which campaigns should I pause?"
            - "Help me build a Q1 strategy"
            
            **It Already Knows:**
            - Your bleeders and winners
            - Campaign performance trends
            - Optimization opportunities
            - Historical patterns
            """)
    
    # =========================================================================
    # TAB 2: MODELS & MATH
    # =========================================================================
    with tab_models:
        
        with st.expander("**Understanding the Impact Dashboard**"):
            st.markdown("""
            The **Impact Analyzer** narrates the story of your optimization in a sequential waterfall flow:
            
            1. **Actions** — Total unique optimizations taken.
            2. **Cost Saved** — Waste removed from bleeding search terms.
            3. **Harvest Gains** — Efficiency lift from moving terms to Exact.
            4. **Bid Changes** — Net profit shift from bid adjustments.
            5. **Net Result** — The final "Account Delta" after all improvements.
            
            **Verification**: Look for the **Story Callout** at the bottom. It automatically identifies the biggest driver (positive or negative) to explain "why" the numbers moved.
            """)

        with st.expander("**Bid Calculation Formula**", expanded=True):
            st.markdown("""
            ```
            New Bid = Current Bid × [1 + (ROAS Deviation × Alpha)]
            ```
            
            Where:
            - **ROAS Deviation** = (Term ROAS ÷ Baseline ROAS) - 1
            - **Alpha** = Moderation factor (default 0.15)
            
            **Example:**
            - Current Bid: AED 1.00
            - Term ROAS: 6.0x
            - Baseline ROAS: 4.0x
            - Deviation: (6.0 ÷ 4.0) - 1 = 0.5
            - New Bid: 1.00 × [1 + (0.5 × 0.15)] = **AED 1.075**
            """)
        
        with st.expander("**Impact Attribution Logic**"):
            st.markdown("""
            We use **Measured Actuals** to calculate true impact:
            
            - **Negatives**: `Spend Before - Spend After` (Direct cost savings).
            - **Harvests**: `Sales After - Sales Before` (Revenue growth from new terms).
            - **Bids**: `(Sales Δ) - (Spend Δ)` (Net profit change).
            
            **Verification**: The Impact Dashboard compares performance 14 days *before* vs 14 days *after* every action to prove results.
            """)
        
        with st.expander("**Harvest Qualification Criteria**"):
            st.markdown("""
            A term qualifies for harvesting when it meets the criteria of your selected **Preset**:
            
            1. **ROAS ≥ Target** (variable by preset)
            2. **Clicks ≥ Threshold** (e.g., 10 for Balanced)
            3. **Orders ≥ Threshold** (e.g., 3 for Balanced)
            4. **Not already Exact Match**
            
            **Note:** Use the **Conservative**, **Balanced**, or **Aggressive** presets to adjust these thresholds instantly.
            """)
        
        with st.expander("**Baseline ROAS Calculation (Winsorized)**"):
            st.markdown("""
            We use outlier-resistant statistics for the account baseline.
            
            **Process:**
            1. Filter terms with < AED 5 spend (noise removal)
            2. Cap ROAS at 99th percentile (outlier control)
            3. Take the **median** of remaining values
            
            **Why Median?**
            - Not skewed by lucky low-spend terms
            - More stable than mean
            - Represents true account performance
            """)
        
        with st.expander("**Simulation Elasticity Model**"):
            st.markdown("""
            The simulator uses curved elasticity to forecast outcomes.
            
            | Factor | Relationship | Coefficient |
            | :--- | :--- | :--- |
            | CPC → Clicks | Diminishing | 0.85x |
            | Bid → Impression Share | Variable | Based on competition |
            | Harvest → Efficiency | Multiplicative | 1.3x boost |
            
            **Scenarios:**
            - **Conservative**: Lower click growth, higher skepticism
            - **Expected**: Historical averages
            - **Aggressive**: Maximum harvest efficiency
            """)
    
    # =========================================================================
    # TAB 3: FAQ
    # =========================================================================
    with tab_faq:
        
        with st.expander("**Why am I seeing no harvest candidates?**", expanded=True):
            st.markdown("""
            Possible reasons:
            - Your data period is too short (need at least 7 days)
            - ROAS thresholds are set too high
            - Winning terms are already Exact Match
            
            **Fix:** Switch to the **Aggressive** preset to lower the qualification thresholds and find more terms.
            """)
        
        with st.expander("**How often should I run the optimizer?**"):
            st.markdown("""
            - **Weekly**: For active accounts with >AED 1000/week spend
            - **Bi-weekly**: For moderate spend accounts
            - **Monthly**: For low-spend or stable accounts
            
            Running too frequently can cause bid instability.
            """)
        
        with st.expander("**What's the difference between soft and hard negatives?**"):
            st.markdown("""
            - **Hard Negative**: Zero orders, high spend → Block immediately
            - **Soft Negative**: Low ROAS, some orders → Review before blocking
            
            Soft negatives might have seasonal value or need more data.
            """)
        
        with st.expander("**Why do my bid changes seem small?**"):
            st.markdown("""
            The Alpha factor (0.15) intentionally limits bid swings.
            
            **Reason:** Large bid changes can:
            - Destabilize Amazon's algorithm
            - Cause sudden spend spikes
            - Lose historical performance data
            
            Gradual changes compound over time for stable optimization.
            """)
        
        with st.expander("**Can I use this for Noon AND Amazon?**"):
            st.markdown("""
            Yes! The system works with any platform that exports:
            - Search Term Reports
            - Bulk Files
            - Performance data in standard formats
            
            Column names are automatically mapped.
            """)
        
        with st.expander("**What data do I need at minimum?**"):
            st.markdown("""
            **Required:**
            - Search Term Report (last 14-30 days)
            
            **Recommended:**
            - Bulk file (for campaign IDs)
            - Advertised Product Report (for SKU mapping)
            
            **Optional:**
            - Category mapping (for category-level views)
            """)
