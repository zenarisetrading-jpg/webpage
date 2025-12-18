"""
Help and Documentation Page - Saddle AdPulse V4
"""

import streamlit as st

def render_readme():
    """Render the comprehensive help & documentation page."""
    
    st.title("📚 Saddle AdPulse Documentation")
    st.caption("Version 4.0 | The Intelligent PPC Operating System")
    
    tab_overview, tab_modules, tab_setup = st.tabs(["🚀 Platform Overview", "📖 Module Guides", "⚙️ Setup & Config"])
    
    # =========================================================================
    # TAB 1: PLATFORM OVERVIEW
    # =========================================================================
    with tab_overview:
        st.markdown("""
        ### Welcome to Saddle AdPulse
        
        This isn't just a reporting tool—it's an **Operating System** for Amazon & Noon PPC. It automates analysis, 
        strategy, and execution to help you scale efficiently while cutting wasted spend.
        
        ---
        
        #### 🔄 The Optimization Workflow
        
        1.  **📥 INGEST (Data Hub)**: Upload your raw reports (Search Terms, Bulk Files, Business Reports).
        2.  **🧠 ANALYZE (AI Strategist & Impact)**: The system finds patterns, bleeders, and opportunities.
        3.  **⚡ OPTIMIZE (Optimization Hub)**: Review and approve bid changes, negatives, and harvests.
        4.  **🚀 EXECUTE (Launcher)**: Push changes to clean up campaigns or launch new growth attackers.
        
        ---
        
        #### 🎯 Core Concepts
        
        | Concept | Definition |
        | :--- | :--- |
        | **Bleeder** | A search term or campaign spending money without profitable returns (High Spend, Low ROAS). |
        | **Harvesting** | Taking a winning search term from an Auto/Broad campaign and moving it to an **Exact Match** campaign to control the bid. |
        | **Negation** | Blocking a search term that doesn't convert, so you stop paying for irrelevant clicks. |
        | **Wasted Spend** | Money spent on clicks that generated ZERO sales. |
        | **ASIN Shielding** | Identifying if an ASIN search is for **Your Product** (good) or a **Competitor** (potential waste). |
        """)
        
        st.info("""
        **💡 Pro Tip:** Start every session at the **Account Overview** to see your health, then go to the **Impact Analyzer** 
        to see exactly where you can make money today.
        """)

    # =========================================================================
    # TAB 2: MODULE GUIDES
    # =========================================================================
    with tab_modules:
        st.markdown("### 🧩 Module Deep Dives")
        
        with st.expander("📂 Data Hub", expanded=True):
            st.markdown("""
            **The Central Nervous System.** All data starts here.
            
            -   **Uploads**: Supports Search Term Reports, Advertised Product Reports, Bulk Files, and Business Reports.
            -   **Status Indicators**: Shows you exactly what's loaded and what's missing.
            -   **Automatic Mapping**: Links SKUs to Parents and Categories automatically if you provide the APR.
            
            > **⚠️ Crucial:** Always upload the **Search Term Report** first. It's the foundation of 90% of the analysis.
            """)
            
        with st.expander("📊 Account Overview"):
            st.markdown("""
            **Your Cockpit.** A high-level view of account performance.
            
            -   **Date Filtering**: Apply global filters to zoom in on specific periods (e.g., "Last Month").
            -   **KPI Cards**: Spend, Sales, ACOS, ROAS, and Orders at a glance.
            -   **Trend Analysis**: Sparklines providing visual context on performance direction.
            """)
            
        with st.expander("📉 Impact Analyzer"):
            st.markdown("""
            **The Money Maker.** This module quantifies the financial impact of optimization.
            
            -   **Waterfall Chart**: Visualization of how Negatives (Savings) and Harvests (Growth) combine to improve your bottom line.
            -   **Projected Savings**: Exact dollar amounts you will save by cutting wasted spend.
            -   **Projected Growth**: Revenue potential from scaling winning keywords.
            -   **Net Impact**: The total estimated improvement to your weekly/monthly profit.
            """)
            
        with st.expander("🧠 AI Strategist"):
            st.markdown("""
            **Your 24/7 PPC Consultant.** Chat with your data using advanced AI.
            
            -   **Context-Aware**: It knows your data. You don't need to paste CSVs. It sees your bleeders, winners, and trends.
            -   **Strategic Queries**: Ask "Why is ACOS going up?", "What are my top wasted terms?", or "Draft a strategy for Q4".
            -   **Knowledge Graph**: It builds connections between search terms, campaigns, and products.
            """)

        with st.expander("⚡ Optimization Hub"):
            st.markdown("""
            **The Engine Room.** Where decisions turn into actions.
            
            -   **🔍 Bids**: Algorithmically calculated bid adjustments to hit your Target ACOS.
            -   **🚫 Negatives**:
                -   **Strict**: 0 orders, High spend.
                -   **Soft**: Low ROAS, High spend.
                -   **Opportunity**: High clicks, 0 sales (Click bleeders).
            -   **🌾 Harvests**: Winning terms found in discovery campaigns ready to be promoted to Exact Match.
            """)
            
        with st.expander("🛡️ Competitor Shield"):
            st.markdown("""
            **Defense Grid.** Manage Amazon Standard Identification Numbers (ASINs).
            
            -   **Identification**: Automatically detects ASINs in your search terms.
            -   **Classification**: Tells you if an ASIN is **Yours** (keep) or a **Competitor** (attack or block).
            -   **Rainforest API**: Fetches real-time product titles and images for context.
            """)
            
        with st.expander("🧪 Simulator"):
            st.markdown("""
            **Crystal Ball.** Forecast the future before you act.
            
            -   **"What-If" Analysis**: See what happens to Spend and Sales if you increase bids by 10% vs 20%.
            -   **Elasticity Modeling**: Uses historical data to predict how sensitive your clicks are to bid changes.
            -   **Risk Assessment**: Helps you find the sweet spot between aggression and efficiency.
            """)
            
        with st.expander("🚀 Campaign Launcher"):
            st.markdown("""
            **Growth Engine.** Create new structure instantly.
            
            -   **Cold Start**: Launch a full Best-Practice campaign structure (Auto, Broad, Exact, PT) for a new product in seconds.
            -   **Harvesting**: Automatically create Single Keyword Campaigns (SKAGs) or segmented campaigns for your harvested winners.
            """)

    # =========================================================================
    # TAB 3: SETUP & CONFIG
    # =========================================================================
    with tab_setup:
        st.markdown("### ⚙️ System Configuration")
        
        st.markdown("""
        #### 1. API Keys (Essential)
        To unlock the full power of **Competitor Shield** and **AI Strategist**, you need API keys in your `.streamlit/secrets.toml` file.
        
        ```toml
        # .streamlit/secrets.toml
        
        [openai]
        api_key = "sk-..."  # For AI Strategist
        
        [rainforest]
        api_key = "..."     # For Competitor Lookup
        
        [general]
        user_brands = ["MyBrand", "SubBrand"] # For identifying your own products
        target_acos = 0.30  # Default Target ACOS (30%)
        ```
        
        #### 2. Troubleshooting Common Issues
        
        | Issue | Solution |
        | :--- | :--- |
        | **"No Data Loaded"** | Go to Data Hub and ensure you've uploaded a **Search Term Report**. |
        | **"Columns Missing"** | Ensure your CSV export is from the standard Amazon/Noon advertising console and headers haven't been renamed. |
        | **"AI Not Responding"** | Check your OpenAI API key credits and validity in `secrets.toml`. |
        | **"Graphs Empty"** | Check your date filters in Account Overview; you might be filtering for a date range with no data. |
        """)
