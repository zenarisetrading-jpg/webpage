"""
AI Campaign Insights Feature

Uses semantic clustering + Claude API to analyze search terms
and provide actionable strategic recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from features._base import BaseFeature
from core.data_loader import SmartMapper, load_uploaded_file, safe_numeric
from api.anthropic_client import AnthropicClient

class AIInsightsModule(BaseFeature):
    """AI-Powered Campaign Insights using semantic clustering."""
    
    def __init__(self):
        super().__init__()
        self.n_clusters = 35
        
    def load_config(self) -> Dict[str, Any]:
        """Load AI insights configuration."""
        config = {
            'api_key': '',
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4000
        }
        
        try:
            if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                config['api_key'] = st.secrets['ANTHROPIC_API_KEY']
        except Exception as e:
            pass  # Will use empty string
            
        return config
    
    def render_ui(self):
        """Render AI Insights UI."""
        st.title("📊 Keyword Cluster Analysis")
        
        # Check if data loaded in Data Hub
        from core.data_hub import DataHub
        hub = DataHub()
        
        if hub.is_loaded('search_term_report'):
            st.success("✅ Using data from Data Hub")
            self.data = hub.get_enriched_data()
        
        st.markdown("""
        <div class="tab-description">
        <strong>What this does:</strong>
        <ul>
            <li>Groups thousands of search terms into semantic themes</li>
            <li>Identifies high-performing and underperforming clusters</li>
            <li>Shows which products tie to which search themes</li>
            <li>Highlights wasted spend on non-converting terms</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload (only if not using Data Hub)
        if not hub.is_loaded('search_term_report'):
            st.markdown("### Upload Search Term Report")
            st.info("💡 **Tip:** Use Data Hub to upload files once and access across all features!")
            uploaded = st.file_uploader(
                "Upload your Search Term Report",
                type=['csv', 'xlsx'],
                key="ai_insights_upload"
            )
            
            if uploaded:
                self.data = load_uploaded_file(uploaded)
                if self.data is not None:
                    st.success(f"✅ Loaded {len(self.data):,} search terms")
                    
        if hasattr(self, 'data') and self.data is not None:
            valid, msg = self.validate_data(self.data)
            if not valid:
                st.error(f"❌ Data Validation Failed: {msg}")
                return
                
            if st.button("🚀 Run Cluster Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing semantic patterns..."):
                    results = self.analyze(self.data)
                    
                    # Persist for AI Assistant
                    st.session_state['latest_ai_insights'] = results
                    
                    # Generate Download
                    output_file = self.generate_output(results)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            "📥 Download Analysis (Excel)",
                            data=output_file,
                            file_name="ai_cluster_insights.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
    
    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate search term report."""
        required = ['Customer Search Term', 'Impressions', 'Clicks', 'Spend', 'Orders']
        
        # Check if columns already in standard format (from Data Hub)
        already_standard = all(col in data.columns for col in ['Customer Search Term', 'Impressions', 'Clicks', 'Spend'])
        
        if already_standard:
            # Data from Data Hub - already renamed
            missing = [r for r in required if r not in data.columns]
            if missing:
                return False, f"Missing columns: {', '.join(missing)}"
        else:
            # Original format - need to map
            col_map = SmartMapper.map_columns(data)
            missing = [r for r in required if r not in col_map]
            if missing:
                return False, f"Missing columns: {', '.join(missing)}"
        
        if len(data) < 100:
            return False, "Need at least 100 search terms for clustering"
        
        return True, ""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform semantic clustering and AI analysis."""
        # Check if columns already in standard format (from Data Hub)
        if 'Customer Search Term' in data.columns:
            # Already renamed - use directly
            term_col = 'Customer Search Term'
            imp_col = 'Impressions'
            clicks_col = 'Clicks'
            spend_col = 'Spend'
            orders_col = 'Orders' if 'Orders' in data.columns else None
        else:
            # Original format - need to map
            col_map = SmartMapper.map_columns(data)
            term_col = col_map['Customer Search Term']
            imp_col = col_map['Impressions']
            clicks_col = col_map['Clicks']
            spend_col = col_map['Spend']
            orders_col = col_map.get('Orders')
        
        # Step 1: Cluster search terms
        st.write("### 🔬 Semantic Clustering")
        
        search_terms = data[term_col].dropna().unique()
        
        with st.spinner("Clustering search terms..."):
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=150,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            
            tfidf_matrix = vectorizer.fit_transform(search_terms)
            
            # Determine optimal number of clusters (natural clustering)
            # Use elbow method with silhouette score
            from sklearn.metrics import silhouette_score
            
            min_clusters = 3
            max_clusters = min(35, len(search_terms) // 10)  # At least 10 terms per cluster
            
            if max_clusters <= min_clusters:
                n_clusters = min_clusters
            else:
                # Try different K values and find optimal
                silhouette_scores = []
                K_range = range(min_clusters, min(max_clusters + 1, min_clusters + 10))  # Test up to 10 values
                
                for k in K_range:
                    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans_test.fit_predict(tfidf_matrix)
                    score = silhouette_score(tfidf_matrix, labels)
                    silhouette_scores.append(score)
                
                # Pick K with highest silhouette score
                best_idx = silhouette_scores.index(max(silhouette_scores))
                n_clusters = list(K_range)[best_idx]
                
                st.info(f"🎯 Optimal clusters detected: {n_clusters} (tested {min_clusters}-{max(K_range)})")
            
            # Final clustering with optimal K
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Map clusters back to data
        term_to_cluster = dict(zip(search_terms, cluster_labels))
        data['cluster'] = data[term_col].map(term_to_cluster)
        
        # Check if we have Sales and Orders columns
        sales_col = 'Sales' if 'Sales' in data.columns else None
        if not orders_col or orders_col not in data.columns:
            # Add dummy orders column if missing
            data['Orders'] = 0
            orders_col = 'Orders'
        
        # Aggregate by cluster
        cluster_summary = []
        
        # Check if we have SKU/ASIN columns from advertised products
        # The advertised product report gets merged with "_advertised" suffix
        # After SmartMapper: "Advertised SKU" -> "SKU", then add_suffix -> "SKU_advertised"
        # So we look for: SKU_advertised, ASIN_advertised (from merged data)
        # Or fallback to: SKU, ASIN (if used directly without merge)
        sku_col = None
        asin_col = None
        
        # Debug: show what columns we have
        st.write(f"DEBUG: Available columns: {list(data.columns)[:20]}")  # First 20 columns
        
        # Priority order for SKU column (exact matches first):
        # 1. SKU_advertised (from DataHub merge after SmartMapper rename)
        # 2. Advertised SKU_advertised (if merge happened without SmartMapper rename)
        # 3. SKU (direct column, no merge)
        # 4. Advertised SKU (original Amazon column name)
        # NEVER match columns containing "sales", "units", or numeric metrics
        
        sku_candidates = ['SKU_advertised', 'Advertised SKU_advertised', 'SKU', 'Advertised SKU']
        for candidate in sku_candidates:
            if candidate in data.columns:
                sku_col = candidate
                st.write(f"DEBUG: Found SKU column: {candidate}")
                break
        
        # Priority order for ASIN column:
        # 1. ASIN_advertised (from DataHub merge after SmartMapper rename)
        # 2. Advertised ASIN_advertised (if merge happened without SmartMapper rename)
        # 3. ASIN (direct column, no merge)
        # 4. Advertised ASIN (original Amazon column name)
        
        asin_candidates = ['ASIN_advertised', 'Advertised ASIN_advertised', 'ASIN', 'Advertised ASIN']
        for candidate in asin_candidates:
            if candidate in data.columns:
                asin_col = candidate
                st.write(f"DEBUG: Found ASIN column: {candidate}")
                break
        
        if sku_col:
            sample_skus = data[sku_col].dropna().unique()[:5].tolist()
            st.write(f"DEBUG: Sample SKU values: {sample_skus}")
        else:
            st.warning("⚠️ No 'Advertised SKU' column found. Upload Advertised Product Report in Data Hub to see product names.")
        
        for cluster_id in range(n_clusters):
            cluster_data = data[data['cluster'] == cluster_id]
            
            if len(cluster_data) < 3:
                continue
            
            # Get performance metrics
            total_impressions = cluster_data[imp_col].sum()
            total_clicks = cluster_data[clicks_col].sum()
            total_spend = safe_numeric(cluster_data[spend_col]).sum()
            total_sales = safe_numeric(cluster_data[sales_col]).sum() if sales_col else 0
            total_orders = cluster_data[orders_col].sum()
            
            # Get top terms
            top_terms = cluster_data.groupby(term_col)[imp_col].sum().nlargest(5).index.tolist()
            
            # Get advertised SKUs/ASINs for this cluster
            advertised_skus = []
            if sku_col:
                skus = cluster_data[sku_col].dropna().unique().tolist()
                # Remove empty strings and filter
                skus = [str(s).strip() for s in skus if str(s).strip() and str(s).strip().lower() != 'nan']
                advertised_skus.extend(skus[:5])  # Top 5 SKUs
            if asin_col:
                asins = cluster_data[asin_col].dropna().unique().tolist()
                # Remove empty strings and filter
                asins = [str(a).strip() for a in asins if str(a).strip() and str(a).strip().lower() != 'nan']
                # Only add ASINs if they're different from SKUs already added
                for asin in asins[:5]:
                    if asin not in advertised_skus:
                        advertised_skus.append(asin)
            
            # Better conversion logic:
            # Converting term = term with at least 1 order
            # Non-converting term = term with clicks but 0 orders (actual waste)
            term_performance = cluster_data.groupby(term_col).agg({
                clicks_col: 'sum',
                orders_col: 'sum',
                spend_col: lambda x: safe_numeric(x).sum()
            })
            
            converting_terms = len(term_performance[term_performance[orders_col] > 0])
            # Non-converting = has clicks but zero orders (these are wasters)
            non_converting_terms = len(term_performance[
                (term_performance[clicks_col] > 0) & (term_performance[orders_col] == 0)
            ])
            
            # Wasted spend = spend on terms that got clicks but zero orders
            wasted_spend_data = term_performance[
                (term_performance[clicks_col] > 0) & (term_performance[orders_col] == 0)
            ]
            wasted = wasted_spend_data[spend_col].sum() if len(wasted_spend_data) > 0 else 0
            
            # Waste percentage of THIS cluster's spend (not total)
            waste_pct = (wasted / total_spend * 100) if total_spend > 0 else 0
            
            cluster_summary.append({
                'cluster_id': cluster_id + 1,
                'size': len(cluster_data),
                'top_terms': top_terms,
                'advertised_products': advertised_skus if advertised_skus else ['No SKU data'],
                'impressions': int(total_impressions),
                'clicks': int(total_clicks),
                'spend': float(total_spend),
                'sales': float(total_sales),
                'orders': int(total_orders),
                'converting_terms': converting_terms,
                'non_converting_terms': non_converting_terms,
                'wasted_spend': float(wasted),
                'waste_pct': float(waste_pct),
                'ctr': (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
                'cvr': (total_orders / total_clicks * 100) if total_clicks > 0 else 0
            })
        
        cluster_df = pd.DataFrame(cluster_summary)
        
        st.success(f"✅ Created {len(cluster_df)} clusters")
        
        # Display cluster table
        st.markdown("### 📊 Cluster Overview")
        
        # Format for display
        display_df = cluster_df.copy()
        display_df['top_terms'] = display_df['top_terms'].apply(lambda x: ', '.join(x[:3]))  # Show top 3
        display_df['products'] = display_df['advertised_products'].apply(lambda x: ', '.join(x[:3]))  # Show top 3
        from utils.formatters import get_account_currency
        currency = get_account_currency()
        display_df['spend'] = display_df['spend'].apply(lambda x: f"{currency} {x:,.2f}")
        display_df['wasted_spend'] = display_df['wasted_spend'].apply(lambda x: f"{currency} {x:,.2f}")
        display_df['waste_pct'] = display_df['waste_pct'].apply(lambda x: f"{x:.1f}%")
        display_df['cvr'] = display_df['cvr'].apply(lambda x: f"{x:.2f}%")
        
        # Select columns for display
        display_cols = [
            'cluster_id', 'size', 'top_terms', 'products',
            'impressions', 'clicks', 'orders', 'cvr',
            'spend', 'wasted_spend', 'waste_pct',
            'converting_terms', 'non_converting_terms'
        ]
        
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            column_config={
                'cluster_id': 'ID',
                'size': 'Terms',
                'top_terms': 'Top Search Terms',
                'products': 'Advertised Products',
                'impressions': 'Impr',
                'clicks': 'Clicks',
                'orders': 'Orders',
                'cvr': 'CVR',
                'spend': 'Spend',
                'wasted_spend': 'Wasted',
                'waste_pct': 'Waste %',
                'converting_terms': '✅ Convert',
                'non_converting_terms': '❌ Waste'
            }
        )
        
        with st.expander("ℹ️ How to Read This Table"):
            st.markdown("""
            **Cluster Metrics:**
            - **Terms**: Number of unique search terms in this cluster
            - **Top Search Terms**: Most common searches in this theme
            - **Advertised Products**: Your SKUs/ASINs tied to these searches
            
            **Performance:**
            - **CVR** (Conversion Rate): % of clicks that became orders
            - **Spend**: Total spend on this cluster
            - **Wasted**: Spend on terms with clicks but zero orders
            - **Waste %**: What % of THIS cluster's spend is wasted (not total campaign)
            
            **Term Breakdown:**
            - **✅ Convert**: Terms that got at least 1 order
            - **❌ Waste**: Terms with clicks but zero orders (consider negating these)
            
            **High waste % means:** This cluster has many terms getting clicks but not converting. 
            Consider: (1) Negate wasters, (2) Improve product listing, (3) Check if product matches intent
            """)
        
        # Return results
        return {
            'clusters': cluster_df,
            'data': data
        }
    
    def generate_output(self, results: Dict[str, Any]) -> bytes:
        """Generate Excel output with clusters and AI insights."""
        from io import BytesIO
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Cluster summary
            if 'clusters' in results:
                results['clusters'].to_excel(writer, sheet_name='Clusters', index=False)
            
            # AI Analysis
            if 'ai_analysis' in results:
                # Convert to DataFrame for export
                ai_df = pd.DataFrame({
                    'Type': ['Opportunity'] * 2 + ['Mismatch'] * 2,
                    'Title': [o['title'] for o in results['ai_analysis']['opportunities'][:2]] + 
                             [m['title'] for m in results['ai_analysis']['mismatches'][:2]],
                    'Description': [o['description'] for o in results['ai_analysis']['opportunities'][:2]] +
                                  [m['description'] for m in results['ai_analysis']['mismatches'][:2]],
                    'Action': [o['action'] for o in results['ai_analysis']['opportunities'][:2]] +
                             [m['fix'] for m in results['ai_analysis']['mismatches'][:2]]
                })
                ai_df.to_excel(writer, sheet_name='AI Insights', index=False)
        
        return output.getvalue()
    
    def display_results(self, results: Dict[str, Any]):
        """Display clustering results."""
        if 'clusters' in results:
            st.success(f"✅ Clustered into {len(results['clusters'])} themes")

    def render_chat_interface(self):
        """Render the AI Chat Interface."""
        st.divider()
        st.subheader("🤖 Ask the AI Analyst")
        st.info("The AI has access to your granular performance data, including optimization decisions (Harvests, Negatives, Bids).")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat logic
        if prompt := st.chat_input("Ask about your data (e.g. 'Why is ROAS low?', 'Show me top waste')"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing granular data..."):
                    # 1. Construct Context
                    master_df = self.construct_granular_dataset(self.data)
                    context_str = self.summarize_context(master_df)
                    
                    # 2. Call API
                    config = self.load_config()
                    if not config['api_key']:
                        st.error("❌ No API Key found. Please add ANTHROPIC_API_KEY to .streamlit/secrets.toml")
                        response_text = "I need an API key to function. Please verify your configuration."
                    else:
                        client = AnthropicClient(config['api_key'])
                        
                        system_prompt = f"""
                        You are an expert Amazon PPC Analyst. You have access to a granular dataset of the user's ad performance.
                        
                        DATA CONTEXT:
                        {context_str}
                        
                        INSTRUCTIONS:
                        1. Answer the user's question based strictly on the provided data.
                        2. Cite specific metrics, campaign names, or search terms to support your answer.
                        3. Be concise and actionable.
                        4. If the user asks about optimization, refer to the 'Optimization Status' (Harvest/Negative/Bid) provided in the data.
                        """
                        
                        try:
                            # streaming could be added given client support, using simple generation for now
                            response_text = client.generate_response(system_prompt, prompt)
                            if not response_text:
                                response_text = "I'm sorry, I couldn't generate a response. Please try again."
                        except Exception as e:
                            response_text = f"Error communicating with AI: {str(e)}"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    def summarize_context(self, df: pd.DataFrame) -> str:
        """
        Create a rich, dense summary of the dataset for the LLM.
        """
        # Overall Metrics
        total_spend = safe_numeric(df['Spend']).sum()
        total_sales = safe_numeric(df['Sales']).sum() if 'Sales' in df.columns else 0
        total_orders = df['Orders'].sum() if 'Orders' in df.columns else 0
        roas = total_sales / total_spend if total_spend > 0 else 0
        
        # Breakdown by Optimization Status
        harvest_count = df['Is_Harvest_Candidate'].sum() if 'Is_Harvest_Candidate' in df.columns else 0
        negative_count = df['Is_Negative_Candidate'].sum() if 'Is_Negative_Candidate' in df.columns else 0
        
        summary = f"""
        OVERALL METRICS:
        - Total Spend: {total_spend:,.2f}
        - Total Sales: {total_sales:,.2f}
        - Total Orders: {total_orders:,}
        - Global ROAS: {roas:.2f}x
        
        OPTIMIZATION STATUS:
        - Harvest Candidates (Winners): {harvest_count} terms
        - Negative Candidates (Bleeders): {negative_count} terms
        
        TOP 10 SPENDING TERMS (Granular Detail):
        """
        
        # Top 10 High Spend Rows
        top_spend = df.sort_values('Spend', ascending=False).head(10)
        cols = ['Customer Search Term', 'Impressions', 'Clicks', 'Spend', 'Orders', 'Sales']
        
        # Add context columns if they exist
        if 'Is_Harvest_Candidate' in df.columns: cols.append('Is_Harvest_Candidate')
        if 'Is_Negative_Candidate' in df.columns: cols.append('Is_Negative_Candidate')
        if 'Optimized_Bid' in df.columns: cols.append('Optimized_Bid')
        
        # Ensure cols exist
        cols = [c for c in cols if c in df.columns]
        
        summary += top_spend[cols].to_markdown(index=False)
        
        summary += "\\n\\nTOP 10 WASTED SPEND (0 Orders):\\n"
        wasted = df[df['Orders'] == 0].sort_values('Spend', ascending=False).head(10)
        summary += wasted[cols].to_markdown(index=False)
        
        return summary

