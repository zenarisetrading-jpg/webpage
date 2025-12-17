# PPC Suite V4 - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
3. [Optimizer Module](#optimizer-module)
4. [ASIN Mapper Module](#asin-mapper-module)
5. [Creator Module](#creator-module)
6. [Data Flow & Integration](#data-flow--integration)
7. [Assistant Module (AI)](#assistant-module-ai)
8. [Configuration & Constants](#configuration--constants)
9. [Account Security & Management](#account-security--management)
10. [v4.3 Release Notes](#v43-release-notes)

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                     ppcsuite_v4.py                      │
│                  (Main Orchestrator)                     │
└───────────────────────┬──────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐  ┌──────────┐  ┌──────────────┐
│   Data Hub   │  │ Features │  │ UI Components│
└──────────────┘  └──────────┘  └──────────────┘
        │               │               │
        │         ┌─────┼─────┐         │
        │         │     │     │         │
        ▼         ▼     ▼     ▼         ▼
   ┌────────┐  ┌────┐┌────┐┌────┐   ┌──────┐
   │ Loader │  │Opt.││ASIN││Crea││   │Layout│
   │ Mapper │  │    ││Map.││tor │   │Cards │
   └────────┘  └────┘└────┘└────┘   └──────┘
        │         │     │     │
        │         ▼     ▼     ▼
        │      ┌────────────────┐
        └─────▶│  Shared Utils  │
               │ (Matchers, etc)│
               └────────────────┘
               ┌────────────────┐
               │   Assistant    │
               │   (AI Brain)   │◀┐
               └────────────────┘ │
                        ▲         │
                        │         │
                   Read Context   │
                        │         │
                 ┌──────┴─────────┴┐
                 │ Knowledge Graph │
                 └─────────────────┘
```

### Module Responsibilities

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| **Optimizer** | Bid optimization, harvest detection, negative detection | Search Term Report | Bid changes, harvest list, negatives |
| **ASIN Mapper** | ASIN intelligence, competitor detection | Search Term Report | Flagged ASINs, categories |
| **Creator** | Campaign creation (Launch & Harvest) | Harvest list / User Inputs | Amazon bulk upload file |
| **Assistant** | AI Strategic Analysis | Full Context | Chat responses, strategic insights |
| **Data Hub** | Centralized data loading & enrichment | CSV/Excel files | Standardized DataFrames |
| **Impact Dashboard** | Historical performance tracking | Action Logs | Sales/Spend Impact Charts |

---

## Core Modules

### Data Hub (`core/data_hub.py`)

**Purpose**: Centralized file upload and data enrichment across all modules.

**Key Functions**:
- `load_data(file_key, uploaded_file)` - Load and standardize report data
- `get_enriched_data()` - Merge search term + purchased product reports + Smart Category Fallback
- `is_loaded(key)` - Check if specific report exists

**Data Keys**:
- `search_term_report` - Main PPC metrics source
- `purchased_product_report` - SKU mapping for enrichment
- `advertised_product_report` - Brand/ASIN ownership

---

## Optimizer Module

### Overview

**File**: `features/optimizer.py` (1989 lines)

**Purpose**: Complete PPC optimization engine with:
1. Harvest detection (high-performing search terms)
2. Negative detection (isolation + performance bleeders)
3. Bid optimization (ROAS-based)
4. Performance simulation

### Configuration

```python
DEFAULT_CONFIG = {
    # Harvest Thresholds
    'HARVEST_ROAS_THRESHOLD': 3.5,
    'HARVEST_MIN_CLICKS': 10,
    'HARVEST_MIN_SPEND': 20,
    
    # Negative Thresholds
    'NEGATIVE_CLICKS_THRESHOLD': 10,
    'NEGATIVE_SPEND_THRESHOLD': 15,
    
    # Bid Optimization
    'ROAS_TARGET': 2.5,
    'BID_MIN': 0.15,
    'BID_MAX': 5.00,
    'ALPHA': 0.15  # Learning rate
}
```

### Workflow

#### Stage 1: Data Preparation

```
prepare_data(df, config)
  ├─ Validate required columns
  ├─ Normalize column names
  ├─ Calculate derived metrics (ROAS, CPC)
  ├─ Detect date range for weekly normalization
  └─ Return (prepared_df, date_info)
```

**Key Logic**:
```python
# ROAS calculation
df['ROAS'] = df['Sales'] / df['Spend']  # Where Spend > 0

# Date detection for weekly normalization
num_weeks = (max_date - min_date).days / 7
```

---

#### Stage 2: Harvest Detection

```
identify_harvest_candidates(df, config, matcher)
  ├─ Filter by performance thresholds
  │   └─ ROAS >= 3.5 AND Clicks >= 10 AND Spend >= 20
  ├─ Check if already running exact match
  │   └─ ExactMatcher.is_exact_running(term)
  ├─ Select winner campaign for each term
  │   └─ Highest ROAS, break ties by Sales, then Clicks
  └─ Return harvest_df with winner campaigns
```

**Winner Selection Logic**:
```python
if term appears in multiple campaigns:
    winner = campaign with:
        1. Highest ROAS
        2. If ROAS tied → Highest Sales
        3. If Sales tied → Highest Clicks
```

**Output Columns**:
- `Customer Search Term`, `ROAS`, `Clicks`, `Sales`, `Spend`
- `Campaign Name` (winner), `SKU` (if enriched)
- `Is_ASIN` flag

---

#### Stage 3: Negative Detection

```
identify_negative_candidates(df, config, harvest_df)
  ├─ Stage 3.1: Isolation Negatives
  │   ├─ Find harvested terms in NON-exact campaigns
  │   ├─ Aggregate by (Campaign, Ad Group, Term)
  │   ├─ Exclude winner campaigns
  │   └─ Flag for negation in source campaigns
  │
  ├─ Stage 3.2: Performance Negatives (Bleeders)
  │   ├─ Filter: Sales == 0 AND non-exact match
  │   ├─ Aggregate by (Campaign, Ad Group, Term)
  │   ├─ Apply thresholds (Default):
  │   │   └─ Clicks >= 10 OR Spend >= 10
  │   ├─ Classify Severity:
  │   │   ├─ 🔴 Hard Stop: Clicks >= 15 (Statistically confirmed failure)
  │   │   └─ 🟡 Performance: Meets min threshold (Wasting money)
  │   └─ Add to negatives list (Action: Negative Exact)
  │
  └─ Stage 3.3: ASIN Mapper Integration
      ├─ Read session_state['latest_asin_analysis']
      ├─ Extract competitor ASINs
      ├─ Deduplicate against existing negatives
      │   └─ Check (Campaign, AdGroup, Term) uniqueness
      ├─ Track stats: total, added, duplicates
      └─ Return (neg_kw_df, neg_pt_df, your_products_df)
```

**Critical De-duplication**:
```python
seen_keys = set()
for negative in [isolation, bleeders, asin_mapper]:
    key = (campaign, ad_group, term.lower())
    if key in seen_keys:
        skip  # Prevent duplicates
    seen_keys.add(key)
```

**Isolation Example**:
```
Term "phone case" harvested to Campaign_Exact
├─ Found in Campaign_Broad → NEGATE
├─ Found in Campaign_Auto → NEGATE
└─ Found in Campaign_Exact → SKIP (winner)
```

**Output**:
- `neg_kw` - Negative keywords (negativeExact match type)
- `neg_pt` - Negative ASIN product targets
- `your_products` - User's ASINs needing manual review

---

#### Stage 4: Bid Optimization

```
calculate_bid_optimizations(df, config, harvested_terms)
  ├─ Segment data:
  │   ├─ Direct (High Granularity): Exact, Broad, Phrase, Auto (with targets)
  │   │   └─ Process every keyword/target individually to preserve specific text
  │   └─ Aggregated (Fallback): Only undefined generic targets (rare)
  │
  ├─ Process Direct Segment
  │   ├─ Group by (Campaign, AdGroup, Keyword/PT)
  │   ├─ Calculate optimal bid using ROAS formula
  │   └─ Return direct_bids_df
  │
  └─ Process Aggregated Segment
      ├─ Aggregate to match type level
      │   └─ Group by (Campaign, AdGroup, Targeting, MatchType)
      ├─ Calculate optimal bid for aggregate
      └─ Return agg_bids_df
```

**Bid Calculation Formula**:
```python
def _optimize_bid(row, config, alpha=0.15):
    current_bid = row['Cost Per Click (CPC)']
    roas = row['ROAS']
    target_roas = config['ROAS_TARGET']
    
    if roas < target_roas:
        # Underperforming → Decrease bid
        adjustment = -(1 - roas / target_roas)
    else:
        # Overperforming → Increase bid
        adjustment = (roas / target_roas - 1)
    
    new_bid = current_bid * (1 + alpha * adjustment)
    new_bid = max(config['BID_MIN'], min(new_bid, config['BID_MAX']))
    
    return new_bid
```

**Aggregation Logic**:
```python
# Don't optimize at search term level for non-exact!
# Aggregate back to targeting level

Group: (Campaign, AdGroup, "keyword:shoe", "broad")
  ├─ Search term: "red shoes" → 100 clicks, AED 50
  ├─ Search term: "blue shoes" → 50 clicks, AED 30
  └─ Aggregate: 150 clicks, AED 80 → Calculate ONE bid
```

---

#### Stage 5: Heatmap Generation

```
create_heatmap(df, config, harvest_df, neg_kw, neg_pt, direct_bids, agg_bids)
  ├─ Group by (Campaign, Ad Group)
  ├─ Calculate aggregate metrics
  ├─ Assign priority colors:
  │   ├─ 🔴 High: ROAS < 1.5 AND Spend > 100
  │   ├─ 🟡 Medium: ROAS < 2.5 OR Spend > 50
  │   └─ 🟢 Good: ROAS >= 2.5
  ├─ Track optimizer actions:
  │   ├─ Count harvests from this group
  │   ├─ Count negatives from this group
  │   └─ Count bid changes from this group
  └─ Return heatmap_df with action tracking
```

**Output Columns**:
- Metrics: `Clicks`, `Spend`, `Sales`, `ROAS`, `ACoS`
- Priority: `🔴 High | 🟡 Medium | 🟢 Good`
- Actions: `Harvests`, `Negatives`, `Bid Changes`

---

#### Stage 6: Simulation

```
run_simulation(df, direct_bids, agg_bids, harvest_df, config, date_info)
  ├─ Calculate baseline (current performance)
  ├─ Normalize to monthly projections (x4.33 weeks)
  ├─ Forecast 3 scenarios:
  │   ├─ Conservative (70% probability)
  │   │   └─ Elasticity: CPC↑ → Clicks↓ (0.4x), CVR↑ (0.05x)
  │   ├─ Expected (25% probability)
  │   │   └─ Elasticity: CPC↑ → Clicks↓ (0.7x), CVR↑ (0.1x)
  │   └─ Aggressive (5% probability)
  │       └─ Elasticity: CPC↑ → Clicks↑ (0.95x), CVR↑ (0.15x)
  ├─ Calculate impact for each scenario
  └─ Return weighted average forecast
```

**Elasticity Model**:
```python
# Conservative scenario
if bid increases by 10%:
    clicks decrease by 4% (0.4 elasticity)
    CVR increases by 0.5% (absolute)
    
# Harvest efficiency multiplier
harvest_sales = harvest_clicks * baseline_cvr * 1.30  # +30% efficiency
```

---

## ASIN Mapper Module

### Overview

**File**: `features/asin_mapper.py` (725 lines)

**Purpose**: Automatic ASIN detection, API lookup, categorization, and negative suggestion

### Workflow

```
analyze(data)
  ├─ Step 1: Detect ASIN Searches
  │   └─ Regex: \b[bB]0[a-zA-Z0-9]{8}\b
  │
  ├─ Step 2: Aggregate by ASIN + Campaign + Ad Group
  │   ├─ Group by (ASIN, Campaign Name, Ad Group Name)
  │   ├─ Sum: Impressions, Clicks, Spend, Orders
  │   └─ Flag: Converting (Orders > 0)
  │
  └─ Step 3: Prioritize for API Lookup
      ├─ Filter non-converting ASINs
      ├─ Apply thresholds:
      │   └─ Clicks >= 10 AND Spend >= 15
      └─ Return top 30 by spend
```

**Key Change (Per-Campaign Tracking)**:
```python
# OLD (Wrong): Aggregated globally
agg = data.groupby('ASIN').sum()
# If ASIN works in Campaign A but bleeds in Campaign B
# → Would flag for ALL campaigns (WRONG!)

# NEW (Correct): Per campaign/ad-group
agg = data.groupby(['ASIN', 'Campaign Name', 'Ad Group Name']).sum()
# → Only flags campaigns where it bleeds (CORRECT!)
```

---

### API Lookup & Enrichment

```
display_results(results)
  ├─ Show initial summary (ASINs found, priority count)
  ├─ User clicks "Lookup ASINs"
  │   │
  │   ├─ Initialize Rainforest API client
  │   ├─ For each high-priority ASIN:
  │   │   ├─ Call API: client.lookup_asin(asin)
  │   │   ├─ Merge stats from aggregation:
  │   │   │   ├─ original_clicks, original_spend
  │   │   │   ├─ Campaign Name, Ad Group Name  ← PRESERVED
  │   │   │   └─ CampaignId, AdGroupId (if available)
  │   │   └─ Add to details_df
  │   │
  │   ├─ Categorize each ASIN:
  │   │   └─ _categorize_asin(row)
  │   │       ├─ Check if in uploaded ASIN list → YOUR_PRODUCT
  │   │       ├─ Check brand match → YOUR_PRODUCT
  │   │       └─ Otherwise → COMPETITOR
  │   │
  │   ├─ Flag for negation:
  │   │   └─ Competitors with Clicks >= 10 AND Spend >= 15
  │   │
  │   └─ Format for Optimizer:
  │       └─ _format_for_optimizer(flagged, competitors, your_products)
  │
  └─ Display enriched results
      ├─ Flagged ASINs (auto-negate competitors)
      ├─ Your products (manual review)
      └─ Diagnostic cards (detailed product info)
```

---

### Optimizer Integration

```
_format_for_optimizer(flagged_competitors, all_competitors, your_products)
  ├─ Competitor ASINs (auto-negate):
  │   └─ Format: {
  │         "Type": "ASIN Mapper - Competitor",
  │         "Campaign Name": row['Campaign Name'],  ← Preserved!
  │         "Ad Group Name": row['Ad Group Name'],    ← Preserved!
  │         "Term": asin,
  │         "Is_ASIN": True,
  │         "Clicks": clicks,
  │         "Spend": spend
  │       }
  │
  └─ Your Products (manual review):
      └─ Format: {
            "Term": asin,
            "Brand": brand,
            "Product": title,
            "Clicks": clicks,
            "Spend": spend,
            "Recommendation": generate_recommendation(row)
          }
          
generate_recommendation(row):
    if spend > 50:
        return "⚠️ High waste - review urgently"
    elif clicks > 30:
        return "⚠️ Many clicks, 0 orders - likely wrong"
    else:
        return "ℹ️ Low volume - monitor"
```

**Session State Storage**:
```python
st.session_state['latest_asin_analysis'] = {
    'asin_details': details_df,
    'competitors': competitors_df,
    'your_products': your_products_df,
    'flagged_for_negation': flagged_df,
    'optimizer_negatives': {  ← For Optimizer integration
        'competitor_asins': [...],
        'your_products_review': [...]
    }
}
```

---

## Creator Module

### Overview

**File**: `features/creator.py` (Unified Launch & Harvest)

**Purpose**: A unified tool for both cold-starting new products and harvesting proven winners.

### Dual Modes

#### Tab 1: 🚀 Launch New Product
**Goal:** Create full-funnel structure for cold starts.
*   **Inputs:** SKU, Price, Target ACOS, Budget.
*   **Smart Logic:**
    *   **Base Bid:** `Price * CVR * Target ACOS`.
    *   **Budget Split:** Weighted allocation (Auto > Manual Keywords > PT).
    *   **Structure:**
        *   **Auto:** Close-match (1.5x bid), Loose-match (1.2x), Substitutes (0.8x), Complements (1.0x).
        *   **Manual:** Waterfall structure (Exact [Top 5] > Phrase [Next 7] > Broad [Rest]).
        *   **Product Targeting:** Competitor ASINs (Bid * 1.1).

#### Tab 2: 🌾 Harvest Winners
**Goal:** Scale high-performing search terms from Optimizer.
*   **Momentum Bidding:** `Bid = Actual CPC * 1.1` (Safe scaling).
*   **Structure:**
    *   **Weekly Consolidation:** All harvests for the week go into `HarvestExact_WK{Week}_{Year}`.
    *   **SKU Grouping:** One Ad Group per SKU (`AG_KW_Exact_{SKU}...`).
    *   **Smart Mapping:** Auto-maps SKUs using the "Purchased Product Report" from Data Hub.

### Workflow

```
render_ui()
  ├─ Tab 1: _render_launch_tab()
  │   └─ User Input -> _generate_launch_bulk_rows()
  │
  └─ Tab 2: _render_harvest_tab()
      ├─ Load candidates from Session State
      ├─ Auto-map SKUs (Data Hub lookup)
      └─ Button -> generate_harvest_bulk_file()
```

---

### Account Health Diagnostics
**(New in v4.1)** - Replaced "Projected Impact" tiles with factual diagnostics to avoid Simulation conflict.

**Method**: `_calculate_account_health(df, r)`

**Metrics**:
1.  **Health Score (0-100)**: Composite of ROAS (40%), Waste Ratio (40%), CVR (20%).
2.  **Waste Ratio**: % of spend on terms with 0 orders.
3.  **Optimization Coverage**: % of search terms covered by Negatives/Harvests.
4.  **Current ROAS**: Actual realized ROAS (vs projected).

**Logic**:
```python
health_score = (roas_score * 0.4 + waste_score * 0.4 + cvr_score * 0.2)
# where roas_score = min(100, current_roas / 4.0 * 100)
```

---

## Assistant Module (AI)

### Overview

**File**: `features/assistant.py` (rewritten v4.2)

**Purpose**: "Deep Strategist" AI that uses a **Knowledge Graph** to provide context-aware insights rather than simple data summaries.

### Architecture: The Knowledge Graph

Instead of feeding raw rows to the LLM, we construct a structured JSON context:

```python
knowledge_graph = {
    "strategic_insights": {
        "market_position": "Aggressive scaling (ROAS > 4.0)",
        "inefficiency_detection": "High waste in 'Generic' portfolio"
    },
    "cross_references": {
        "harvest_negative_paradox": ["term 'x' bleeds in Campaign A but converts in Campaign B"]
    },
    "patterns_detected": {
        "semantic_themes": ["'stainless steel' terms have 30% higher CVR"]
    },
    "impact_forecast": {
        "savings": "AED 1,200/mo",
        "upside": "AED 4,500/mo"
    }
}
```

### Prompt Engineering

**System Prompt**: Enforces a "Senior Strategist" persona.
1.  **Scan Knowledge Graph**: Look for pre-computed anomalies.
2.  **Cross-Reference**: Check conflicting signals (e.g. Harvest + Negative).
3.  **Root Cause Analysis**: Why is it bleeding? (Competitor? Intent mismatch?).
4.  **Quantify Impact**: Use the dollar values from the forecast.
5.  **Actionable Advice**: specific campaign/bid changes.

---

## Data Flow & Integration

### Module Integration Flow

```
1. User Workflow
   └─ Upload Search Term Report → Data Hub
      │
      ├─ Navigate to Optimizer
      │   ├─ Analyze & Optimize
      │   ├─ Get: Harvest list, Negatives, Bid changes
      │   └─ Download bulk files
      │
      ├─ Navigate to ASIN Mapper
      │   ├─ Auto-detect ASINs from same report
      │   ├─ Lookup via API
      │   ├─ Categorize & flag competitors
      │   └─ Integration → Flows to Optimizer Negatives
      │
      └─ Navigate to Creator
          ├─ Upload harvest list (from Optimizer)
          ├─ Upload SKU mapping report
          └─ Generate campaign bulk file

2. Cross-Module Integration
   
   ASIN Mapper → Optimizer:
      st.session_state['latest_asin_analysis']['optimizer_negatives']
      └─ Read in identify_negative_candidates() Stage 3
      
   Optimizer → Creator:
      Download harvest.xlsx
      └─ Upload in Creator UI
      
   Data Hub → All Modules:
      DataHub.get_data('search_term_report')
      └─ Standard DataFrame with mapped columns
```

---

### Session State Architecture

```python
st.session_state = {
    # Data Hub
    'data': {
        'search_term_report': DataFrame,
        'purchased_product_report': DataFrame,
        'advertised_product_report': DataFrame
    },
    
    # Optimizer
    'latest_optimizer_run': {
        'df': prepared_df,
        'date_info': {...},
        'harvest': harvest_df,
        'neg_kw': neg_kw_df,
        'neg_pt': neg_pt_df,
        'your_products_review': your_products_df,  ← NEW
        'direct_bids': direct_bids_df,
        'agg_bids': agg_bids_df,
        'heatmap': heatmap_df,
        'simulation': simulation_dict
    },
    
    # ASIN Mapper
    'latest_asin_analysis': {
        'asin_details': details_df,
        'competitors': competitors_df,
        'your_products': your_products_df,
        'flagged_for_negation': flagged_df,
        'optimizer_negatives': {  ← Integration point
            'competitor_asins': [...],
            'your_products_review': [...]
        }
    },
    
    # Integration Stats
    'asin_mapper_integration_stats': {
        'total': 27,
        'added': 0,
        'duplicates': 27
    }
}
```

---

## Configuration & Constants

### Required Environment Variables

**File**: `.streamlit/secrets.toml`

```toml
RAINFOREST_API_KEY = "your_rainforest_api_key"
ANTHROPIC_API_KEY = "your_anthropic_api_key"
USER_BRANDS = ["s2c", "yourbrand", "zenarisetrading"]
USER_ASINS = ["B09...", "B08..."]
```

---

### Optimizer Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `HARVEST_ROAS_THRESHOLD` | 3.5 | Min ROAS to harvest |
| `HARVEST_MIN_CLICKS` | 10 | Min clicks to harvest |
| `HARVEST_MIN_SPEND` | 20 AED | Min spend to harvest |
| `NEGATIVE_CLICKS_THRESHOLD` | 10 | Min clicks to negate |
| `NEGATIVE_SPEND_THRESHOLD` | 10 AED | Min spend to negate |
| `ROAS_TARGET` | 2.5 | Target ROAS for bids |
| `BID_MIN` | 0.15 AED | Min bid allowed |
| `BID_MAX` | 5.00 AED | Max bid allowed |
| `ALPHA` | 0.15 | Bid adjustment learning rate |

---

### ASIN Mapper Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_clicks` | 10 | Min clicks for high priority |
| `min_wasted_spend` | 15 AED | Min spend for high priority |
| `bleeder_clicks` | 10 | Flagging threshold (competitors) |
| `bleeder_spend` | 15 AED | Flagging threshold (competitors) |

---

## API Integrations

### Rainforest API (ASIN Lookups)

**File**: `api/rainforest_client.py`

**Endpoint**: `https://api.rainforestapi.com/request`

**Request**:
```python
params = {
    'api_key': api_key,
    'type': 'product',
    'asin': asin,
    'amazon_domain': 'amazon.ae'  # or .com, .uk, etc
}
response = requests.get(url, params=params)
```

**Response Parsing**:
```python
product = response.json()['product']

title = product.get('title', '')
brand = product.get('brand', '')
          OR product['buybox_winner'].get('brand', '')
          OR product['specifications'][...]['value']  # Fallback in specs
          
seller = product['buybox_winner'].get('name', '')
price = product['buybox_winner']['price'].get('value')
category = ' > '.join([c['name'] for c in product['categories']])
```

**Caching**: 
- In-memory cache by ASIN + marketplace
- Prevents redundant API calls

---

## Error Handling & Edge Cases

### Optimizer

**Edge Case**: Term appears in multiple campaigns
- **Solution**: Winner selection by ROAS → Sales → Clicks

**Edge Case**: ASIN suggested for negation already in bleeder list
- **Solution**: Deduplication by (Campaign, AdGroup, Term) key

**Edge Case**: Insufficient data for bid optimization
- **Solution**: Skip bid if Clicks < 5

### ASIN Mapper

**Edge Case**: API credits exhausted
- **Solution**: Display clear error message, suggest paid plan

**Edge Case**: ASIN works in Campaign A, bleeds in Campaign B
- **Solution**: Per-campaign aggregation ensures Campaign B gets flagged, not A

**Edge Case**: Brand name variations ("S2C" vs "s2c trading")
- **Solution**: Case-insensitive substring matching

---

## Testing & Validation

### Optimizer Validation Checks

1. **No term harvested AND negated in same campaign** ✓
2. **Negatives unique per (Campaign, AdGroup, Term)** ✓
3. **Bid changes within [BID_MIN, BID_MAX] range** ✓
4. **Simulation totals match baseline + changes** ✓

### ASIN Mapper Validation Checks

1. **Campaign/AdGroup preserved through API lookup** ✓
2. **Deduplication against bleeder list** ✓
3. **User feedback when all ASINs are duplicates** ✓
4. **ASIN regex matches valid Amazon format** ✓

---

## Future Enhancements

### Planned Features

1. **Category Matching**
   - Compare ASIN category vs campaign category
   - Auto-recommend Keep/Negate based on category match

2. **Historical Trend Analysis**
   - Track ASIN performance over time
   - Alert when performance degrades

3. **Budget Pacing**
   - Monitor daily spend vs budget
   - Auto-adjust bids if overspending

4. **Multi-Marketplace Support**
   - Unified dashboard for .ae, .sa, .eg
   - Cross-marketplace ASIN analysis

---

## Account Security & Management

### Strict Account Validation (v4.3)
**Objective:** Prevent data leakage between accounts ("Ghost Account" issue).

**Mechanism:**
1. **DB Validation:** Every page load validates `active_account_id` against the SQLite `accounts` table.
2. **Session Clearing:** Switching accounts triggers an immediate wipe of:
   - `unified_data` (Data Hub cache)
   - `optimizer_results` (Analysis cache)
   - `impact_analysis_cache`
3. **No Implicit Parsing:** Removed logic that inferred `client_id` from campaign names, ensuring data is ONLY attributed to the explicitly active account.

---

## v4.3 Release Notes (Dec 2025)

### Core Hardening
- **Ghost Account Fix:** Validated single source of truth for Account ID.
- **Smart Category Fallback:** Data Hub now auto-links Categories to Ad Groups if `Ad Group Name == SKU`, removing the strict dependency on the Advertised Product Report.
- **Indentation Fix:** Resolved syntax error in Data Hub fallback logic.

### UX Polish
- **Waterfall Chart:** "Total" bar is now grounded (Total) instead of floating (Relative).
- **Metric Clarity:** Replaced ambiguous "ROI" (which causes confusion on negative values) with "Profit Impact ($)".
- **Tooltips:** Added explanatory tooltips to all Impact Dashboard tiles and Optimizer Context widgets.

---

## Deployment & Production

### Requirements

- Python 3.8+
- Streamlit 1.28+
- Dependencies: `pandas`, `requests`, `openpyxl`, `anthropic`

### Launch Commands

```bash
# Local development
streamlit run ppcsuite_v4.py

# Production (with SSL)
streamlit run ppcsuite_v4.py --server.port 8501 --server.address 0.0.0.0
```

### File Structure Checklist

```
ppcsuite_refactored/
├── ppcsuite_v4.py          ✓ Main entry point
├── .streamlit/
│   └── secrets.toml        ✓ API keys, config
├── core/
│   ├── data_hub.py         ✓ Centralized data loading
│   └── data_loader.py      ✓ CSV/Excel processing
├── features/
│   ├── optimizer.py        ✓ Bid optimization engine
│   ├── asin_mapper.py      ✓ ASIN intelligence
│   ├── assistant.py        ✓ AI Strategist (New)
│   ├── ai_insights.py      ✓ Semantic Clustering (New)
│   ├── impact_dashboard.py ✓ Active/Dormant Analysis
│   └── creator.py          ✓ Unified Launch & Harvest
├── api/
│   ├── rainforest_client.py ✓ ASIN API client
│   └── anthropic_client.py  ✓ AI Client
└── utils/
    ├── matchers.py         ✓ Exact match detection
    └── formatters.py       ✓ Output formatting
```

---

## Appendix: Key Algorithms

### Winner Selection (Harvest)

```python
def select_winner(term_group):
    """Select winning campaign for a harvested term."""
    return term_group.sort_values(
        by=['ROAS', 'Sales', 'Clicks'],
        ascending=[False, False, False]
    ).iloc[0]
```

### ROAS-Based Bid Adjustment

```python
def calculate_new_bid(current_cpc, roas, target_roas=2.5, alpha=0.15):
    """Calculate optimal bid using ROAS elasticity."""
    if roas < target_roas:
        adjustment = -(1 - roas / target_roas)  # Decrease
    else:
        adjustment = (roas / target_roas - 1)   # Increase
    
    return current_cpc * (1 + alpha * adjustment)
```

### Campaign-Level Deduplication

```python
def deduplicate_negatives(negatives_list):
    """Ensure unique negatives per campaign/ad-group."""
    seen = set()
    unique = []
    
    for neg in negatives_list:
        key = (
            neg['Campaign Name'],
            neg['Ad Group Name'],
            neg['Term'].lower()
        )
        if key not in seen:
            seen.add(key)
            unique.append(neg)
    
    return unique
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-10  
**Maintained By**: Development Team
