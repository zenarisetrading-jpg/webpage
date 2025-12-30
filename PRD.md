# PPC Optimizer - Product Requirements Document (PRD)

**Version**: 1.0  
**Last Updated**: December 25, 2025  
**Document Owner**: Zayaan Yousuf

---

## Executive Summary

PPC Optimizer is a comprehensive Amazon Advertising optimization platform that automates the analysis, optimization, and management of Sponsored Products campaigns. The system ingests performance data, identifies optimization opportunities, generates actionable recommendations, and tracks the impact of implemented changes.

---

## Product Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PPC OPTIMIZER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────────────────┐  │
│  │   DATA      │ => │   PERFORMANCE   │ => │       OPTIMIZER            │  │
│  │   INPUT     │    │   REPORTING     │    │  ┌──────────────────────┐  │  │
│  │   MODEL     │    │   MODEL         │    │  │ Harvest Module       │  │  │
│  └─────────────┘    └─────────────────┘    │  │ Negative Module      │  │  │
│                                             │  │ Bid Optimizer        │  │  │
│       ┌─────────────────────────────────┐  │  │ Campaign Launcher    │  │  │
│       │        IMPACT MODEL             │  │  │ Bulk Export          │  │  │
│       │  (Before/After Measurement)     │  │  └──────────────────────┘  │  │
│       └─────────────────────────────────┘  └────────────────────────────┘  │
│                                                                              │
│       ┌─────────────────────────────────┐                                   │
│       │       FORECAST MODEL            │                                   │
│       │   (Simulation & Projections)    │                                   │
│       └─────────────────────────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Input Model

### 1.1 Overview
The Data Input Model is the foundation of the system. It handles ingestion, validation, normalization, and persistence of all input data required for optimization.

### 1.2 Data Sources

| Data Source | Required | Description | Key Columns |
|-------------|----------|-------------|-------------|
| **Search Term Report** | ✅ Required | Primary performance data from Amazon Advertising | Campaign Name, Ad Group Name, Customer Search Term, Targeting, Match Type, Spend, Sales, Clicks, Impressions, Orders |
| **Advertised Product Report** | Optional | Maps campaigns/ad groups to SKUs and ASINs | Campaign Name, Ad Group Name, SKU, ASIN |
| **Bulk ID Mapping** | Optional | Amazon bulk file with Campaign IDs, Ad Group IDs, Keyword IDs | Entity, Campaign Id, Ad Group Id, Keyword Id, Product Targeting Id |
| **Category Mapping** | Optional | Internal SKU to Category/Sub-Category mapping | SKU, Category, Sub-Category |

### 1.3 Data Processing Pipeline

```
Raw File Upload
      │
      ▼
┌─────────────────────┐
│  Column Detection   │ ← SmartMapper identifies columns automatically
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Column Normalization │ ← Map to standard schema (Campaign Name, Spend, etc.)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Data Type Casting  │ ← Ensure numeric types for metrics
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Match Type Inference │ ← Detect AUTO/PT/CATEGORY from targeting expressions
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Data Enrichment    │ ← Merge bulk IDs, SKUs, categories
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Database Persistence │ ← Save aggregated data to PostgreSQL
└─────────────────────┘
```

### 1.4 Data Aggregation Rules

Data is aggregated at the **Campaign + Ad Group + Target + Week** level:
- Daily rows are summed into weekly aggregates
- Metrics aggregated: Spend, Sales, Clicks, Impressions, Orders
- Week defined as Monday-Sunday (ISO week format)

### 1.5 Database Schema

**Primary table: `target_stats`**
| Column | Type | Description |
|--------|------|-------------|
| client_id | VARCHAR | Account identifier |
| start_date | DATE | Week start date (Monday) |
| campaign_name | VARCHAR | Campaign name (normalized) |
| ad_group_name | VARCHAR | Ad Group name (normalized) |
| target_text | VARCHAR | Targeting expression or keyword |
| match_type | VARCHAR | exact/broad/phrase/auto/pt/category |
| spend | DECIMAL | Total spend for period |
| sales | DECIMAL | Total sales for period |
| clicks | INTEGER | Total clicks for period |
| impressions | INTEGER | Total impressions for period |
| orders | INTEGER | Total orders for period |

### 1.6 Target Grouping and Identification (CANONICAL REFERENCE)

> **CRITICAL**: This section defines the grouping and identification logic used throughout the entire codebase. Reference this whenever implementing action logging, impact calculation, or optimization logic.

#### 1.6.1 Universal Grouping Key

**All actions and performance data are grouped by:**
```
Campaign Name + Ad Group Name + Target (Keyword/PT/ASIN/Auto Type)
```

This applies to:
- Action logging
- Impact measurement
- Bid optimization
- Performance aggregation

#### 1.6.2 Target Type Identification

| Target Type | Identification Pattern | Example |
|-------------|------------------------|---------|
| **Category** | `category="..."` | `category="brahmi" price>149` |
| **Product Targeting (PT)** | `asin="B0XXXXXXX"` or `asin-expanded="B0XXXXXXX"` | `asin="B08TT9LR1W"` |
| **Auto** | `close-match`, `loose-match`, `complements`, `substitutes` | `close-match` |
| **Keyword (Exact)** | match_type = `exact` | `moss stick` (exact) |
| **Keyword (Phrase)** | match_type = `phrase` | `moss stick` (phrase) |
| **Keyword (Broad)** | match_type = `broad` | `moss stick` (broad) |

#### 1.6.3 Customer Search Term (CST) Usage

**CST is ONLY used for:**
- Identifying harvest candidates (search terms to graduate to Exact match)
- Identifying negative candidates (search terms to block)

**CST is NOT used for:**
- Grouping keys
- Impact calculation joins
- Bid optimization grouping

#### 1.6.4 Impact Calculation Joins

When matching actions to performance data in `target_stats`:

```sql
-- Correct: Join on Campaign + Ad Group
ON LOWER(action.campaign_name) = LOWER(stats.campaign_name)
AND LOWER(action.ad_group_name) = LOWER(stats.ad_group_name)

-- Wrong: Join on target_text (only 10% match rate)
ON LOWER(action.target_text) = LOWER(stats.target_text)
```

**Performance data is aggregated at the Ad Group level** to capture all search term activity affected by the action.

---

## 2. Performance Reporting Model

### 2.1 Overview
The Performance Reporting Model provides comprehensive dashboards for analyzing campaign performance, identifying trends, and understanding performance distribution by various dimensions.

### 2.2 Key Features

#### 2.2.1 Executive KPIs
- **Total Spend**: Aggregate ad spend for selected period
- **Total Sales**: Attributed sales revenue
- **ROAS**: Return on Ad Spend (Sales / Spend)
- **ACoS**: Advertising Cost of Sale (Spend / Sales × 100)
- **CTR**: Click-Through Rate
- **CVR**: Conversion Rate (Orders / Clicks)

#### 2.2.2 Period Comparison
- Current vs. Previous period comparison
- Trend indicators (↑/↓) with percentage change
- Configurable periods: 7D, 14D, 30D

#### 2.2.3 Performance Breakdown Views

| Dimension | Hierarchy | Metrics Shown |
|-----------|-----------|---------------|
| **Match Type** | Exact > Broad > Phrase > Auto > PT > Category | Spend, Sales, ROAS, ACoS |
| **Category** | Category > Sub-Category > SKU | Spend, Sales, ROAS, Orders |
| **Campaign** | Campaign > Ad Group > Target | Full metrics drill-down |

#### 2.2.4 Trend Visualization
- Time-series charts for Spend/Sales
- ROAS trend line overlay
- Interactive date range selection

### 2.3 Data Flow

```
Session State / Database
        │
        ▼
┌─────────────────────┐
│  Date Range Filter  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Metric Calculation │ ← ROAS, ACoS, CVR, CTR
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Period Comparison  │ ← vs. previous period
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Visualization      │ ← Charts, tables, KPIs
└─────────────────────┘
```

---

## 3. Optimizer Model

### 3.1 Overview
The Optimizer Model is the core intelligence of the system. It analyzes performance data and generates three types of optimization recommendations:
1. **Harvest** - Promote high-performing search terms to exact match
2. **Negative** - Block wasteful or bleeding search terms
3. **Bid** - Adjust bids for optimal ROI

### 3.2 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| TARGET_ROAS | 3.5 | Target Return on Ad Spend |
| TARGET_ACOS | 25% | Target Advertising Cost of Sale |
| MIN_IMPRESSIONS | 200 | Minimum impressions for analysis |
| MIN_CLICKS | 3 | Minimum clicks for analysis |
| MIN_SPEND | 5.0 | Minimum spend (AED) for analysis |

### 3.3 Harvest Module

#### 3.3.1 Purpose
Identify high-performing search terms or ASINs that should be "harvested" as exact match keywords in dedicated campaigns to capture more of their traffic at higher efficiency.

#### 3.3.2 Harvest Criteria

| Criteria | Threshold | Description |
|----------|-----------|-------------|
| Minimum Clicks | 10+ | Statistical significance |
| Minimum Orders | 3+ (CVR-adjusted) | Proven conversion |
| Minimum Sales | AED 150+ | Revenue threshold |
| ROAS Requirement | ≥80% of bucket median | Relative performance |

#### 3.3.3 Winner Selection Logic
When the same search term appears in multiple campaigns:
1. Calculate **Winner Score** = Sales + (ROAS × 5)
2. Select campaign with highest score as **Winner**
3. Other campaigns become targets for **Isolation Negatives**

#### 3.3.4 Output
- Harvest candidate list with winner campaign/SKU
- Recommended bid (based on historical CPC × efficiency multiplier)
- Bulk file template for creating exact match keywords

---

### 3.4 Negative Module (Defence)

#### 3.4.1 Purpose
Identify search terms that should be negated to stop wasted spend. Two types:
1. **Isolation Negatives** - Harvest terms to negate in non-winner campaigns
2. **Performance Negatives** (Bleeders) - High-spend, zero-conversion terms

#### 3.4.2 Isolation Negative Logic
```
For each Harvested Term:
    Winner Campaign = Campaign with highest winner score
    
    For each OTHER campaign where term appears:
        Create Negative Keyword for that campaign
        (Prevents cannibalization, funnels traffic to winner)
```

#### 3.4.3 Performance Negative (Bleeder) Criteria

| Type | Criteria | Description |
|------|----------|-------------|
| **Soft Stop** | Clicks ≥ Expected × 2, Orders = 0 | High click, no conversion |
| **Hard Stop** | Clicks ≥ Expected × 3, Orders = 0 | Very high waste |
| **High Spend** | Spend > Threshold, ROAS = 0 | Burning budget |

**Expected Clicks Calculation**:
```
Account CVR = Clamped(Account Orders / Account Clicks, 1%, 10%)
Expected Clicks = 1 / Account CVR
Soft Threshold = Expected Clicks × 2
Hard Threshold = Expected Clicks × 3
```

#### 3.4.4 ASIN Detection
Identifies Product Targeting (PT) negatives separately:
- Pattern: `B0` followed by 8 alphanumeric characters
- Pattern: `asin="..."` or `asin-expanded="..."`

#### 3.4.5 Output Categories
| Category | Entity Type | Application Level |
|----------|-------------|-------------------|
| Negative Keywords | Keyword | Campaign or Ad Group |
| Negative Product Targeting | ASIN | Campaign or Ad Group |
| Your Products Review | Own ASINs | Manual review (no auto-action) |

---

### 3.5 Bid Optimizer Module

#### 3.5.1 Purpose
Calculate optimal bid adjustments for all targets based on performance vs. target ROAS, using a bucketed approach.

#### 3.5.2 Bucketing Strategy

| Bucket | Criteria | Description |
|--------|----------|-------------|
| **Exact** | Match Type = EXACT only | Manual keyword bids |
| **Product Targeting** | `asin=`, `asin-expanded=` | ASIN/PT bids |
| **Broad/Phrase** | Match Type = BROAD or PHRASE | Keyword bids |
| **Auto/Category** | Match Type = AUTO or targeting is auto-type | Auto campaign targets |

#### 3.5.3 Bid Calculation Formula
```
Performance Gap = (Actual ROAS / Target ROAS) - 1

If Performance Gap > 0 (Outperforming):
    Bid Multiplier = 1 + (Gap × 0.5)  # Scale up cautiously
    New Bid = Current CPC × Bid Multiplier
    Cap at 2× current bid

If Performance Gap < 0 (Underperforming):
    Bid Multiplier = 1 + (Gap × 0.35)  # Scale down conservatively
    New Bid = Current CPC × Bid Multiplier
    Floor at 50% current bid

Clamp New Bid: Min = 0.10 AED, Max = 20.00 AED
```

#### 3.5.5 Visibility Boost (NEW - Dec 2025)

For targets that are **not winning auctions** despite running for 2+ weeks.

| Condition | Threshold |
|-----------|-----------|
| Data window | ≥ 14 days |
| Impressions | < 100 (not winning auctions) |
| Impressions | > 0 (not paused) |

**Eligible Match Types:**
- ✅ Exact, Phrase, Broad (explicit keyword choices)
- ✅ Close-match (most relevant auto type)

**NOT Eligible (Amazon decides relevance):**
- ❌ loose-match, substitutes, complements
- ❌ ASIN targeting (product targeting)
- ❌ Category targeting

**Action:** Increase bid by **30%** to gain visibility.

**Rationale:**
- High impressions + low clicks = CTR problem (not bid issue)
- LOW impressions = bid not competitive (needs boost)
- Only boost keywords the advertiser explicitly chose

#### 3.5.6 Exclusions
- Terms already in Harvest list (will be promoted to exact)
- Terms already in Negative list (will be blocked)
- Low-data targets (below minimum thresholds)

#### 3.5.7 Output
- Bid adjustment recommendations per target
- Grouped by bucket (Exact, PT, Broad/Phrase, Auto)
- Before/After bid comparison
- Expected impact calculation

---

### 3.6 Campaign Launcher Module

#### 3.6.1 Purpose
Generate Amazon bulk upload files for creating new campaigns based on optimization results.

#### 3.6.2 Launch Types

| Type | Use Case | Output |
|------|----------|--------|
| **Harvest Launch** | Create exact match campaigns from harvest candidates | New campaigns + keywords |
| **Cold Start Launch** | Launch new product campaigns from scratch | Full campaign structure |

#### 3.6.3 Harvest Launch Structure
```
For each Harvested Keyword:
    1. Create Campaign: "[PRODUCT]_Harvest_Exact"
    2. Create Ad Group: "[KEYWORD]_AG"
    3. Create Keyword: EXACT match
    4. Set Bid: Momentum Bid = Historical CPC × Efficiency Multiplier
```

#### 3.6.4 Cold Start Launch Structure
- Auto Campaign (discovery)
- Broad Match Campaign
- Exact Match Campaign (if seeds provided)
- Product Targeting Campaign (if competitor ASINs provided)

#### 3.6.5 Output
- Amazon-compatible bulk upload XLSX file
- Includes: Campaign, Ad Group, Keyword/PT rows
- Full bulk sheet schema (67 columns)

---

### 3.7 Bulk Export Module

#### 3.7.1 Purpose
Generate Amazon-compliant bulk upload files for all optimization actions.

#### 3.7.2 Export Types

| Type | Contents | Entity Type |
|------|----------|-------------|
| **Negatives Bulk** | Negative keywords + PT | Negative Keyword, Negative Product Targeting |
| **Bids Bulk** | Bid updates | Keyword, Campaign Negative Keyword, Product Targeting |
| **Harvest Bulk** | New exact match keywords | Keyword |
| **Combined Bulk** | All actions in one file | Mixed |

#### 3.7.3 Validation Rules
- Campaign/Ad Group ID matching (uses Bulk ID Mapping)
- ASIN format validation
- Duplicate detection
- Entity type consistency

#### 3.7.4 Bulk File Schema
Standard Amazon bulk upload columns (67 total):
- Product, Entity, Operation
- Campaign Id, Ad Group Id, Keyword Id, Product Targeting Id
- Campaign Name, Ad Group Name
- Bid, State
- Keyword Text, Negative Keyword Text
- Product Targeting Expression
- Match Type, etc.

---

## 4. Impact Model

### 4.1 Overview
The Impact Model measures the real-world effectiveness of optimization actions by comparing performance before and after implementation.

### 4.2 Measurement Methodology

```
Action Logged (T0)
      │
      ▼
┌─────────────────────┐
│  "Before" Period    │ ← 7 days before T0
│  (Baseline)         │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  "After" Period     │ ← 7 days after T0
│  (Measurement)      │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Delta Calculation  │ ← After - Before
└─────────────────────┘
```

### 4.3 Key Metrics

| Metric | Calculation | Description |
|--------|-------------|-------------|
| **Revenue Impact** | (After Sales - Before Sales) | Incremental revenue |
| **ROAS Change** | (After ROAS - Before ROAS) | Efficiency change |
| **Spend Change** | (After Spend - Before Spend) | Budget impact |
| **Implementation Rate** | Executed / Total Actions | Applied vs. logged |

### 4.4 Action Types Tracked

| Action Type | Description | Expected Impact |
|-------------|-------------|-----------------|
| NEGATIVE_ISOLATION | Harvest term negated in source | Reduced waste, focused traffic |
| NEGATIVE_PERFORMANCE | Bleeder term blocked | Direct cost savings |
| BID_INCREASE | Bid increased for performer | More traffic, higher sales |
| BID_DECREASE | Bid decreased for underperformer | Cost savings, improved efficiency |
| HARVEST_NEW | New exact match keyword created | Higher conversion on proven terms |

### 4.5 Dashboard Components

- **Hero Tiles**: Actions, ROAS Change, Revenue Impact, Implementation %
- **Waterfall Chart**: Revenue contribution by action type
- **Winners/Losers Chart**: Top/bottom performing actions
- **Drill-Down Table**: Individual action details with before/after

### 4.5 Validation Methodology

#### 4.5.1 Purpose
Validation distinguishes **action-driven impact** from **market-wide shifts**. An action is only "confirmed" if we can prove the advertiser's decision caused the observed change, not broader market conditions.

#### 4.5.2 Validation Status Definitions

| Status | Meaning | Toggle Included? |
|--------|---------|-----------------| 
| **✓ Confirmed blocked** | Negative keyword → Spend dropped to $0 | ✅ Yes |
| **✓ Normalized match** | Negative keyword → Spend dropped significantly MORE than account baseline | ✅ Yes |
| **✓ CPC Validated** | Bid change → After CPC within ±15% of suggested bid | ✅ Yes |
| **✓ CPC Match** | Bid change → After CPC closely matches recommendation | ✅ Yes |
| **✓ Directional match** | Bid change → CPC moved in expected direction (bid up→CPC up, bid down→CPC down) by >5% | ✅ Yes |
| **✓ Volume Validated** | Bid change → Impressions changed significantly (+20%/-15%) in expected direction | ✅ Yes |
| **✓ Directional + Volume** | Bid change → Weak CPC signal + moderate impressions signal combined | ✅ Yes |
| **✓ Harvested to exact** | Harvest action → Source auto/broad has $0 spend (traffic moved to exact) | ✅ Yes |
| **◐ Unverified** | Target has no performance data in after window | ❌ No |
| **⚠️ Source still active** | Harvest action → Source campaign still has spend (not fully migrated) | ❌ No |
| **Preventative** | Negative with $0 spend before action (blocking future waste) | ❌ No |
| **Part of harvest consolidation** | Isolation negative (blocks source to funnel to winner) | ❌ No |

#### 4.5.3 What is "Normalized" Validation?

**Problem**: If the whole account's spend dropped 30%, every target's spend dropping 30% does NOT mean negatives are working—it's a market shift.

**Solution**: Compare target change against **account baseline**:
- Calculate **baseline change** = Account After Spend / Account Before Spend
- Calculate **target change** = Target After Spend / Target Before Spend 
- If target change is **≥50% worse than baseline** (or 100% drop), the negative is confirmed

**Example**:
```
Account baseline: Spend dropped 20% overall
Target spend: Dropped 85%

Threshold = baseline - 0.50 = -0.20 - 0.50 = -0.70 (or 70% drop)
Target dropped 85% > 70% threshold → ✓ Normalized match
```

#### 4.5.4 Validation Layers for Bid Changes

**Layer 1: CPC Match (Primary)**
- Check if `actual_after_cpc` is within ±20% of `suggested_bid`
- If yes → ✓ CPC Validated

**Layer 2: Directional Match (Fallback)**
- Check if CPC moved in expected direction by >5%
- Bid DOWN → CPC should decrease
- Bid UP → CPC should increase
- If direction matches → ✓ Directional match

**Layer 3: Baseline Beat (Secondary)**
- Check if target's ROAS change outperformed account's ROAS change
- If target beat baseline → ✓ Normalized winner

### 4.6 "Validated Only" Toggle Behavior

When **enabled** (default):
- Shows only actions with statuses containing: `✓`, `CPC Validated`, `CPC Match`, `Directional`, `Confirmed`, `Normalized`
- Excludes: Unverified, Preventative, Partial, Source still active
- **PARTIAL is NOT included** — only fully confirmed actions count

When **disabled**:
- Shows all logged actions regardless of validation status
- Useful for seeing complete picture including pending items

### 4.7 Decision Impact Methodology

#### 4.7.1 Purpose
Decision Impact measures the **incremental revenue** generated by bid optimization decisions, compared to a counterfactual "do nothing" scenario.

#### 4.7.2 Formula

```
Decision_Impact = Actual_After_Sales - Expected_Sales

Where:
  Expected_Clicks = After_Spend / Before_CPC
  Expected_Sales = Expected_Clicks × SPC_Baseline
```

#### 4.7.3 SPC Baseline (30D Rolling Average)

**Why not use the 7D window SPC?**
- Small sample sizes create high volatility (±10x swings)
- One large order can skew the baseline
- Week-to-week random noise dominates

**Solution: 30-Day Rolling Average SPC**
- Uses the target's **last 30 days of performance** to calculate SPC
- Reduces baseline volatility by ~27%
- Falls back to window-based SPC if 30D data unavailable

```
SPC_Baseline = COALESCE(Rolling_30D_SPC, Window_SPC)
```

#### 4.7.4 Key Metrics

| Metric | Formula | Applies To |
|--------|---------|------------|
| **Decision Impact** | Actual Sales - Expected Sales | Bid changes, Harvests |
| **Spend Avoided** | Before Spend - After Spend (if positive) | Negatives only |

#### 4.7.5 Interpretation

| Decision Impact | Meaning |
|-----------------|---------|
| **Positive** | Bid decision generated MORE sales than counterfactual |
| **Zero** | Performance matched expectations |
| **Negative** | Bid decision generated LESS sales than counterfactual |

### 4.8 Multi-Horizon Impact Measurement

#### 4.8.1 Purpose
Amazon's attribution window is 7-14 days. Measuring at 7 days produces incomplete data and false negatives. We use a principled multi-horizon approach for accurate impact measurement.

#### 4.8.2 Measurement Horizons

| Horizon | Before Window | After Window | Maturity | Purpose |
|---------|---------------|--------------|----------|---------|
| **14D** | 14 days | 14 days | 17 days | Early signal — did the action have an effect? |
| **30D** | 14 days | 30 days | 33 days | Confirmed — is the impact sustained? |
| **60D** | 14 days | 60 days | 63 days | Long-term — did the gains hold? |

#### 4.8.3 Maturity Formula
```
is_mature(horizon) = (action_date + horizon_days + 3) ≤ latest_data_date
```

- **Before window**: Always 14 days (fixed baseline)
- **After window**: 14, 30, or 60 days (per selected horizon)
- **Buffer**: 3 days for attribution to settle

#### 4.8.4 Example (data through Dec 28)
| Action Date | 14D Mature? | 30D Mature? | 60D Mature? |
|-------------|-------------|-------------|-------------|
| Dec 11 | ✅ (Dec 28) | ❌ (Jan 13) | ❌ (Feb 12) |
| Nov 25 | ✅ | ✅ (Dec 28) | ❌ (Jan 27) |
| Oct 1 | ✅ | ✅ | ✅ |

#### 4.8.5 Dashboard Behavior
- User selects horizon via radio toggle (14D / 30D / 60D)
- **Aggregates** (ROAS change, revenue impact, win rate) include ONLY actions mature for selected horizon
- **Pending actions** excluded from aggregates, shown separately with expected maturity date
- Action counts will decrease at longer horizons (fewer actions are old enough)

> **Why not 7 days?**  
> Most PPC tools measure at 7 days. This captures only ~75% of attributed conversions and measures bid changes before they stabilize. We choose accuracy over speed.

---

## 5. Forecast Model (Simulator)

### 5.1 Overview
The Forecast Model simulates the expected impact of proposed optimizations before implementation, helping advertisers understand potential outcomes.

### 5.2 Simulation Approach

#### 5.2.1 Elasticity Model
```
For each Bid Change:
    Δ CPC = Bid Change × CPC Elasticity
    Δ Clicks = Δ CPC × Click Elasticity
    Δ CVR = Δ Click Volume × CVR Elasticity
    
    Projected Sales = Current Sales × (1 + Δ Clicks) × (1 + Δ CVR)
    Projected Spend = Current Spend × (1 + Δ CPC) × (1 + Δ Clicks)
```

#### 5.2.2 Elasticity Scenarios

| Scenario | CPC Elasticity | Click Elasticity | CVR Effect | Probability |
|----------|---------------|------------------|------------|-------------|
| Conservative | 0.30 | 0.50 | 0% | 15% |
| Expected | 0.50 | 0.85 | +10% | 70% |
| Aggressive | 0.60 | 0.95 | +15% | 15% |

#### 5.2.3 Harvest Efficiency Multiplier
For new exact match keywords (harvest):
```
Harvest Efficiency = 1.30  # 30% efficiency gain from exact match
Projected Revenue = Historical Revenue × Harvest Efficiency
```

#### 5.2.4 Harvest Launch Multiplier
New harvest keywords start with an aggressive bid to win impressions:
```
Harvest Launch Multiplier = 2.0  # 2x the source keyword's CPC
New Bid = Source_CPC × Launch_Multiplier
```
**Rationale**: Exact match auctions are more competitive. Starting at 2x the source CPC ensures the new keyword can win impressions. The bid optimizer will correct down if performance is poor.

### 5.3 Simulation Output

| Metric | Description |
|--------|-------------|
| **Projected Spend** | Expected spend after bid changes |
| **Projected Sales** | Expected sales after optimizations |
| **Projected ROAS** | Expected ROAS improvement |
| **Confidence Range** | Low / Expected / High scenarios |

### 5.4 Visualization

- **Before/After Comparison**: Key metrics side-by-side
- **Confidence Intervals**: Probabilistic range of outcomes
- **Scenario Analysis**: Conservative to Aggressive projections

---

## 6. Data Persistence

### 6.1 Database Architecture

The system uses PostgreSQL for persistent storage:

| Table | Purpose |
|-------|---------|
| `accounts` | Account registry |
| `target_stats` | Aggregated performance data |
| `actions_log` | Optimization action history |
| `bulk_mappings` | Campaign/AdGroup/Keyword IDs |
| `category_mappings` | SKU to Category mapping |
| `advertised_product_cache` | Campaign to SKU/ASIN mapping |
| `account_health_metrics` | Periodic health snapshots |

### 6.2 Session State vs. Database

| Data | Session State | Database |
|------|--------------|----------|
| Fresh upload (raw) | ✅ Full granularity | Aggregated weekly |
| Historical data | ❌ Lost on reload | ✅ Persistent |
| Optimization results | ✅ Current session | ❌ Not persisted |
| Action history | ❌ | ✅ Persistent |

---

## 7. User Interface

### 7.1 Navigation Structure

```
Home (Account Overview)
├── Data Hub (Upload & Manage)
├── Performance Snapshot (Reports)
├── Optimization Engine
│   ├── Overview
│   ├── Defence (Negatives)
│   ├── Bids
│   ├── Harvest
│   ├── Audit
│   └── Bulk Export
├── Impact & Results
├── Simulator
├── Campaign Creator
├── ASIN Mapper
└── AI Strategist
```

### 7.2 Key UI Patterns

- **Premium Dark Theme**: Modern glassmorphism design
- **Lazy Loading**: Heavy modules load on-demand
- **Fragmented UI**: Interactive elements don't cause full reruns
- **Responsive Layout**: Adapts to screen size

---

## 8. Open Issues & Backlog

### 8.1 Known Issues

| Issue | Priority | Description |
|-------|----------|-------------|
| Negative detection discrepancy | High | DB data shows fewer negatives than session state |
| Weekly aggregation granularity | Medium | Daily patterns may be lost during weekly aggregation |

### 8.2 Future Enhancements

- [ ] Real-time Amazon Ads API integration
- [ ] Automated scheduled optimization runs
- [ ] Multi-marketplace support
- [ ] Budget allocation optimizer
- [ ] Dayparting recommendations

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ROAS** | Return on Ad Spend = Sales / Spend |
| **ACoS** | Advertising Cost of Sale = Spend / Sales × 100 |
| **CVR** | Conversion Rate = Orders / Clicks |
| **CTR** | Click-Through Rate = Clicks / Impressions |
| **CPC** | Cost Per Click = Spend / Clicks |
| **Harvest** | Promoting a proven search term to exact match |
| **Isolation** | Negating a harvested term in non-winner campaigns |
| **Bleeder** | High-spend search term with no conversions |
| **PT** | Product Targeting (ASIN-based targeting) |
| **Normalized Validation** | Comparing target's change against account baseline to distinguish action-driven impact from market shifts |
| **Directional Match** | CPC moved in expected direction (bid up → CPC up) by >5% |
| **Baseline Beat** | Target's ROAS/spend change outperformed account average |
| **Confirmed Action** | Action validated via CPC match, directional check, or normalized threshold |
| **Decision Impact** | Market-adjusted revenue change attributable to advertiser decisions |
| **30D Rolling SPC** | 30-day average Sales Per Click, used as stable baseline for counterfactual calculation |
| **Harvest Launch Multiplier** | 2.0x bid multiplier for new harvest keywords to compete in exact match auctions |

---

## Appendix B: File Locations

| Component | File Path |
|-----------|-----------|
| Data Hub | `core/data_hub.py` |
| Optimizer | `features/optimizer.py` |
| Performance Snapshot | `features/performance_snapshot.py` |
| Impact Dashboard | `features/impact_dashboard.py` |
| Simulator | `features/simulator.py` |
| Campaign Creator | `features/creator.py` |
| Bulk Export | `features/bulk_export.py` |
| Database Manager | `core/postgres_manager.py` |
| Main UI | `ppcsuite_v4_ui_experiment.py` |
