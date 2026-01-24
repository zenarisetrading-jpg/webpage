# PPC Optimizer - Product Requirements Document (PRD)

**Version**: 2.0  
**Last Updated**: January 6, 2026  
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

> **Note:** 0 impressions = bid SO low it can't even enter auctions. Paused targets are identified by `state='paused'`, not impressions.

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
│  "Before" Period    │ ← 14 days before T0
│  (Baseline)         │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  "After" Period     │ ← 14 days after T0
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
| **VISIBILITY_BOOST** | Bid +30% for low-impression keywords | More auction wins, increased traffic |
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

##### ✅ Validated Statuses (Included in "Validated Only" toggle)

| Status | Action Type | Meaning | Threshold/Calculation |
|--------|-------------|---------|----------------------|
| **✓ Confirmed blocked** | Negative | Search term completely eliminated | After spend = $0 |
| **✓ Normalized match** | Negative | Spend dropped significantly vs baseline | Drop ≥50% more than account baseline |
| **✓ Harvested (complete)** | Harvest | Source term fully migrated | Source after spend = $0 |
| **✓ Harvested (90%+ blocked)** | Harvest | Near-complete migration | Source spend dropped ≥90% |
| **✓ Harvested (migrated)** | Harvest | Strong migration | Source spend dropped ≥75% |
| **✓ Harvested (partial)** | Harvest | Partial migration | Source spend dropped ≥50% |
| **✓ CPC Validated** | Bid Change | CPC matches suggested bid | Observed CPC within ±15% of new bid |
| **✓ CPC Match + Baseline** | Bid Change | CPC matches AND beat trend | CPC validated + outperformed baseline |
| **✓ Directional match** | Bid Change | CPC moved in expected direction | CPC change >5% in correct direction |
| **✓ Directional + Baseline** | Bid Change | Direction correct + beat baseline | Directional match + baseline beat |
| **✓ Directional + Volume** | Bid Change | Direction + click volume aligned | Weak CPC + moderate volume signals |
| **✓ Volume Validated** | Bid Change | Clicks changed as expected | Clicks +20%/-15% in expected direction |
| **✓ Spend Eliminated** | Any | Target has zero spend | After spend = $0 |
| **✓ Confirmed paused** | Pause | Pause action confirmed | After spend = $0 |

##### ⚠️ Not Validated Statuses (Excluded from "Validated Only")

| Status | Action Type | Meaning | Cause |
|--------|-------------|---------|-------|
| **⚠️ NOT IMPLEMENTED** | Negative | Negative not applied | Spend continues after action date |
| **⚠️ Source still active** | Harvest | Source term not blocked | Source spend drop ≤0% |
| **⚠️ Migration X%** | Harvest | Partial harvest progress | Source dropped X% (less than 50%) |
| **⚠️ Not validated** | Bid Change | No validation criteria met | CPC, direction, and volume all failed |
| **⚠️ Still has spend** | Pause | Pause not working | Target still has spend after pause |

##### ◐ Inconclusive Statuses (Excluded from "Validated Only")

| Status | Meaning | Cause |
|--------|---------|-------|
| **◐ Unverified (no target data)** | No matching stats found | Target text not in target_stats |
| **◐ Unverified (low baseline)** | Before period too small | Before spend < minimum threshold |
| **◐ Dormant target** | Zero activity both periods | Spend = $0 before AND after |
| **◐ Low click volume** | Not enough clicks | Clicks < 5 in measurement period |
| **◐ Beat baseline only** | Only passed secondary test | Baseline beat but no primary validation |
| **◐ No after data** | No post-action data yet | Data upload not current |

##### Special Statuses

| Status | Meaning | Notes |
|--------|---------|-------|
| **Preventative - no spend to save** | Proactive block | Negative added for term with $0 spend (prevents future waste) |
| **Part of harvest consolidation** | Isolation negative | Blocking source as part of harvest migration strategy |

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

### 4.6.1 Market Tag Definitions

Market Tags identify whether observed changes are due to user actions or broader market conditions.

| Market Tag | Meaning | Calculation |
|------------|---------|-------------|
| **Normal** | Market conditions stable | No significant account-wide decline detected |
| **Market Downshift** | Account-wide decline | Account baseline spend/ROAS dropped significantly vs prior period |
| **Low Data** | Insufficient data | `before_clicks = 0` (no click history to assess) |

**Why it matters:**
- Actions under "Market Downshift" that saved spend are still credited as "Good" — the decision helped in a down market
- Actions under "Normal" market with negative Decision Impact are flagged as "Bad" — decision error
- Actions under "Low Data" are marked "Neutral" — no basis to judge

### 4.6.2 Decision Outcome Definitions

Decision Outcome summarizes whether the action was beneficial, neutral, or harmful.

| Decision Outcome | Meaning | Logic |
|------------------|---------|-------|
| **🟢 Good** | Positive impact | Decision Impact > 0 OR (defensive action + significant spend avoided) |
| **🟡 Neutral** | Small/ambiguous impact | \|Decision Impact\| < 5% of before_sales or $10 |
| **🔴 Bad** | Negative impact in stable market | Decision Impact < 0 AND Market Tag = "Normal" |

**Special cases:**
- Defensive actions (BID_DOWN, PAUSE, NEGATIVE) with Spend Avoided ≥ 10% of before_spend are "Good" even if Decision Impact ≤ 0
- Market Downshift actions with negative impact are "Neutral" (market conditions masked the action effect)

### 4.6.3 Confidence Classification (High / Medium / Low)

Confidence is a classification layer that indicates data reliability — it does NOT alter Decision Impact values.

#### Calculation

1. **Per-Action Variance**: `sigma_i = |decision_impact_i| × (1 - confidence_weight_i)`
   - Apply market multiplier: `sigma_i *= 1.3` if Market Downshift

2. **Aggregate Variance**: `total_sigma = sqrt(Σ sigma_i²)` (validated actions only)

3. **Signal-to-Noise Ratio**: `signal_ratio = |total_decision_impact| / total_sigma`

4. **Classification**:

| Condition | Confidence |
|-----------|------------|
| signal_ratio ≥ 1.5 AND validated_actions ≥ 30 | **High** |
| signal_ratio ≥ 0.8 | **Medium** |
| else | **Low** |

#### Downgrade Rule
If >40% of impact comes from "Market Downshift" actions → downgrade one level.

#### UI Display
Shown under Decision Impact hero tile: `Confidence: High | Medium | Low`

**Tooltip**: "Confidence reflects data sufficiency, variance, and market stability."

#### Constraints
- ❌ Do NOT use "statistically significant" in customer-facing copy
- ✅ Z-scores ARE calculated in backend for confidence (see 4.6.6)
- ❌ Do NOT add toggles for confidence

### 4.6.6 Confidence Weighting (Jan 2026)

#### Purpose
Dampen impact attribution for low-click decisions to reduce noise from statistically unreliable samples.

#### Formula
```
confidence_weight = min(1.0, before_clicks / 15)
final_decision_impact = decision_impact × confidence_weight
```

#### Impact Tiers
| Tier | Criteria | Treatment |
|------|----------|----------|
| **Excluded** | before_clicks < 5 | Not counted in totals |
| **Directional** | 5 ≤ clicks < 15 | Partial weight (33%-99%) |
| **Validated** | clicks ≥ 15 | Full weight (100%) |

#### Dampening Effect
Typically 5-8% reduction in total attributed impact (reduces noise from low-data decisions).

### 4.6.7 Statistical Confidence (Z-Score Based)

#### Purpose
Provide statistically rigorous confidence measure for aggregate impact values.

#### Backend Calculation
```python
mean_impact = impact_values.mean()
std_error = impact_values.std() / sqrt(n)
z_score = mean_impact / std_error
confidence_pct = min(99, stats.norm.cdf(z_score) * 100)
```

#### Confidence Labels
| z-score | Label | Meaning |
|---------|-------|--------|
| ≥ 2.58 | Very High | 99% confident |
| ≥ 1.96 | High | 95% confident |
| ≥ 1.645 | Moderate | 90% confident |
| < 1.645 | Directional | < 90% confident |

#### Display
Shown in "How we know this" expander:
- Confidence label (Very High/High/Moderate/Directional)
- Number of validated decisions
- Win rate percentage
- Confidence percentage (capped at 99%)

### 4.6.8 Maturity Calculation (Fixed Jan 2026)

#### The Bug (Pre-Fix)
`actual_after_days` used `COUNT(DISTINCT start_date)` which counted **weekly report files** (2-4), not **calendar days** (14-64).

#### The Fix
```sql
-- Now uses calendar days
actual_after_days = latest_date - action_date + 1
```

#### Impact
- Oct 28 actions: 2 days → 64 days ✅
- Correctly identifies 86% of actions as mature

### 4.6.9 Incremental Contribution Badge

#### Purpose
Show what percentage of total revenue was contributed by optimizations.

#### Formula
```
incremental_pct = attributed_impact / (before_sales + after_sales) × 100
```

#### Display
Shown as badge next to Hero impact value: `+7.6% of revenue`

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

### 4.9 Decision Outcome Matrix (Jan 2026 - Counterfactual Framework)

#### 4.9.1 Philosophy
Isolate **decision quality** from **market conditions** by comparing actual performance to a counterfactual baseline.

#### 4.9.2 Counterfactual Logic

**X-Axis: Expected Trend %**
- Formula: `(Expected Sales - Before Sales) / Before Sales * 100`
- Expected Sales = `(New Spend / Baseline CPC) × Baseline SPC`
- **Translation**: "If we maintained our old efficiency, what would sales be at the new spend level?"

**Y-Axis: vs Expectation %**
- Formula: `Actual Change % - Expected Trend %`
- **Translation**: "How much did we BEAT or MISS the counterfactual baseline?"

#### 4.9.3 Quadrants

| Quadrant | Criteria | Meaning | Attribution |
|----------|----------|---------|-------------|
| **Offensive Win** | X≥0, Y≥0 | Spend increased + beat baseline → Efficient scaling | ✅ Included |
| **Defensive Win** | X<0, Y≥0 | Market shrank, but we beat the expected drop → Good defense | ✅ Included |
| **Decision Gap** | X≥0, Y<0 | Spend increased but missed expectations → Inefficient scale | ✅ Included |
| **Market Drag** | X<0, Y<0 | Market shrank AND we underperformed → External confound | ❌ **EXCLUDED** |

#### 4.9.4 Decision-Attributed Impact (Refined Hero Metric)

**Formula**: `Sum(Offensive Wins + Defensive Wins + Decision Gaps)`

**Critical Exclusion**: Market Drag is **EXCLUDED** from all impact totals.

**Reasoning**:
- Market Drag represents external headwinds we didn't control
- Including it would conflate market luck with decision quality
- We ONLY attribute impact where our DECISION had clear directional influence

**Display Format**:
- Main Number: Net Impact (Green if positive, Red if negative)
- Breakdown: "✅ Wins: +X (Offensive + Defensive) | ❌ Gaps: -Y"
- Footnote: "ℹ️ Z actions excluded (Market Drag — ambiguous attribution)"

### 4.10 Capital Protected (Refined Logic)

**Definition**: Wasteful spend eliminated from confirmed negative keyword blocks.

**Formula**: `Sum of before_spend for NEGATIVE actions where observed_after_spend == 0`

**Why This Works**:
- Only counts actions **INTENDED** to protect capital (negatives)
- `after_spend == 0` proves the block was successful
- Bid increases **SHOULD** increase spend — that's scaling winners

**Display**: "From X confirmed negatives" + "Confidence: High"

**Why Not Total Spend Reduction?**
- Bid optimizations may increase or decrease spend — both can be correct
- Only NEGATIVE actions have the explicit goal of capital protection
- Counting only confirmed blocks (spend = 0) provides clear proof



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

---

## 9. Organization & User Governance (Phases 1-4)

### 9.1 System Model (Canonical)
The system follows a strict hierarchy where **Organizations** are the primary entities, owning both **Users** (Seats) and **Amazon Accounts**.

```
Organization (Billing Entity)
├── Users (Seats)
│   └── Role (Global + Account Overrides)
└── Amazon Accounts (Resource)
```

### 9.2 Organization Model
- **Attributes**: Name, Type (Agency/Seller), Subscription Plan, Amazon Account Limit.
- **Rules**:
    - Organizations own Amazon accounts (not users).
    - Amazon accounts are **hard-capped** by plan.
    - Users (Seats) are **unlimited** but billable (Soft enforcement).

### 9.3 User & Role Model

#### 9.3.1 Global Roles (Hierarchy)
Roles are cumulative. Higher roles inherit all permissions of lower roles.

| Role | Access Level | Description | Key Capabilities |
|------|--------------|-------------|------------------|
| **OWNER** | Level 4 | Strategic Control | Billing, Delete Org, Transfer Ownership. |
| **ADMIN** | Level 3 | Operational Control | Manage Users, Add Amazon Accounts, System Config. |
| **OPERATOR** | Level 2 | Execution | Run Optimizers, Upload Data, Trigger Actions. |
| **VIEWER** | Level 1 | Read-Only | View Dashboards, Download Reports. |

#### 9.3.2 Login & Authentication
- **Organization-Scoped**: Users belong to exactly one Organization.
- **Authentication**: Email + Password.
- **Session**: Stateful session tracking current User, Role, and Active Account permissions.

### 9.4 Access Control Logic

#### 9.4.1 Global vs. Account-Specific Access
By default, a user's **Global Role** applies to ALL Amazon accounts in the organization.

#### 9.4.2 Account Access Overrides (Phase 3.5)
To support Agency use cases (e.g., restricting interns from VIP clients), Admins can set **Account-Specific Overrides**.
- **Downgrade Only**: Overrides can only *reduce* permissions (e.g., OPERATOR → VIEWER). They cannot grant more access than the Global Role.
- **Resolution**: `Effective_Permission = MIN(Global_Role, Override_Role)`

| Global Role | Account Override | Effective Access | Scenario |
|-------------|------------------|------------------|----------|
| OPERATOR | NONE (Default) | OPERATOR | Standard workflow |
| OPERATOR | VIEWER | VIEWER | Intern on VIP client |
| OPERATOR | NO_ACCESS | BLOCKED | Partitioned teams |

### 9.5 Workflows

#### 9.5.1 User Invitation
1.  Admin enters email & selects Global Role.
2.  System sends invite link.
3.  User sets password & joins.
4.  Billing updates automatically (new billable seat).

#### 9.5.2 Permission Management
- Admins can modify Global Roles or set Account Overrides at any time via the "Team Settings" UI.
- Changes take effect immediately (requiring session refresh for active users).

---

## 10. Open Issues & Backlog

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
| **Confidence Weight** | Dampening factor (0-1) based on before_clicks / 15, reduces noise from low-data decisions |
| **Final Decision Impact** | Weighted impact = decision_impact × confidence_weight |
| **Impact Tier** | Classification: Excluded (<5 clicks), Directional (5-14), Validated (15+) |
| **actual_after_days** | Calendar days from action_date to latest data (not report file count) |
| **Z-Score Confidence** | Statistical confidence based on standard error of mean impact |

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
