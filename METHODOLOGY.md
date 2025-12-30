# PPC Suite V4 - Optimization Methodology

This document outlines the mathematical models, statistical thresholds, and logical reasoning behind the PPC Suite optimization engine.

## 1. Benchmarking & Data Normalization

### Outlier-Resistant Median ROAS (Winsorized)
To ensure that account-level benchmarks are not skewed by "lucky" low-spend search terms or extreme outliers, we use a robust statistical approach:

1.  **Substantial Data Filter**: We ignore any search terms with less than **$5.00 in total spend**. This filters out low-volume noise that can generate artificially high ROAS.
2.  **Winsorization**: We calculate the **99th percentile** of ROAS for all remaining substantial rows. Any values above this cap are clipped to the 99th percentile. 
    *   *Reasoning*: This preserves the distribution while neutralizing the impact of rare, extreme successes that aren't representative of the account's baseline.
3.  **Account Baseline**: The median of this Winsorized dataset becomes the `Universal Median ROAS`, which serves as the anchor for bid adjustments.

---

## 2. Negative Keyword Detection

We use a two-tiered "Dynamic Hard Stop" strategy to identify bleeders.

### CVR-Based Thresholds
Instead of using a static click limit, we calculate thresholds based on the **Account Conversion Rate (CVR)**. These benchmarks are user-controllable via the **"Negative Blocking"** sidebar panel:

1.  **Expected Clicks**: `1 / Account_CVR`. (e.g. If CVR is 5%, you expect a sale every 20 clicks).
2.  **Soft Negative Threshold**: `max(Config_Min_Clicks, Expected Clicks)`.
3.  **Hard Stop Threshold**: `max(Config_Min_Spend, Expected Clicks * 2.5)`.
    *   *Reasoning*: This ensures we don't prematurely negate low-volume terms while strictly stopping proven failures.

### Isolation Strategy
We implement a "Winner-Takes-All" isolation model. 
1.  When a search term is "harvested" (moved to an Exact Match campaign), we automatically generate **Negative Exact** recommendations for that term in all other campaigns (Auto, Broad, Phrase).
2.  *Reasoning*: This prevents multiple campaigns from bidding on the same term, ensuring the "Winner" campaign receives 100% of the data and budget for that term.

---

## 3. Bid Optimization Logic

### Bid Baseline Selection
To ensure stability and respect account structure:
- **Ad Group Default Bid**: Used as the primary baseline for bid adjustments where specific keyword-level bids are missing or undefined.
- **Current Bid**: Used for individual targets with existing granular overrides.
- **Safety Floor**: A hard floor of **AED 0.30** is applied to all recommendations to ensure viability on the Amazon marketplace.

### Formula:
`New_Bid = Baseline_Bid * [1 + (ROAS_Deviation * Alpha)]`

Where `ROAS_Deviation` is `(Row_ROAS / Baseline_ROAS) - 1`.

### Bucketing Logic (The 4 Tabs)
1.  **Direct (Exact/PT)**: Optimized at the individual target level.
2.  **Aggregated (Broad/Phrase)**: Search terms are grouped by their parent KeywordId. The total Spend/Sales for those terms determines the bid for that keyword.
3.  **Auto/Category**: Combined into a single logic block to ensure Auto targeting receives the same statistical rigor as manual keywords.

### Visibility Boost (NEW - Dec 2025)

Targets with **low impressions over 2+ weeks** are not competitive in auctions - their bids are too low to win placements.

**Trigger Conditions:**
- Data window ≥ 14 days (sufficient time to judge)
- Impressions < 100 (not winning auctions)
- Impressions > 0 (not paused)
- Match Type = **Exact, Phrase, Broad, or Close-match** only

**NOT eligible (Amazon decides relevance):**
- loose-match, substitutes, complements
- ASIN targeting (product targeting)
- Category targeting

**Action:** Increase bid by **30%** to gain visibility in auctions.

**Rationale:** High impressions + low clicks = CTR problem (ad quality). LOW impressions = bid problem (not competitive). We only boost the latter for explicitly chosen keywords.

---

## 4. Currency-Neutral Thresholds (Dec 2025 Update)

To support multi-region accounts (USD, AED, SAR, etc.), all thresholds are now **clicks-based** rather than currency-based:

| Old Threshold (REMOVED) | New Threshold |
|-------------------------|---------------|
| HARVEST_SALES = $150 | ❌ Removed - uses clicks/orders only |
| NEGATIVE_SPEND_THRESHOLD = $10 | ❌ Removed - uses clicks only |

**Why:** A $10 threshold makes sense in USD but is too low for AED and too high for INR. Clicks-based thresholds work universally.

---

## 4. Harvest Detection (The "Golden Terms")

Candidates are identified based on three criteria, all of which are configurable via the **"Harvest Graduation"** sidebar panel:

1.  **Relative Efficiency**: `ROAS >= (Baseline_ROAS * Config_Multiplier)`. 
    *   *Example*: If your account baseline is 4.0x and your multiplier is 80%, the threshold is 3.2x.
2.  **Volume**: `Spend >= Config_Min_Sales` and `Clicks >= Config_Min_Clicks`.
3.  **Uniqueness**: The term must NOT already be running as an active Exact Match keyword.

---

## 5. Simulation & Impact Forecasting

Our simulator uses a **Curved Elasticity Model** to project the outcome of recommended changes.

| Factor | Relationship | Coefficient |
| :--- | :--- | :--- |
| **CPC vs Clicks** | Diminishing Returns | 0.85 (Increases in bid yield 0.85x growth in clicks) |
| **Sales vs Spend** | Variable Efficiency | Calculates "Efficiency Delta" based on Harvest/Negate ratio |
| **ACoS Impact** | Mathematical | `(New_Spend / New_Sales) * 100` |

### Probability Scenarios
- **Conservative**: High bid skepticism, lower click growth.
- **Expected**: Balanced historical averages.
- **Aggressive**: High confidence in harvest efficiency.

---

## 6. Verified Impact Methodology (Rule-Based)

To prevent the "Inflation of Success" common in many ad optimizers, we use a conservative **Rule-Based Impact Logic** that attributes value only to specific, verifiable outcomes of an action.

### 1. The Attribution Rules
Impact is not based on total account fluctuations, but on the specific delta created by each action type:

| Action Type | Impact Calculation (Rule) | Rationale |
| :--- | :--- | :--- |
| **Negatives** | `+Before Spend` | Total cost avoidance of previously wasteful spend. |
| **Harvests** | `+10% Net Sales Lift` | Assumes a conservative 10% efficiency gain from exact match isolation. |
| **Bid Changes** | `(Sales Delta) - (Spend Delta)` | Net profit change from the observed shift in performance. |
| **Visibility Boost** | `(Sales Delta) - (Spend Delta)` | Same as bid changes - measures incremental traffic & sales from better auction wins. |
| **Pauses** | `(Sales Delta) - (Spend Delta)` | Total dollar impact of removing the entity from the mix. |

### 2. Verified Deduplication
To prevent overcounting (e.g., when a search term is negated in one campaign but exists in another), we apply a high-fidelity **Deduplication Engine**:
- **Key Matching**: We group actions by `campaign_name` + `action_type` + `before_spend` + `before_sales`.
- **Logic**: If the same impact value is spotted across multiple records for the same campaign, we count it only once.
- **Outcome**: The `Net Result` hero tile is an **additive sum** of these unique, verified impacts.

### 3. Comparison Windows
Impact is calculated by comparing a **"Before" period** (the data upload immediately preceding the action) to an **"After" period** (the most recent data upload). This ensures that we are always comparing like-for-like performance windows based on your actual data availability.

### 4. Direct Validation
We don't just "guess" impact. We confirm it:
- **Confirmed Blocked**: For negatives, we verify that subsequent spend is actually $0.00.
- **Source Isolated**: For harvests, we confirm the source campaign stopped bidding on the term.
- **Observed Data**: For bids, we use actual spend/sales shifts from the Ads Console.

### 5. Multi-Horizon Impact Measurement

We measure impact at three horizons to balance speed vs accuracy:

| Horizon | After Window | Maturity | Purpose |
|---------|--------------|----------|---------|
| **14D** | 14 days | 17 days | Early signal — did the action have an effect? |
| **30D** | 30 days | 33 days | Confirmed — is the impact sustained? |
| **60D** | 60 days | 63 days | Long-term — did the gains hold? |

**Why not 7 days?**
Amazon's attribution window is 7-14 days. Measuring at 7 days produces incomplete data and false negatives. Bid increases especially need 10-14 days to show effect.

### Maturity Formula

```
is_mature(horizon) = (action_date + horizon_days + 3) ≤ latest_data_date

Where horizon_days = {
    "14D": 14,
    "30D": 30,
    "60D": 60
}
```

- **Before window**: Always 14 days (fixed baseline)
- **After window**: 14, 30, or 60 days (per horizon)
- **Buffer**: 3 days for attribution to settle
- Actions not yet mature are shown as "Pending" for that horizon
