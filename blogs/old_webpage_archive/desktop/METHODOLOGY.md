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
- Includes 0 impressions (bid SO low it can't even enter auctions)
- Note: Paused targets are identified by `state='paused'`, not impressions

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
2.  **Volume**: `Clicks >= Config_Min_Clicks`.
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

---

## 7. Impact Dashboard - Counterfactual Framework (Jan 2026)

### Philosophy: Isolating Decision Quality from Market Conditions

The Impact Dashboard uses a **counterfactual analysis** approach to separate what we CONTROLLED (optimization decisions) from what we DIDN'T (external market trends).

### Decision Outcome Matrix

A 2x2 matrix that plots each action based on:

**X-Axis: Expected Trend %**
- Formula: `(Expected Sales - Before Sales) / Before Sales * 100`
- Expected Sales = `(New Spend / Baseline CPC) × Baseline SPC`
- **Translation**: "If we maintained our old efficiency, what would sales be at the new spend level?"

**Y-Axis: vs Expectation %**
- Formula: `Actual Change % - Expected Trend %`
- **Translation**: "How much did we BEAT or MISS the counterfactual baseline?"

### Quadrants

1. **Offensive Win** (X≥0, Y≥0): Spend increased + beat baseline efficiency → Efficient scaling
2. **Defensive Win** (X<0, Y≥0): Market shrank, but we beat the expected drop → Good defense
3. **Decision Gap** (X≥0, Y<0): Spend increased but missed expectations → Inefficient scale
4. **Market Drag** (X<0, Y<0): Market shrank AND we underperformed → **EXCLUDED from attribution**

### Decision-Attributed Impact (Hero Metric)

**Formula**: `Sum(Offensive Wins + Defensive Wins + Decision Gaps)`

**Critical Exclusion**: **Market Drag is EXCLUDED** from all impact totals.

**Reasoning**:
- Market Drag represents external headwinds we didn't control
- Including it would conflate market luck with decision quality
- We ONLY attribute impact where our DECISION had clear directional influence

**Display Format**:
- Main Number: Net Impact (Green if positive, Red if negative)
- Breakdown: "✅ Wins: +X (Offensive + Defensive) | ❌ Gaps: -Y"
- Footnote: "ℹ️ Z actions excluded (Market Drag — ambiguous attribution)"

### Capital Protected (Refined Logic)

**Definition**: Wasteful spend eliminated from confirmed negative keyword blocks.

**Formula**: `Sum of before_spend for NEGATIVE actions where observed_after_spend == 0`

**Why This Works**:
- Only counts actions **INTENDED** to protect capital (negatives)
- `after_spend == 0` proves the block was successful
- Bid increases **SHOULD** increase spend — that's scaling winners

**Display**: "From X confirmed negatives" + "Confidence: High"

### Key Metric Definitions

| Metric | Definition | Use Case |
|--------|-----------|----------|
| **Verified Impact vs Baseline** | Decision-Attributed Impact (excludes Market Drag) | Prove optimization value |
| **Capital Protected** | Spend saved from confirmed negative blocks | Show waste elimination |
| **Win Rate** | % of actions in Win quadrants | Measure decision quality |
| **Decision Gap** | Actions that scaled but missed efficiency | Identify optimization errors |
| **Market Drag** | Actions confounded by external conditions | Transparent exclusion |

---

## 8. Refined Attribution Framework (ROAS Decomposition)

While **Decision Impact** measures the micro-level success of individual optimization actions, **ROAS Decomposition** explains the macro-level movement of the entire account's efficiency.

We decompose the total change in Account ROAS into 5 distinct components:

### 1. Decision Impact (Internal - Controlled)
The verified outcomes of our specific optimization actions (Bids, Harvests, Negatives).
*   **Formula**: `Sum(Validated Action Impacts) / Total Spend`
*   **Meaning**: How much value did *we* add through direct intervention?

### 2. Market Forces (External - Uncontrolled)
Changes driven by external market conditions (CPC inflation, conversion rate shifts, AOV changes).
*   **CPC Impact**: `(Prior ROAS) × (CPC_Change_Pct * -1)` (Higher CPC = Lower ROAS)
*   **CVR Impact**: `(Prior ROAS) × (CVR_Change_Pct)` (Higher CVR = Higher ROAS)
*   **AOV Impact**: `(Prior ROAS) × (AOV_Change_Pct)` (Higher AOV = Higher ROAS)

### 3. Scale Effect (Internal - Strategic)
The natural efficiency loss that comes from aggressively scaling spend.
*   **Logic**: "Diminishing Returns" curve.
*   **Formula**: `If Spend increases >20%, assume 0.5% ROAS drop for every 10% spend hike.`
*   **Meaning**: If we spent 50% more, ROAS *should* drop slightly. This isn't "bad performance," it's the cost of growth.

### 4. Portfolio Effect (Internal - Structural)
Changes driven by launching new campaigns or shifting budget mix.
*   **Logic**: New campaigns (Launch Phase) typically run at 60-70% of the efficiency of mature campaigns.
*   **Formula**: `(New Campaign Spend % of Total) × (1 - New_Campaign_Efficiency_Factor) × Baseline_ROAS`
*   **Meaning**: Launching 10 new products will drag down account ROAS temporarily. This isolates that structural drag.

### 5. Unexplained Residual
The mathematical variance left over.
*   **Formula**: `Total ROAS Change - (Decision + Market + Scale + Portfolio)`
*   **Target**: Should be <20% of the total change for a high-confidence model.
