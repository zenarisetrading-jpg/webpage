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

We use a "V-Next Bucketed" logic to adjust bids proportionally based on performance deviations from the median.

### The Alpha Factor
We use an `Alpha` coefficient (default: 0.15) to moderate bid changes. This prevents massive swings in bids that could destabilize the Amazon algorithm.

### Formula:
`New_Bid = Current_Bid * [1 + (ROAS_Deviation * Alpha)]`

Where `ROAS_Deviation` is `(Row_ROAS / Baseline_ROAS) - 1`.

### Bucketing Logic (The 4 Tabs)
1.  **Direct (Exact/PT)**: Optimized at the individual target level.
2.  **Aggregated (Broad/Phrase)**: Search terms are grouped by their parent KeywordId. The total Spend/Sales for those terms determines the bid for that keyword.
3.  **Auto/Category**: Combined into a single logic block to ensure Auto targeting receives the same statistical rigor as manual keywords.

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
