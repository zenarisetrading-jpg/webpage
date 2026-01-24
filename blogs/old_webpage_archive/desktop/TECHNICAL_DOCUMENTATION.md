# PPC Optimizer - Technical Documentation

**Version**: 3.0  
**Last Updated**: January 6, 2026

---

## 1. Technology Stack

### 1.1 Current Architecture (V4)
High-performance Python application with server-side rendering.

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.9+ |
| **UI Framework** | Streamlit (Reactive server-side UI) |
| **Data Processing** | Pandas (Vectorized operations) |
| **Visualization** | Plotly (Interactive charts) |
| **Database** | PostgreSQL 15+ |
| **DB Interface** | Psycopg2 / SQLAlchemy |
| **External APIs** | Amazon Bulk API (manual upload), Rainforest API (ASIN enrichment) |
| **Statistics** | SciPy (z-score confidence calculations) |

### 1.2 Future Architecture (V5 Roadmap)
Decoupled full-stack for scalability and multi-user support.

* **Frontend**: React.js 18+ with Tailwind CSS & Shadcn/UI
* **Backend**: FastAPI (async, type-safe REST API)
* **Task Queue**: Celery + Redis
* **Infrastructure**: Dockerized microservices

---

## 2. Backend Structure & Data Model

### 2.1 Database Schema (PostgreSQL)

#### **Table: `target_stats` (Granular Performance)**
Primary storage for aggregated search term and keyword performance.

| Column | Type | Purpose |
|--------|------|---------|
| `id` | SERIAL | PRIMARY KEY |
| `client_id` | VARCHAR | Account owner (REFERENCES accounts) |
| `start_date` | DATE | Week start (Monday), INDEXED |
| `campaign_name` | VARCHAR | Normalized campaign name, INDEXED |
| `ad_group_name` | VARCHAR | Normalized ad group name |
| `target_text` | TEXT | Keyword, Search Term, or PT expression, INDEXED |
| `customer_search_term` | TEXT | Raw CST for matching (added Jan 2026) |
| `match_type` | VARCHAR | exact, broad, phrase, auto, pt |
| `spend` | DECIMAL | Total ad spend |
| `sales` | DECIMAL | Attributed sales |
| `clicks` | INTEGER | Total clicks |
| `orders` | INTEGER | Conversion count |
| `impressions` | INTEGER | Total impressions |

#### **Table: `actions_log` (Optimizer History)**
Audit trail for every optimization recommendation.

| Column | Type | Purpose |
|--------|------|---------|
| `action_id` | UUID | PRIMARY KEY |
| `client_id` | VARCHAR | Account reference |
| `action_date` | TIMESTAMP | When action was logged |
| `action_type` | VARCHAR | NEGATIVE, NEGATIVE_ADD, HARVEST, BID_CHANGE |
| `campaign_name` | VARCHAR | Target campaign |
| `ad_group_name` | VARCHAR | Target ad group |
| `target_text` | TEXT | Optimized term |
| `match_type` | VARCHAR | Keyword match type |
| `old_value` | DECIMAL | Previous bid/state |
| `new_value` | DECIMAL | New bid/state |
| `reason` | TEXT | Optimization rationale |

#### **Secondary Tables**
* **`accounts`**: Client metadata, target ACoS, currency
* **`bulk_mappings`**: Campaign Name ↔ Amazon ID sync
* **`category_mappings`**: SKU groupings for roll-up reporting

---

## 3. Impact Calculation Engine

### 3.1 Before/After Windowing
Dynamic windowing anchored to action date:

```
T0 = action_date
Baseline = T0 - 14 days (pre-optimization)
Measurement = T0 + 14 days (post-optimization)
```

### 3.2 Refined Attribution Framework (Decomposition)
Implemented in [`desktop/scripts/refined_attribution.py`](file:///Users/zayaanyousuf/Documents/Amazon%20PPC/saddle/saddle/desktop/scripts/refined_attribution.py).
Decomposes total ROAS change into:
1.  **Decision Impact**: From `actions_log` (sum of validated impacts).
2.  **Market Forces**: Calculated via CPC/CVR shifts (`calculate_cpc_impact`, `calculate_cvr_impact`).
3.  **Scale Effect**: `estimate_scale_effect` (approx -0.5% ROAS per +10% Spend).
4.  **Portfolio Effect**: `estimate_portfolio_effect` (efficiency drag from new campaigns).
5.  **Residual**: `Total Δ - (Sum of Above)`.

### 3.3 Maturity Calculation (Fixed Jan 2026):
```sql
-- Calendar days from action to latest data (not report count)
actual_after_days = latest_date - action_date + 1

-- Action is "mature" if full window is available
is_mature = actual_after_days >= 14
```

### 3.4 Confidence Weighting (Added Jan 2026)
Low-click decisions are dampened to prevent noise:

| Column | Formula | Purpose |
|--------|---------|---------|
| `confidence_weight` | `min(1.0, before_clicks / 15)` | 0-1 weight based on data volume |
| `final_decision_impact` | `decision_impact × confidence_weight` | Dampened impact value |
| `impact_tier` | See below | Classification label |

**Impact Tier Classification:**
| Tier | Criteria | Treatment |
|------|----------|-----------|
| **Excluded** | before_clicks < 5 | Not counted |
| **Directional** | 5 ≤ clicks < 15 | Partial weight (33%-99%) |
| **Validated** | clicks ≥ 15 | Full weight (100%) |

### 3.5 Decision Impact Formula
```
decision_impact = observed_after_sales - expected_after_sales

where:
  expected_after_sales = before_sales × (after_days / before_days)
```

### 3.6 Statistical Confidence (Z-Score Based)
Proper statistical significance testing:

```python
# Calculate z-score
mean_impact = impact_values.mean()
std_error = impact_values.std() / sqrt(n)
z_score = mean_impact / std_error

# Convert to confidence via normal CDF (capped at 99%)
confidence_pct = min(99, stats.norm.cdf(z_score) * 100)
```

| z-score | Label | Meaning |
|---------|-------|---------|
| ≥ 2.58 | Very High | 99% confident |
| ≥ 1.96 | High | 95% confident |
| ≥ 1.645 | Moderate | 90% confident |
| < 1.645 | Directional | < 90% confident |

### 3.7 Incremental Contribution %
Shows what percentage of total revenue optimizations contributed:

```python
incremental_pct = attributed_impact / (before_sales + after_sales) × 100
```

Displayed as badge: `+7.6% of revenue`

---

## 4. Dashboard Components

### 4.1 Hero Banner
| Metric | Source |
|--------|--------|
| **Impact Value** | Sum of `final_decision_impact` (dampened) |
| **Contribution %** | `impact / total_account_sales × 100` |
| **Confidence** | Z-score based (Very High/High/Moderate/Directional) |

### 4.2 Quadrant Breakdown
| Quadrant | Definition |
|----------|------------|
| **Offensive Win** | Increased spend, increased ROAS |
| **Defensive Win** | Decreased spend, maintained ROAS |
| **Gap** | Missed opportunity (lower than expected) |
| **Market Drag** | External factors (excluded from attributed total) |

### 4.3 Validation Statuses
| Status | Meaning |
|--------|---------|
| `✓ CPC Validated` | After CPC matches suggested bid (±30%) |
| `✓ Directional` | Spend moved in expected direction |
| `✓ Volume Match` | Click pattern confirms implementation |
| `Not validated` | Changes not confirmed |

---

## 5. Optimizer Summary Metrics

| Tile | Metric |
|------|--------|
| **Search Terms** | Unique CSTs analyzed from STR data |
| **Bids** | Total bid change recommendations |
| **Negatives** | Negative keyword + PT recommendations |
| **Harvest** | Keywords harvested to exact match |

---

## 6. Security & Maintenance

* **Database Security**: Row Level Security (RLS) planned for client_id isolation
* **Environment Config**: `.env` for DB credentials and API keys
* **Testing**: Unit tests in `tests/` directory
* **Caching**: Streamlit `@st.cache_data` with version-based invalidation

---

## Changelog

### January 6, 2026
* **Confidence Weighting**: Added dampening for low-click decisions
* **Maturity Fix**: `actual_after_days` now uses calendar days, not report count
* **Statistical Confidence**: Z-score based confidence with proper CDF calculation
* **Incremental Badge**: Shows "X% of revenue" contribution in Hero Banner
* **Search Terms Metric**: Replaced "Touched" with unique CST count

### January 1, 2026
* **Impact Dashboard UI Redesign**: Separated Performance Overview from Estimated Impact
* **Query Optimization Revert**: Restored LATERAL joins for accuracy

### December 25, 2025
* **Impact Dashboard Redesign**: CPC-based validation, ROAS-based formula
* **Match Type Waterfall**: Breakdown by exact/broad/auto
* **Brand Colors**: Purple (#5B556F), Slate (#8F8CA3), Cyan (#22d3ee)
