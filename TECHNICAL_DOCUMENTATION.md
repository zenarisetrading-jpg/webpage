# PPC Optimizer - Technical Documentation

**Version**: 2.0  
**Last Updated**: December 25, 2025

---

## 1. Technology Stack

### 1.1 Current Architecture (V4)
The current system is a high-performance Python application utilizing server-side rendering for complex data processing and visualization.

*   **Language**: Python 3.9+
*   **UI Framework**: Streamlit (Reactive server-side UI)
*   **Data Processing**: Pandas (Vectorized operations for large-scale analysis)
*   **Visualization**: Plotly (Interactive financial & trend charts)
*   **Database**: PostgreSQL 15+ (Persistent storage for account history & actions)
*   **DB Interface**: Psycopg2 / SQLAlchemy (Blocking & non-blocking queries)
*   **External APIs**: Amazon Advertising Bulk API (Via manual upload), Rainforest API (ASIN/Marketplace enrichment)

### 1.2 Future Architecture (V5 Roadmap)
Transitioning to a modern decoupled full-stack architecture for enhanced scalability and multi-user support.

*   **Frontend**: React.js 18+
    *   **Styling**: Tailwind CSS & Shadcn/UI (Design System)
    *   **State Management**: React Query (Data fetching) & Zustand (Global state)
*   **Backend**: Python (FastAPI)
    *   **Framework**: FastAPI (Asynchronous, Type-safe REST API)
    *   **Task Queue**: Celery + Redis (Asynchronous execution for heavy optimization runs)
*   **Communication**: RESTful API (JSON payloads)
*   **Infrastructure**: Dockerized microservices

---

## 2. Backend Structure & Data Model

### 2.1 Database Schema (PostgreSQL)

The system utilizes a relational schema optimized for time-series advertising data.

#### **Table: `target_stats` (Granular Performance)**
Primary storage for aggregated search term and keyword performance.
| Column | Type | Constraints | Purpose |
| :--- | :--- | :--- | :--- |
| `id` | SERIAL | PRIMARY KEY | Unique record ID |
| `client_id` | VARCHAR | REFERENCES accounts | Account owner |
| `start_date` | DATE | INDEXED | Week start (Monday) |
| `campaign_name` | VARCHAR | INDEXED | Raw normalized campaign name |
| `ad_group_name` | VARCHAR | | Raw normalized ad group name |
| `target_text` | TEXT | INDEXED | Keyword, Search Term, or PT expression |
| `match_type` | VARCHAR | | exact, broad, phrase, auto, pt |
| `spend` | DECIMAL | | Total ad spend |
| `sales` | DECIMAL | | Attributed sales |
| `clicks` | INTEGER | | Total clicks |
| `orders` | INTEGER | | Conversion count |

#### **Table: `actions_log` (Optimizer History)**
Audit trail for every optimization recommendation implemented.
| Column | Type | Purpose |
| :--- | :--- | :--- |
| `action_id` | UUID | PRIMARY KEY |
| `client_id` | VARCHAR | Account reference |
| `action_date` | TIMESTAMP | When action was logged |
| `action_type` | VARCHAR | NEGATIVE_ISOLATION, BID_ADJUST, etc. |
| `campaign_name` | VARCHAR | Target campaign |
| `target_text` | TEXT | Optimized term |
| `before_val` | DECIMAL | e.g., Old bid |
| `after_val` | DECIMAL | e.g., New bid |

#### **Secondary Tables**
*   **`accounts`**: Client metadata, target ACoS settings, currency.
*   **`bulk_mappings`**: Synchronization between Campaign Names and Amazon IDs.
*   **`category_mappings`**: Logical grouping of SKUs for roll-up reporting.

### 2.2 API Communication Layer (Current)
Currently, communication is handled via Streamlit's internal RPC:
1.  **Request**: User interacts with UI (slider change, button click).
2.  **Processing**: Streamlit server triggers Python callback.
3.  **Data Retrieval**: Python script queries PostgreSQL via `DBManager`.
4.  **Serialization**: DataFrames are processed and rendered directly to HTML/JS components.

### 2.3 API Communication Layer (Future REST)
1.  **Auth**: JWT-based authentication via FastAPI.
2.  **Endpoints**:
    *   `GET /api/v1/performance`: Returns time-series data for snapshots.
    *   `POST /api/v1/optimize`: Triggers asynchronous optimization engine.
    *   `GET /api/v1/actions`: Retrieves implemented action history for impact analysis.
3.  **Format**: All communication via standard JSON.

---

## 3. Core Engine Logic (`features/optimizer.py`)

### 3.1 Data Preparation Pipeline
Before optimization, data undergoes strict normalization in `prepare_data()`:
*   **ASIN Cleaning**: Strips `asin="B0..."` prefixes to unified 10-char strings.
*   **Match Type Inference**: Detects `pt`, `category`, and `auto` types from targeting expressions.
*   **Currency Normalization**: Ensures all metrics are cast to floating-point numbers.

### 3.2 Dynamic Benchmarking
**Function**: `calculate_account_benchmarks()`
Establishes statistical thresholds based on current account performance rather than static rules.
*   **Expected Clicks**: Calculated as `1 / Account_CVR`.
*   **Negation Floors**: `Soft Stop = Expected Clicks * 2`, `Hard Stop = Expected Clicks * 3`.

---

## 4. Impact & Measurement Methodology

### 4.1 Before/After Windowing
The `Impact Dashboard` employs a dynamic windowing algorithm:
*   **T0**: Reference date of optimization action.
*   **Baseline**: N-days before T0 (Pre-optimisation).
*   **Measurement**: N-days after T0 (Post-optimisation).
*   **Delta**: Realized change in ROAS, Spend, and Sales specifically for affected targets.

### 4.2 CPC-Based Validation
Actions are validated using a **CPC matching algorithm** to confirm implementation:
*   **Before CPC**: `before_spend / before_clicks`
*   **After CPC**: `after_spend / after_clicks`
*   **Validation**: `after_cpc` must be within ±30% of `suggested_bid` (new_value)
*   **Minimum Data**: Requires 5+ clicks in both periods for reliable ROAS-based impact

**Validation Statuses:**
*   `✓ CPC Validated` — After CPC matches suggested bid
*   `✓ Directional Match` — Spend moved in expected direction
*   `Not validated` — Changes not confirmed

### 4.3 Incremental Revenue Calculation (ROAS-Based)
The primary impact metric uses the classic incrementality formula:

```
Incremental Revenue = Before_Spend × (ROAS_After - ROAS_Before)
```

This measures **efficiency gain** — how much *extra* revenue per AED spent — rather than raw sales delta, which may include organic/seasonal factors.

**Capping**: ROAS-based impact is capped at 2× the actual `delta_sales` to prevent inflation from low-click scenarios.

### 4.4 Dashboard Visualization

#### Hero Tiles
| Metric | Calculation |
|--------|-------------|
| **Actions** | Total validated optimization runs |
| **ROAS Lift** | `(roas_after - roas_before) / roas_before × 100%` |
| **Revenue Impact** | `before_spend × (roas_after - roas_before)` |
| **Implementation** | `validated_actions / total_actions × 100%` |

#### ROAS Contribution Waterfall
Breaks down incremental revenue by **match type** (AUTO, BROAD, EXACT, etc.):
*   Each bar shows contribution: `before_spend × (roas_after - roas_before)` per type
*   Contributions are **scaled proportionally** so the total exactly matches the hero tile
*   **Colors**: Brand Purple (#5B556F), Slate (#8F8CA3), Cyan for total (#22d3ee)

#### Revenue Comparison (Stacked Bar)
*   **Before**: Baseline revenue from validated actions
*   **After**: Baseline + ROAS-based Incremental (stacked)
*   Uses consistent `incremental_revenue` from summary

### 4.5 Validation Toggle
A **universal toggle** ("Validated Only") at the top of the dashboard filters:
*   Hero tile metrics
*   All charts
*   Drill-down table

When ON: Shows only CPC-validated actions (conservative, attributable)
When OFF: Shows all actions including unvalidated and pending

### 4.6 Deduplication Engine
When terms appear across multiple campaigns (Bleeders), the impact model uses a **Winner-Take-All** attribution to ensure incremental gains aren't double-counted across account-wide totals.

---

## 5. Security & Maintenance

*   **Database Security**: Row Level Security (RLS) implementation planned for client_id isolation.
*   **Environment Config**: Managed via `.env` for DB credentials and API keys.
*   **Testing**: Unit tests for bid algorithms in `tests/test_optimizer.py`.

---

## Changelog

### December 25, 2024
*   **Impact Dashboard Redesign**:
    *   Added CPC-based validation for action confirmation
    *   Changed incremental revenue to ROAS-based formula
    *   Added match type breakdown waterfall chart
    *   Universal validation toggle for all views
    *   Brand color palette alignment (Purple/Slate/Cyan)
    *   Consistent numbers across hero tiles, waterfall, and stacked bar
