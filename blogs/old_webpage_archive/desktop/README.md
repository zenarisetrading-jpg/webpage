# PPC Suite V4 - The Decision Hub

A high-performance Amazon PPC optimization engine designed for stability, scale, and strategic insight.

## 🚀 Key Features (V4)

- **Impact Dashboard (Storytelling Flow)**: A redesigned waterfall visualizer that narrates the impact from `Actions -> Cost Saved -> Harvest Gains -> Bid Changes -> Net Result`. Includes a dynamic "Storyteller" callout that identifies the primary driver of the period.
- **Rule-Based Impact Attribution**: More conservative and precise calculation logic that rewards spend savings (negatives) and sales lift (harvests) appropriately, with **Verified Deduplication** to prevent overcounting.
- **Unified Decision Hub**: A consolidated dashboard merging Bid Optimization, Harvest Detection, and Negative Identification with **Session Persistence** (settings stay locked across tabs).
- **Premium Design System**: Complete SVG icon architecture and glassmorphism UI overhaul for a modern, professional aesthetic.
- **Simulation Engine**: Forecast the impact of changes before you apply them.
- **Precision Auditing**: A visual heatmap tracking every action taken per campaign/ad-group.

---

## 📂 Project Structure

```
saddle/
├── ppcsuite_v4.py       # Main Entry Point (Consolidated Hub)
├── core/                # Core Data & Logic
│   ├── data_hub.py      # Session management & Database integration
│   ├── db_manager.py    # PostgreSQL/Local DB orchestrator
│   └── data_loader.py   # Raw file ingestion & cleaning
├── features/            # Heavy Business Logic
│   ├── optimizer.py     # Main optimization engine (~2200 lines)
│   ├── creator.py       # Campaign generation & bulk file export
│   ├── asin_mapper.py   # Competitor intelligence
│   ├── kw_cluster.py    # AI-powered keyword analysis
│   └── simulator.py     # Forecasting & elasticity models
├── ui/                  # Premium UI Components
│   ├── layout.py        # Glassmorphism & custom styling
│   └── components.py    # Reusable metric cards & charts
└── utils/               # Helpers
    ├── matchers.py      # ExactMatch lookup logic
    └── formatters.py    # Currency & Percent formatting
```

---

## 🛠 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create `.streamlit/secrets.toml`:
```toml
# API Keys
RAINFOREST_API_KEY = "..."
ANTHROPIC_API_KEY = "..."

# Databases
DATABASE_URL = "postgresql://..."
```

### 3. Usage
```bash
streamlit run ppcsuite.py
```

---

## 📈 Methodology
For detailed mathematical models (Winsorization, Alpha-adjustments, Dynamic CVR Stop), see [METHODOLOGY.md](./METHODOLOGY.md).

## 📄 Technical Specs
For deep architectural details, see [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md).