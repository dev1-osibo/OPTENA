# Optena — Data Centre Energy Optimization (demo application)

A Streamlit dashboard for data-centre energy analysis: KPI reporting, Prophet-based
energy and renewable forecasting, IT equipment inventory and optimisation, efficiency
recommendations, and a physical-layer cascade-failure module (DCPLOS).

- **Entry point:** `optena1.py` (1,601 lines)
- **Runtime:** CPython 3.14.4, Streamlit 1.65
- **Data:** supplied by the user at runtime via upload. The app ships a sample file,
  `synthetic_data_center_data.csv`.
- **External services:** none. The app makes no network calls, reads no database,
  and requires no API keys or credentials.

## Contents

| Path | Purpose |
|---|---|
| `optena1.py` | the application |
| `dcplos_engine.py` | cascade-simulation engine — **required import**, Tab 5 fails without it |
| `requirements.txt` | dependency lock, verified on CPython 3.14.4 |
| `synthetic_data_center_data.csv` | sample dataset to upload |
| `optena.jpg`, `favicon.ico` | branding assets loaded by relative path |
| `.streamlit/config.toml` | server and theme configuration |
| `docs/screenshots/` | reference screenshots of each tab |
| `deploy/` | systemd unit, nginx config and deployment runbook |

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run optena1.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

> The app loads `optena.jpg` and `favicon.ico` by **relative path**, so it must be
> started with the repository root as the working directory.

## Important: the app starts empty

Four of the five tabs render only after data is loaded. On first open you will see
`📊 Please upload data to view recommendations` and similar notices. That is expected
behaviour, not a fault.

**Load data first:** sidebar → **📁 Data Upload** → upload `synthetic_data_center_data.csv`.

## The five tabs

| Tab | What it shows |
|---|---|
| 1. Dashboard | KPIs, energy consumption over time, renewable availability, pricing, equipment power distribution, alerts |
| 2. Forecasting | Prophet forecasts for energy demand and renewable availability |
| 3. Equipment Inventory & Optimization | editable inventory, utilisation and efficiency grades, optimisation run |
| 4. Recommendations | scheduling windows, equipment and infrastructure opportunities, water conservation, LBNL checklist, **report download** |
| 5. DCPLOS | cascade-failure simulation, cascade risk scores, facility risk heatmap, risk-aware workload placement, unlock moves |

## Getting reports out

There are three export routes. See `REVIEWER_GUIDE.md` for the exact click path.

1. **Text report** — Tab 4 → `⬇️ Download Report` → `optena_report_<timestamp>.txt`.
   Plain text, covering KPIs, optimal scheduling windows, optimisation results and
   equipment inventory. Requires data to be uploaded first, and the optimisation to
   have been run if you want the savings section populated.
2. **Print to PDF** — sidebar → `🖨️ Print Mode`, which hides the sidebar and triggers
   the browser print dialogue. Choose "Save as PDF". This is the best route for a
   full-dashboard artefact.
3. **On-screen tables** — six tables across the tabs. Note there is **no CSV export
   button** on any of them; they can only be copied or screenshotted.

## Known limitations

- The downloadable report is plain text only; there is no PDF or Excel writer.
- No CSV export on the data tables.
- The "Presentation Demo" feature writes static images via `kaleido`; this is pinned in
  `requirements.txt` but was missing from the original dependency list.
- Tab 5 (DCPLOS) is a simulation engine. Its cascade model and risk scores are
  illustrative; see the note in `REVIEWER_GUIDE.md` before drawing conclusions from it.

## Provenance note

This repository contains the demo application only. The research pipeline that shares
the "Optena" name — the reinforcement-learning and forecasting benchmark — is a
separate body of work with its own results, archives and caveats, and is not required
to run or review this app.
