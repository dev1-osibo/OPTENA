# Reviewer guide

A 15-minute path through the app that exercises every feature and produces two
downloadable artefacts.

## 0. Get in

**Hosted instance** — open the URL you were given and enter the username and password
supplied separately. Nothing to install.

**Or run it yourself:**

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run optena1.py
```

Requires CPython 3.14 (the pinned set has no wheels for older versions). Run from the
repository root — the app loads `optena.jpg` by relative path.

## 1. Load data (do this first)

The app opens empty by design. Four of five tabs show a "please upload data" notice
until you load a dataset.

Sidebar → **📁 Data Upload** → upload **`synthetic_data_center_data.csv`** (included).

The uploader accepts `.csv` and `.h5`. The `.h5` branch needs the optional `tables`
and `h5py` packages, which are commented out in `requirements.txt` — use the CSV.

## 2. Set the scenario

Sidebar → **🏢 Data Center Type** → `Hyperscale`, `Enterprise`, or `Small/Medium`.
Sidebar → **🎛️ Parameters** → adjust and watch the dashboard recompute.

Changing the type changes the KPI baselines, so it is worth doing one pass per type if
you want to probe the sensitivity of the recommendations.

## 3. Walk the tabs

| Step | Tab | What to look at |
|---|---|---|
| 3a | **Dashboard** | Do the KPIs, the 2026 energy series, renewable availability, pricing and equipment power distribution agree with each other? Check the Alerts panel. |
| 3b | **Forecasting** | Prophet forecast for energy and renewables. Judge whether the horizon and confidence bands are presented honestly. |
| 3c | **Equipment Inventory** | Use `➕ Add Equipment` to add a row. Check the utilisation and efficiency grading, and the low-utilisation table. Then run **⚡ Energy Optimization** and pick a strategy from the radio group. |
| 3d | **Recommendations** | Scheduling windows, equipment and infrastructure opportunities, water conservation, LBNL best-practices checklist. |
| 3e | **DCPLOS** | Pick a component in **💥 Cascade Failure Simulation** and run it. Then review Cascade Risk Scores, the facility risk heatmap, risk-aware placement, and unlock moves. |

## 4. Pull out the two artefacts

**Text report.** Tab 4 (Recommendations) → **⬇️ Download Report**.
Produces `optena_report_<timestamp>.txt` containing KPIs, optimal scheduling windows,
optimisation results and equipment inventory totals, plus references to LBNL, ASHRAE
TC 9.9 and DOE guidance.

> Run the optimisation in Tab 3 **before** downloading, or the `OPTIMIZATION RESULTS`
> section of the report will be empty. This is the single most common surprise.

**Full dashboard PDF.** Sidebar → **🖨️ Print Mode**. The sidebar hides itself and the
browser print dialogue opens; choose "Save as PDF". Print CSS is already in place, so
the layout expands to the full page width.

## 5. What is in scope for your review

Fair game:

- Is the analysis logic sound, and are the KPI and savings calculations defensible?
- Are the forecasts presented with appropriate uncertainty?
- Is the optimisation advice actionable and correctly derived from the inputs?
- Is the UI clear, and does the report contain what a facilities or energy manager needs?
- Does anything overstate what the underlying model can support?

Please read these caveats first:

- **The sample dataset is synthetic.** `synthetic_data_center_data.csv` is generated,
  not measured. Numbers in the dashboard demonstrate the mechanism, not real savings.
- **Tab 5 (DCPLOS) is illustrative.** The cascade engine is a simulation. An earlier
  internal review found that one headline property of this engine (equipment-quality
  independence of cascade magnitude) was an artefact of a code defect and was later
  reversed once equipment tier was given a real causal path. Treat the risk scores as a
  demonstration of the mechanism, not as validated predictions.
- **No authentication inside the app.** Streamlit has no user model. Any access control
  is provided by the hosting layer, not by the application.
- **Session state is per-browser-session.** Equipment you add is not persisted; a reload
  resets it. There is no database.

## 6. Reporting back

Useful things to send: the downloaded `.txt` report, the printed PDF, your browser and
OS, and for any defect the tab, the exact click path, and what you expected instead.
