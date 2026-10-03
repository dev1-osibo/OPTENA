import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import time
import os

# Set the page configuration
st.set_page_config(
    page_title="OPTENA: Energy Optimization Systems", 
    page_icon="favicon.ico", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create demo_screenshots directory
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# App Header with Logo — fixed center regardless of sidebar
# Print-friendly CSS: hides sidebar and expands content when printing
st.markdown("""
<style>
@media print {
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .stApp > header { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stMainBlockContainer"] { max-width: 100% !important; padding: 0 10px !important; }
    section[data-testid="stMain"] { margin-left: 0 !important; width: 100% !important; }
    
    /* Each chart section starts on a new page */
    .print-page-break {
        page-break-before: always !important;
        break-before: page !important;
    }
    
    /* Let charts fill the page naturally — no height restriction */
    [data-testid="stPlotlyChart"] {
        width: 100% !important;
        break-inside: avoid !important;
        page-break-inside: avoid !important;
    }
    
    /* Keep annotation boxes from splitting */
    div[style*="border-left"] {
        break-inside: avoid !important;
        page-break-inside: avoid !important;
        margin-top: 8px !important;
    }
    
    /* Keep metric rows together */
    [data-testid="stHorizontalBlock"] {
        break-inside: avoid !important;
        page-break-inside: avoid !important;
    }
    
    /* Hide interactive elements in print */
    .stPlotlyChart .modebar-container { display: none !important; }
    button, [data-testid="stFileUploader"] { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align: center;'></div>",
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    st.image("optena.jpg", width=300)

st.markdown(
    "<h2 style='text-align: center; color: #555; margin-top: -10px;'>AI-Powered Data Center Energy Optimization Platform</h2>",
    unsafe_allow_html=True
)

# Initialize session state
if 'equipment_inventory' not in st.session_state:
    st.session_state.equipment_inventory = pd.DataFrame(columns=[
        'Equipment Type', 'Category', 'Count', 'Power per Unit (W)', 
        'Max Power (W)', 'Utilization (%)', 'Performance Level'
    ])

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False

if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 0

if 'demo_active' not in st.session_state:
    st.session_state.demo_active = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_data(file, file_type="csv"):
    """Load data from CSV or HDF5 file"""
    if file_type == "csv":
        data = pd.read_csv(file)
    elif file_type == "h5":
        data = pd.read_hdf(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or HDF5 file.")
    
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
    return data

def standardize_columns(data):
    """Standardize column names for consistency"""
    required_columns = {
        'Renewable Availability (%)': ['Renewable Availability', 'Renewables (%)'],
        'Workload Energy Consumption (kWh)': ['Workload Consumption', 'Energy (kWh)'],
        'Energy Price ($/kWh)': ['Energy Price', 'Price ($)'],
    }
    for standard_col, possible_names in required_columns.items():
        for col in possible_names:
            if col in data.columns:
                data.rename(columns={col: standard_col}, inplace=True)
                break
    missing_columns = [col for col in required_columns.keys() if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()
    return data

def calculate_pue(it_power, cooling_power, power_distribution_loss=0.05):
    """Calculate Power Usage Effectiveness (PUE)"""
    total_facility_power = it_power + cooling_power + (it_power * power_distribution_loss)
    pue = total_facility_power / it_power if it_power > 0 else 1.0
    return round(pue, 2)

def calculate_equipment_power(inventory_df):
    """Calculate total power consumption from equipment inventory"""
    if inventory_df.empty:
        return 0
    total_power = 0
    for _, row in inventory_df.iterrows():
        actual_power = row['Power per Unit (W)'] * row['Count'] * (row['Utilization (%)'] / 100)
        total_power += actual_power
    return round(total_power / 1000, 2)

def get_equipment_efficiency_grade(utilization, performance_level):
    """Calculate efficiency grade for equipment (A-F scale)"""
    if performance_level == 'High':
        if utilization >= 70: return 'A'
        elif utilization >= 50: return 'B'
        elif utilization >= 30: return 'C'
        elif utilization >= 15: return 'D'
        else: return 'F'
    elif performance_level == 'Mid':
        if utilization >= 60: return 'A'
        elif utilization >= 40: return 'B'
        elif utilization >= 25: return 'C'
        elif utilization >= 10: return 'D'
        else: return 'F'
    else:
        if utilization >= 50: return 'A'
        elif utilization >= 35: return 'B'
        elif utilization >= 20: return 'C'
        elif utilization >= 10: return 'D'
        else: return 'F'

def forecast_prophet(data, columns, periods=365*24):
    """Generate forecasts using Prophet"""
    forecasts = {}
    for column in columns:
        prophet_data = data.reset_index()[['Timestamp', column]].rename(
            columns={'Timestamp': 'ds', column: 'y'}
        )
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=periods, freq='h')
        forecast = model.predict(future)
        forecasts[column] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return forecasts

def run_simulation(data, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable):
    """Run energy optimization simulation"""
    data = data.copy()
    data['Baseline Emissions'] = (
        data['Workload Energy Consumption (kWh)'] * emission_factor_non_renewable * (1 - data['Renewable Availability (%)'] / 100) +
        data['Workload Energy Consumption (kWh)'] * emission_factor_renewable * (data['Renewable Availability (%)'] / 100)
    )
    data['Optimized Energy'] = np.where(
        data['Renewable Availability (%)'] >= renewable_threshold * 100,
        data['Workload Energy Consumption (kWh)'],
        data['Workload Energy Consumption (kWh)'] * 0.9
    )
    data['Optimized Emissions'] = (
        data['Optimized Energy'] * emission_factor_non_renewable * (1 - data['Renewable Availability (%)'] / 100) +
        data['Optimized Energy'] * emission_factor_renewable * (data['Renewable Availability (%)'] / 100)
    )
    results = {
        'optimized_energy': round(data['Optimized Energy'].sum(), 2),
        'optimized_cost': round((data['Optimized Energy'] * energy_price_per_kwh).sum(), 2),
        'optimized_emissions': round(data['Optimized Emissions'].sum(), 2),
        'energy_savings': round(data['Workload Energy Consumption (kWh)'].sum() - data['Optimized Energy'].sum(), 2),
        'cost_savings': round(
            (data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum() - 
            (data['Optimized Energy'] * energy_price_per_kwh).sum(), 2
        ),
        'emissions_savings': round(data['Baseline Emissions'].sum() - data['Optimized Emissions'].sum(), 2),
        'baseline_energy': round(data['Workload Energy Consumption (kWh)'].sum(), 2),
        'baseline_cost': round((data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum(), 2),
        'baseline_emissions': round(data['Baseline Emissions'].sum(), 2),
        'data': data
    }
    return results

def calculate_water_consumption(energy_kwh, pue=1.5, water_per_kwh=1.8):
    """Calculate water consumption (liters) based on energy use"""
    total_facility_energy = energy_kwh * pue
    water_consumption = total_facility_energy * water_per_kwh
    return round(water_consumption, 2)

def add_chart_annotations(fig, data_series, title_context=""):
    """Add peak/trough annotations to a Plotly figure for decision context"""
    if len(data_series) == 0:
        return fig
    
    # Find peak
    peak_idx = data_series.idxmax()
    peak_val = data_series.max()
    
    # Find trough
    trough_idx = data_series.idxmin()
    trough_val = data_series.min()
    
    # Add peak annotation
    fig.add_annotation(
        x=peak_idx, y=peak_val,
        text=f"▲ PEAK: {peak_val:.1f}<br>High demand period",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor="#F44336", font=dict(size=11, color="#F44336"),
        bgcolor="rgba(244,67,54,0.1)", bordercolor="#F44336", borderwidth=1,
        ax=40, ay=-40
    )
    
    # Add trough annotation
    fig.add_annotation(
        x=trough_idx, y=trough_val,
        text=f"▼ TROUGH: {trough_val:.1f}<br>Optimal maintenance window",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor="#4CAF50", font=dict(size=11, color="#4CAF50"),
        bgcolor="rgba(76,175,80,0.1)", bordercolor="#4CAF50", borderwidth=1,
        ax=-40, ay=40
    )
    
    return fig

def save_chart_screenshot(fig, filename):
    """Save a Plotly chart as an image for the demo"""
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    try:
        fig.write_image(filepath, width=1400, height=600, scale=2)
    except Exception:
        # If kaleido not installed, skip silently
        pass


# ============================================================
# SIDEBAR - GLOBAL CONTROLS
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Demo Section - only show button if demo is NOT running
    if not st.session_state.demo_running:
        st.subheader("🚀 Presentation Demo")
        
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button("▶️ Run Demo", use_container_width=True, type="primary"):
                # Load demo equipment
                demo_equipment = pd.DataFrame([
                    {'Equipment Type': 'Dell PowerEdge R750', 'Category': 'Servers', 'Count': 120, 'Power per Unit (W)': 380, 'Max Power (W)': 500, 'Utilization (%)': 72, 'Performance Level': 'High'},
                    {'Equipment Type': 'HPE ProLiant DL380', 'Category': 'Servers', 'Count': 80, 'Power per Unit (W)': 320, 'Max Power (W)': 450, 'Utilization (%)': 55, 'Performance Level': 'High'},
                    {'Equipment Type': 'Lenovo ThinkSystem SR650', 'Category': 'Servers', 'Count': 45, 'Power per Unit (W)': 290, 'Max Power (W)': 400, 'Utilization (%)': 38, 'Performance Level': 'Mid'},
                    {'Equipment Type': 'NetApp AFF A400', 'Category': 'Storage', 'Count': 12, 'Power per Unit (W)': 750, 'Max Power (W)': 1000, 'Utilization (%)': 68, 'Performance Level': 'High'},
                    {'Equipment Type': 'Dell PowerStore 500T', 'Category': 'Storage', 'Count': 8, 'Power per Unit (W)': 550, 'Max Power (W)': 700, 'Utilization (%)': 45, 'Performance Level': 'Mid'},
                    {'Equipment Type': 'Cisco Nexus 9336C', 'Category': 'Network', 'Count': 16, 'Power per Unit (W)': 520, 'Max Power (W)': 680, 'Utilization (%)': 78, 'Performance Level': 'High'},
                    {'Equipment Type': 'Arista 7280R3', 'Category': 'Network', 'Count': 24, 'Power per Unit (W)': 180, 'Max Power (W)': 250, 'Utilization (%)': 42, 'Performance Level': 'Mid'},
                    {'Equipment Type': 'Eaton 9PXM UPS', 'Category': 'Power/UPS', 'Count': 6, 'Power per Unit (W)': 1200, 'Max Power (W)': 1500, 'Utilization (%)': 55, 'Performance Level': 'High'},
                    {'Equipment Type': 'Schneider Galaxy VX', 'Category': 'Power/UPS', 'Count': 4, 'Power per Unit (W)': 950, 'Max Power (W)': 1200, 'Utilization (%)': 48, 'Performance Level': 'Mid'},
                    {'Equipment Type': 'Vertiv Liebert CRV', 'Category': 'Cooling', 'Count': 18, 'Power per Unit (W)': 680, 'Max Power (W)': 850, 'Utilization (%)': 65, 'Performance Level': 'High'},
                ])
                st.session_state.equipment_inventory = demo_equipment
                st.session_state.demo_running = True
                st.session_state.demo_step = 1
                st.session_state.demo_active = True
                st.session_state.optimization_results = None
                st.rerun()
        
        with demo_col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.equipment_inventory = pd.DataFrame(columns=[
                    'Equipment Type', 'Category', 'Count', 'Power per Unit (W)', 
                    'Max Power (W)', 'Utilization (%)', 'Performance Level'
                ])
                st.session_state.optimization_results = None
                st.session_state.demo_active = False
                st.session_state.demo_running = False
                st.session_state.demo_step = 0
                if 'forecasts' in st.session_state:
                    del st.session_state.forecasts
                if 'forecast_data' in st.session_state:
                    del st.session_state.forecast_data
                st.rerun()
    else:
        st.info("🎬 **Demo presentation running...**")
        st.progress(min(st.session_state.demo_step / 7, 1.0))
    
    st.divider()
    
    # Print mode
    st.subheader("🖨️ Print Mode")
    if st.button("📄 Print-Friendly View", use_container_width=True):
        st.markdown("""
        <script>
        // Hide sidebar and expand content for printing
        window.print();
        </script>
        """, unsafe_allow_html=True)
    st.caption("Tip: Use Ctrl+P at 75% zoom for best results")
    
    st.divider()
    
    # Data Upload
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Energy Data (CSV or HDF5)", type=["csv", "h5"])
    
    # Load data
    data = None
    if uploaded_file:
        file_type = "csv" if uploaded_file.name.endswith('.csv') else "h5"
        data = load_data(uploaded_file, file_type)
        data = standardize_columns(data)
        st.success(f"✅ Loaded {len(data)} records")
    else:
        try:
            data = load_data('synthetic_data_center_data.csv', file_type="csv")
            data = standardize_columns(data)
            st.info("📊 Using 2026 dataset")
        except:
            st.warning("⚠️ No data loaded")
    
    # Global Parameters
    st.subheader("🎛️ Parameters")
    energy_price_per_kwh = st.slider(
        "Energy Price ($/kWh)", 
        min_value=0.05, max_value=1.00, value=0.12, step=0.01
    )
    renewable_threshold = st.slider(
        "Renewable Threshold (%)", 
        min_value=0, max_value=100, value=50, step=5
    ) / 100
    emission_factor_non_renewable = st.slider(
        "Non-Renewable Emissions (kg CO₂/kWh)", 
        min_value=0.1, max_value=1.0, value=0.45, step=0.05
    )
    emission_factor_renewable = st.slider(
        "Renewable Emissions (kg CO₂/kWh)", 
        min_value=0.0, max_value=0.5, value=0.02, step=0.01
    )
    
    # Data Center Type
    st.subheader("🏢 Data Center Type")
    dc_type = st.selectbox("Select Type", ["Hyperscale", "Enterprise", "Small/Medium"])
    pue_defaults = {"Hyperscale": 1.2, "Enterprise": 1.5, "Small/Medium": 1.8}
    pue_target = st.number_input("Target PUE", min_value=1.0, max_value=3.0, value=pue_defaults[dc_type], step=0.1)


# ============================================================
# DEMO PRESENTATION MODE CONTROLLER
# ============================================================

if st.session_state.demo_running:
    # Run all demo steps at once with a visible spinner
    with st.spinner("🎬 **Running Demo** — Loading equipment, running optimization, generating forecasts..."):
        if data is not None:
            # Step 1: Run optimization
            results = run_simulation(data, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable)
            st.session_state.optimization_results = results
            
            # Step 2: Generate forecasts
            columns_to_forecast = [
                'Renewable Availability (%)',
                'Workload Energy Consumption (kWh)',
                'Energy Price ($/kWh)'
            ]
            forecasts = forecast_prophet(data, columns_to_forecast, periods=90*24)
            st.session_state.forecasts = forecasts
            st.session_state.forecast_data = pd.DataFrame({
                'Timestamp': forecasts['Workload Energy Consumption (kWh)']['ds'],
                'Workload Energy Consumption (kWh)': forecasts['Workload Energy Consumption (kWh)']['yhat'],
                'Renewable Availability (%)': forecasts['Renewable Availability (%)']['yhat'],
                'Energy Price ($/kWh)': forecasts['Energy Price ($/kWh)']['yhat']
            })
            st.session_state.forecast_data.set_index('Timestamp', inplace=True)
    
    st.session_state.demo_running = False
    st.session_state.demo_active = True
    st.rerun()

# ============================================================
# MAIN CONTENT - TAB NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🔮 Forecasting", 
    "⚙️ Equipment & Optimization",
    "💡 Recommendations",
    "🛡️ DCPLOS"
])


# ============================================================
# TAB 1: DASHBOARD
# ============================================================

with tab1:
    st.header("📊 Data Center Performance Dashboard")
    
    if data is not None:
        total_energy = data['Workload Energy Consumption (kWh)'].sum()
        avg_renewable = data['Renewable Availability (%)'].mean()
        total_cost = (data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum()
        equipment_power_kw = calculate_equipment_power(st.session_state.equipment_inventory)
        cooling_power_kw = equipment_power_kw * 0.45
        
        if equipment_power_kw > 0:
            actual_pue = calculate_pue(equipment_power_kw, cooling_power_kw)
        else:
            actual_pue = pue_target
        
        total_water = calculate_water_consumption(total_energy, actual_pue)
        
        # --- #1: SUMMARY BANNER (shows after optimization is run) ---
        if st.session_state.optimization_results:
            r = st.session_state.optimization_results
            savings_pct = r['energy_savings'] / r['baseline_energy'] * 100
            cost_pct = r['cost_savings'] / r['baseline_cost'] * 100
            co2_pct = r['emissions_savings'] / r['baseline_emissions'] * 100
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%); 
                        padding: 20px 30px; border-radius: 10px; margin-bottom: 20px; color: white;">
                <h3 style="margin: 0 0 12px 0; color: white; font-size: 18px;">✅ OPTENA Optimization Results — Annual Projected Savings</h3>
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <div style="font-size: 28px; font-weight: bold;">{savings_pct:.1f}%</div>
                        <div style="font-size: 13px; opacity: 0.9;">Energy Reduction</div>
                        <div style="font-size: 14px; margin-top: 4px;">{r['energy_savings']:,.0f} kWh/yr</div>
                    </div>
                    <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 20px;">
                        <div style="font-size: 28px; font-weight: bold;">${r['cost_savings']:,.0f}</div>
                        <div style="font-size: 13px; opacity: 0.9;">Cost Savings</div>
                        <div style="font-size: 14px; margin-top: 4px;">{cost_pct:.1f}% reduction</div>
                    </div>
                    <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 20px;">
                        <div style="font-size: 28px; font-weight: bold;">{r['emissions_savings']:,.0f} kg</div>
                        <div style="font-size: 13px; opacity: 0.9;">CO₂ Eliminated</div>
                        <div style="font-size: 14px; margin-top: 4px;">{co2_pct:.1f}% cleaner</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- #4: COLOR-CODED KPI METRICS ---
        st.subheader("🎯 Key Performance Indicators")
        
        # Determine PUE status color
        if actual_pue <= pue_target:
            pue_color = "#4CAF50"  # green
            pue_status = "✅ On Target"
        elif actual_pue <= pue_target * 1.1:
            pue_color = "#FF9800"  # amber
            pue_status = "⚠️ Near Target"
        else:
            pue_color = "#F44336"  # red
            pue_status = "🔴 Above Target"
        
        # Renewable status
        if avg_renewable >= 50:
            renew_color = "#4CAF50"
            renew_status = "✅ Good"
        elif avg_renewable >= 30:
            renew_color = "#FF9800"
            renew_status = "⚠️ Moderate"
        else:
            renew_color = "#F44336"
            renew_status = "🔴 Low"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Energy Consumption", f"{total_energy:,.0f} kWh",
                      help="Total IT equipment energy consumption for 2026")
        with col2:
            pue_delta = actual_pue - pue_target
            st.metric("Power Usage Effectiveness (PUE)", f"{actual_pue}",
                      delta=f"{pue_delta:+.2f} vs target", delta_color="inverse",
                      help="PUE = Total Facility Energy / IT Equipment Energy. Lower is better.")
            st.markdown(f"<span style='color:{pue_color}; font-size: 12px; font-weight: bold;'>{pue_status}</span>", unsafe_allow_html=True)
        with col3:
            st.metric("Total Energy Cost", f"${total_cost:,.2f}",
                      help="Total cost based on energy consumption and price")
        with col4:
            st.metric("Avg Renewable Availability", f"{avg_renewable:.1f}%",
                      help="Average renewable energy availability across 2026")
            st.markdown(f"<span style='color:{renew_color}; font-size: 12px; font-weight: bold;'>{renew_status}</span>", unsafe_allow_html=True)
        
        # KPI Metrics Row 2
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("IT Equipment Power", f"{equipment_power_kw:.1f} kW",
                      help="Current IT equipment power draw from inventory")
        with col6:
            st.metric("Estimated Cooling Power", f"{cooling_power_kw:.1f} kW",
                      help="Estimated cooling infrastructure power (45% of IT load)")
        with col7:
            st.metric("Water Consumption", f"{total_water:,.0f} L",
                      help="Estimated annual water usage for cooling")
        with col8:
            if st.session_state.optimization_results:
                savings_pct = (st.session_state.optimization_results['energy_savings'] / 
                              st.session_state.optimization_results['baseline_energy'] * 100)
                st.metric("Optimization Savings", f"{savings_pct:.1f}%",
                          delta=f"${st.session_state.optimization_results['cost_savings']:,.2f} saved")
                st.markdown("<span style='color:#4CAF50; font-size: 12px; font-weight: bold;'>✅ Active</span>", unsafe_allow_html=True)
            else:
                st.metric("Optimization Savings", "—", help="Run optimization to see savings")
        
        # --- ENERGY CONSUMPTION CHART WITH ANNOTATIONS ---
        st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
        st.subheader("📈 Energy Consumption Over Time (2026)")
        
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=data.index, y=data['Workload Energy Consumption (kWh)'],
            mode='lines', name='Energy Consumption',
            line=dict(color='#4CAF50', width=1.5)
        ))
        
        # Add 7-day rolling average
        rolling_avg = data['Workload Energy Consumption (kWh)'].rolling(168).mean()
        fig_energy.add_trace(go.Scatter(
            x=data.index, y=rolling_avg,
            mode='lines', name='7-Day Average',
            line=dict(color='#FF9800', width=2.5, dash='dot')
        ))
        
        # Annotate peak and trough on rolling average
        if rolling_avg.dropna().shape[0] > 0:
            peak_idx = rolling_avg.idxmax()
            peak_val = rolling_avg.max()
            trough_idx = rolling_avg.idxmin()
            trough_val = rolling_avg.min()
        
        fig_energy.update_layout(
            title=dict(text="Energy Consumption Timeline — 2026 (Hourly with 7-Day Moving Average)", font=dict(size=16)),
            xaxis_title="Date", yaxis_title="Energy (kWh)",
            hovermode='x unified', height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=13))
        )
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # Annotations OUTSIDE the chart
        if rolling_avg.dropna().shape[0] > 0:
            ann_col1, ann_col2 = st.columns(2)
            with ann_col1:
                st.markdown(f"""
                <div style="background: #FFEBEE; border-left: 4px solid #F44336; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                    <strong style="color: #F44336; font-size: 15px;">▲ SUMMER PEAK — {peak_val:.0f} kWh (7-day avg)</strong><br>
                    <span style="font-size: 14px;">Cooling load drives demand up during Jul-Aug. <strong>Action:</strong> Shift batch jobs and non-critical workloads to nighttime hours to reduce peak demand charges.</span>
                </div>
                """, unsafe_allow_html=True)
            with ann_col2:
                st.markdown(f"""
                <div style="background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                    <strong style="color: #4CAF50; font-size: 15px;">▼ WINTER LOW — {trough_val:.0f} kWh (7-day avg)</strong><br>
                    <span style="font-size: 14px;">Reduced cooling overhead in winter months. <strong>Action:</strong> Schedule maintenance windows, firmware updates, and hardware refreshes during this period.</span>
                </div>
                """, unsafe_allow_html=True)
        
        # --- RENEWABLE & PRICING CHARTS (full width, stacked) ---
        st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
        st.subheader("🌱 Renewable Energy Availability")
        fig_renew = go.Figure()
        fig_renew.add_trace(go.Scatter(
            x=data.index, y=data['Renewable Availability (%)'],
            fill='tozeroy', name='Renewable %',
            line=dict(color='#8BC34A', width=1)
        ))
        
        # Add threshold line
        fig_renew.add_hline(
            y=renewable_threshold * 100, line_dash="dash", line_color="#F44336",
            annotation_text=f"Threshold: {renewable_threshold*100:.0f}%",
            annotation_position="top right"
        )
        
        # Annotate
        daily_renewable = data['Renewable Availability (%)'].resample('D').mean()
        best_day = daily_renewable.idxmax()
        best_val = daily_renewable.max()
        worst_day = daily_renewable.idxmin()
        worst_val = daily_renewable.min()
        
        fig_renew.update_layout(
            title=dict(text="Renewable Availability — Below red threshold = grid dependency", font=dict(size=15)),
            xaxis_title="Date", yaxis_title="Availability (%)", height=450
        )
        st.plotly_chart(fig_renew, use_container_width=True)
        
        st.markdown(f"""
        <div style="background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
            <strong style="color: #4CAF50;">▲ Best Day: {best_val:.0f}% renewable ({best_day.strftime('%b %d')})</strong> — Max solar + wind alignment. Schedule heavy compute here.&nbsp;&nbsp;&nbsp;
            <strong style="color: #F44336;">▼ Worst Day: {worst_val:.0f}% renewable ({worst_day.strftime('%b %d')})</strong> — High grid dependency, higher carbon footprint.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
        st.subheader("💰 Energy Pricing")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=data.index, y=data['Energy Price ($/kWh)'],
            mode='lines', name='Energy Price',
            line=dict(color='#FF9800', width=1)
        ))
        
        # Add rolling average
        price_avg = data['Energy Price ($/kWh)'].rolling(168).mean()
        fig_price.add_trace(go.Scatter(
            x=data.index, y=price_avg,
            mode='lines', name='7-Day Avg Price',
            line=dict(color='#9C27B0', width=2.5, dash='dot')
        ))
        
        # Annotate price spikes
        if price_avg.dropna().shape[0] > 0:
            price_peak_idx = price_avg.idxmax()
            price_peak_val = price_avg.max()
            price_low_idx = price_avg.idxmin()
            price_low_val = price_avg.min()
        
        fig_price.update_layout(
            title=dict(text="Energy Price Timeline — Spikes = cost optimization opportunity", font=dict(size=15)),
            xaxis_title="Date", yaxis_title="Price ($/kWh)", height=450
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        if price_avg.dropna().shape[0] > 0:
            st.markdown(f"""
            <div style="background: #FFF3E0; border-left: 4px solid #FF9800; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                <strong style="color: #F44336;">▲ Price Peak: ${price_peak_val:.3f}/kWh ({price_peak_idx.strftime('%b %d')})</strong> — Summer demand surcharge. Defer non-critical workloads.&nbsp;&nbsp;&nbsp;
                <strong style="color: #4CAF50;">▼ Price Low: ${price_low_val:.3f}/kWh ({price_low_idx.strftime('%b %d')})</strong> — Off-peak window. Run energy-intensive operations here.
            </div>
            """, unsafe_allow_html=True)
        
        # --- EQUIPMENT PIE CHART ---
        st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
        if not st.session_state.equipment_inventory.empty:
            st.subheader("⚙️ Equipment Power Distribution")
            equipment_summary = st.session_state.equipment_inventory.copy()
            equipment_summary['Total Power (kW)'] = (
                equipment_summary['Power per Unit (W)'] * 
                equipment_summary['Count'] * 
                equipment_summary['Utilization (%)'] / 100 / 1000
            )
            category_power = equipment_summary.groupby('Category')['Total Power (kW)'].sum().reset_index()
            total_power_kw = category_power['Total Power (kW)'].sum()
            category_power['% of Total'] = (category_power['Total Power (kW)'] / total_power_kw * 100).round(1)
            category_power = category_power.sort_values('Total Power (kW)', ascending=False)
            
            fig_pie = px.pie(
                category_power, values='Total Power (kW)', names='Category',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.35  # donut style — easier to read
            )
            fig_pie.update_traces(
                textposition='outside', 
                textinfo='label+percent',
                textfont_size=14,
                pull=[0.05] * len(category_power)  # slight pull for visibility
            )
            fig_pie.update_layout(
                title=dict(text="Power Consumption by Equipment Category", font=dict(size=16)),
                height=420,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(font=dict(size=13), orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Summary table below the chart
            st.markdown("**Power Breakdown by Category:**")
            table_df = category_power[['Category', 'Total Power (kW)', '% of Total']].copy()
            table_df['Total Power (kW)'] = table_df['Total Power (kW)'].apply(lambda x: f"{x:.1f}")
            table_df['% of Total'] = table_df['% of Total'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            
            # Annotation below
            top_category = category_power.iloc[0]
            st.markdown(f"""
            <div style="background: #E3F2FD; border-left: 4px solid #1976D2; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                <strong style="color: #1976D2; font-size: 14px;">📊 Largest Consumer: {top_category['Category']} — {top_category['Total Power (kW)']:.1f} kW ({top_category['% of Total']:.1f}% of total)</strong><br>
                <span style="font-size: 13px;">Focus optimization efforts here first for maximum impact. Consider consolidation, virtualization, or hardware refresh for this category. Total facility IT power: <strong>{total_power_kw:.1f} kW</strong>.</span>
            </div>
            """, unsafe_allow_html=True)
        
        # --- ALERTS ---
        st.subheader("🚨 Alerts & Notifications")
        if actual_pue > pue_target * 1.1:
            st.warning(f"⚠️ **PUE Alert**: Current PUE ({actual_pue}) is {((actual_pue/pue_target - 1) * 100):.1f}% above target ({pue_target})")
        
        low_renewable_hours = data[data['Renewable Availability (%)'] < renewable_threshold * 100]
        if len(low_renewable_hours) > len(data) * 0.3:
            st.warning(f"⚠️ **Renewable Alert**: {len(low_renewable_hours)} hours ({len(low_renewable_hours)/len(data)*100:.1f}%) below {renewable_threshold*100:.0f}% threshold — high grid dependency")
        
        high_cost = data[data['Energy Price ($/kWh)'] > data['Energy Price ($/kWh)'].quantile(0.75)]
        if len(high_cost) > 0:
            avg_high = data['Energy Price ($/kWh)'].quantile(0.75)
            st.info(f"💡 **Cost Optimization**: {len(high_cost)} hours above ${avg_high:.3f}/kWh. Shift non-critical workloads to save ~{len(high_cost) * 0.1 * energy_price_per_kwh:.0f} kWh equivalent.")
    else:
        st.warning("📊 Please upload data to view dashboard metrics")


# ============================================================
# TAB 2: FORECASTING
# ============================================================

with tab2:
    st.header("🔮 Energy & Renewable Forecasting")
    
    if data is not None:
        st.markdown("Generate future predictions using Facebook Prophet AI model. Forecasts forward from the latest data point.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            forecast_days = st.slider("Forecast Period (days)", min_value=7, max_value=365, value=90, step=7)
        with col2:
            generate_forecast = st.button("🔮 Generate Forecasts", type="primary")
        
        if generate_forecast:
            with st.spinner("Generating AI forecasts... This may take 1-2 minutes."):
                periods = forecast_days * 24
                columns_to_forecast = [
                    'Renewable Availability (%)',
                    'Workload Energy Consumption (kWh)',
                    'Energy Price ($/kWh)'
                ]
                forecasts = forecast_prophet(data, columns_to_forecast, periods=periods)
                st.session_state.forecasts = forecasts
                st.session_state.forecast_data = pd.DataFrame({
                    'Timestamp': forecasts['Workload Energy Consumption (kWh)']['ds'],
                    'Workload Energy Consumption (kWh)': forecasts['Workload Energy Consumption (kWh)']['yhat'],
                    'Renewable Availability (%)': forecasts['Renewable Availability (%)']['yhat'],
                    'Energy Price ($/kWh)': forecasts['Energy Price ($/kWh)']['yhat']
                })
                st.session_state.forecast_data.set_index('Timestamp', inplace=True)
                st.success(f"✅ Forecasts generated for next {forecast_days} days")
        
        # Display forecasts
        if 'forecasts' in st.session_state:
            st.subheader("📊 Forecast Results")
            
            forecast_colors = {
                'Renewable Availability (%)': ('#8BC34A', 'Renewable availability will follow seasonal solar/wind patterns'),
                'Workload Energy Consumption (kWh)': ('#2196F3', 'Energy demand expected to maintain current patterns with seasonal variation'),
                'Energy Price ($/kWh)': ('#FF9800', 'Pricing forecast helps lock in favorable contract rates')
            }
            
            for column, forecast in st.session_state.forecasts.items():
                color, insight = forecast_colors[column]
                st.markdown(f"### {column}")
                st.caption(f"💡 *Insight: {insight}*")
                
                fig = go.Figure()
                
                # Historical
                historical = data.reset_index()
                fig.add_trace(go.Scatter(
                    x=historical['Timestamp'], y=historical[column],
                    mode='lines', name='Historical (2026)',
                    line=dict(color=color, width=1.5)
                ))
                
                # Forecast
                future_mask = forecast['ds'] > data.index.max()
                forecast_future = forecast[future_mask]
                
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'], y=forecast_future['yhat'],
                    mode='lines', name=f'AI Forecast ({forecast_days} days ahead)',
                    line=dict(color='#E91E63', width=2.5, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'], y=forecast_future['yhat_upper'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'], y=forecast_future['yhat_lower'],
                    mode='lines', fill='tonexty', line=dict(width=0),
                    name='95% Confidence Interval',
                    fillcolor='rgba(233,30,99,0.15)'
                ))
                
                # Add annotation on forecast
                if len(forecast_future) > 0:
                    mid_idx = len(forecast_future) // 2
                    mid_row = forecast_future.iloc[mid_idx]
                    band_width = mid_row['yhat_upper'] - mid_row['yhat_lower']
                
                # Add vertical line at forecast start (annotation-free to avoid Plotly bug)
                forecast_start = str(data.index.max())
                fig.add_shape(
                    type="line", x0=forecast_start, x1=forecast_start,
                    y0=0, y1=1, yref="paper",
                    line=dict(color="gray", width=2, dash="dash")
                )
                # Add text annotation near the line
                fig.add_annotation(
                    x=forecast_start, y=1, yref="paper",
                    text="← Historical | Forecast →",
                    showarrow=False, font=dict(size=12, color="gray"),
                    bgcolor="rgba(255,255,255,0.8)", yanchor="bottom"
                )
                
                fig.update_layout(
                    xaxis_title="Date", yaxis_title=column,
                    hovermode='x unified', height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=13))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Annotation OUTSIDE the chart
                if len(forecast_future) > 0:
                    st.markdown(f"""
                    <div style="background: #F3E5F5; border-left: 4px solid #9C27B0; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                        <strong style="color: #9C27B0; font-size: 14px;">📊 Forecast Insight — {column}</strong><br>
                        <span style="font-size: 13px;">Confidence band width: ±{band_width/2:.2f} — Wider band = more uncertainty in prediction. Use narrow-band periods for operational decisions, wide-band periods for risk/contingency planning. The dashed line marks where historical data ends and AI prediction begins.</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.info("💡 Use forecasted data in the Equipment & Optimization tab to plan future energy strategies.")
        else:
            st.info("👆 Click 'Generate Forecasts' to see AI predictions")
    else:
        st.warning("📊 Please upload data to generate forecasts")


# ============================================================
# TAB 3: EQUIPMENT & OPTIMIZATION
# ============================================================

with tab3:
    st.header("⚙️ IT Equipment Inventory & Energy Optimization")
    
    # Equipment Inventory Section
    st.subheader("🖥️ Equipment Inventory")
    
    with st.expander("➕ Add Equipment", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            eq_type = st.text_input("Equipment Name", placeholder="e.g., Dell PowerEdge R750")
            eq_category = st.selectbox("Category", ["Servers", "Storage", "Network", "Power/UPS", "Cooling", "Other"])
            eq_count = st.number_input("Count", min_value=1, value=1, step=1)
        with col2:
            eq_power = st.number_input("Power per Unit (W)", min_value=1, value=300, step=10)
            eq_max_power = st.number_input("Max Power (W)", min_value=1, value=500, step=10)
            eq_utilization = st.slider("Utilization (%)", min_value=0, max_value=100, value=50, step=5)
        with col3:
            eq_performance = st.selectbox("Performance Level", ["High", "Mid", "Low"])
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add to Inventory"):
                new_equipment = pd.DataFrame([{
                    'Equipment Type': eq_type, 'Category': eq_category,
                    'Count': eq_count, 'Power per Unit (W)': eq_power,
                    'Max Power (W)': eq_max_power, 'Utilization (%)': eq_utilization,
                    'Performance Level': eq_performance
                }])
                st.session_state.equipment_inventory = pd.concat(
                    [st.session_state.equipment_inventory, new_equipment], ignore_index=True
                )
                st.success(f"✅ Added {eq_type}")
                st.rerun()
    
    # Display inventory
    if not st.session_state.equipment_inventory.empty:
        st.markdown("### Current Equipment Inventory")
        display_df = st.session_state.equipment_inventory.copy()
        display_df['Efficiency Grade'] = display_df.apply(
            lambda row: get_equipment_efficiency_grade(row['Utilization (%)'], row['Performance Level']), axis=1
        )
        display_df['Total Power (kW)'] = (
            display_df['Power per Unit (W)'] * display_df['Count'] * display_df['Utilization (%)'] / 100 / 1000
        )
        st.dataframe(display_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Equipment", f"{display_df['Count'].sum()} units")
        with col2:
            st.metric("Total IT Power", f"{display_df['Total Power (kW)'].sum():.1f} kW")
        with col3:
            avg_util = (display_df['Utilization (%)'] * display_df['Count']).sum() / display_df['Count'].sum()
            st.metric("Avg Utilization", f"{avg_util:.1f}%")
        with col4:
            grade_counts = display_df['Efficiency Grade'].value_counts()
            a_b_count = grade_counts.get('A', 0) + grade_counts.get('B', 0)
            st.metric("A/B Grade Equipment", f"{a_b_count}/{len(display_df)}")
        
        # Efficiency analysis
        st.markdown("### 🎯 Equipment Efficiency Analysis")
        
        low_util = display_df[display_df['Utilization (%)'] < 30]
        if not low_util.empty:
            st.warning(f"⚠️ **Low Utilization**: {len(low_util)} equipment types below 30% — consolidation candidates")
            st.dataframe(low_util[['Equipment Type', 'Category', 'Utilization (%)', 'Efficiency Grade', 'Total Power (kW)']])
        
        poor_grade = display_df[display_df['Efficiency Grade'].isin(['D', 'F'])]
        if not poor_grade.empty:
            st.error(f"🔴 **Poor Efficiency**: {len(poor_grade)} items graded D/F — immediate action recommended")
        
        if st.button("🗑️ Clear All Equipment"):
            st.session_state.equipment_inventory = pd.DataFrame(columns=[
                'Equipment Type', 'Category', 'Count', 'Power per Unit (W)', 
                'Max Power (W)', 'Utilization (%)', 'Performance Level'
            ])
            st.rerun()
    else:
        st.info("📝 No equipment in inventory. Add equipment or run the demo.")
    
    st.markdown("---")
    
    # Energy Optimization Section
    st.subheader("⚡ Energy Optimization")
    
    if data is not None:
        use_forecast = st.radio(
            "Data Source", ["Historical Data (2026)", "Forecasted Data (2026)"], index=0
        )
        
        if use_forecast == "Forecasted Data (2026)" and 'forecast_data' in st.session_state:
            data_for_optimization = st.session_state.forecast_data
            st.info("🔮 Using AI-forecasted data for forward-looking optimization")
        else:
            data_for_optimization = data
        
        run_opt = st.button("⚡ Run Energy Optimization", type="primary")
        
        if run_opt:
            with st.spinner("Optimizing..."):
                results = run_simulation(data_for_optimization, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable)
                st.session_state.optimization_results = results
                st.success("✅ Optimization complete!")
        
        if st.session_state.optimization_results:
            results = st.session_state.optimization_results
            
            st.markdown("### 📊 Optimization Results — Baseline vs Optimized")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Baseline (Current State)")
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Energy", f"{results['baseline_energy']:,.0f} kWh")
                with m2: st.metric("Cost", f"${results['baseline_cost']:,.2f}")
                with m3: st.metric("CO₂", f"{results['baseline_emissions']:,.0f} kg")
            
            with col2:
                st.markdown("#### 🟢 Optimized (With Optena)")
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Energy", f"{results['optimized_energy']:,.0f} kWh", delta=f"-{results['energy_savings']:,.0f}")
                with m2: st.metric("Cost", f"${results['optimized_cost']:,.2f}", delta=f"-${results['cost_savings']:,.2f}")
                with m3: st.metric("CO₂", f"{results['optimized_emissions']:,.0f} kg", delta=f"-{results['emissions_savings']:,.0f}")
            
            # Savings summary
            st.markdown("### 💰 Annual Savings Summary")
            s1, s2, s3 = st.columns(3)
            with s1:
                pct = results['energy_savings'] / results['baseline_energy'] * 100
                st.metric("Energy Reduction", f"{pct:.1f}%", f"{results['energy_savings']:,.0f} kWh/year")
            with s2:
                cpct = results['cost_savings'] / results['baseline_cost'] * 100
                st.metric("Cost Reduction", f"{cpct:.1f}%", f"${results['cost_savings']:,.2f}/year")
            with s3:
                epct = results['emissions_savings'] / results['baseline_emissions'] * 100
                st.metric("CO₂ Reduction", f"{epct:.1f}%", f"{results['emissions_savings']:,.0f} kg/year")
            
            # Before/After chart with annotations
            st.markdown("### 📈 Energy Consumption: Before vs After Optimization")
            
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=results['data'].index, y=results['data']['Workload Energy Consumption (kWh)'],
                mode='lines', name='Baseline (Current)',
                line=dict(color='#F44336', width=1.5)
            ))
            fig_opt.add_trace(go.Scatter(
                x=results['data'].index, y=results['data']['Optimized Energy'],
                mode='lines', name='Optimized (With Optena)',
                line=dict(color='#4CAF50', width=1.5)
            ))
            
            # Annotate the gap
            mid_point = results['data'].index[len(results['data'])//2]
            baseline_mid = results['data']['Workload Energy Consumption (kWh)'].iloc[len(results['data'])//2]
            optimized_mid = results['data']['Optimized Energy'].iloc[len(results['data'])//2]
            
            fig_opt.update_layout(
                title=dict(text="Baseline vs Optimized — Gap represents energy savings from intelligent scheduling", font=dict(size=15)),
                xaxis_title="Date", yaxis_title="Energy (kWh)",
                hovermode='x unified', height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=13))
            )
            st.plotly_chart(fig_opt, use_container_width=True)
            
            # Annotation OUTSIDE the chart
            st.markdown(f"""
            <div style="background: #E3F2FD; border-left: 4px solid #1976D2; padding: 12px; margin-bottom: 16px; border-radius: 4px;">
                <strong style="color: #1976D2; font-size: 15px;">↕ GAP = Energy Saved — {pct:.1f}% reduction through renewable-aware scheduling</strong><br>
                <span style="font-size: 14px;">Red line = current baseline consumption. Green line = optimized consumption with Optena.<br>
                The visible gap between lines represents <strong>{results['energy_savings']:,.0f} kWh</strong> of annual savings (${results['cost_savings']:,.2f} cost reduction).<br>
                Savings are largest during periods of low renewable availability, when the system curtails non-critical workloads by 10%.</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("📊 Please upload data to run optimization")


# ============================================================
# TAB 4: RECOMMENDATIONS
# ============================================================

with tab4:
    st.header("💡 Recommendations & Best Practices")
    
    if data is not None:
        st.markdown("Actionable insights based on LBNL/DOE research and your facility's data patterns.")
        
        # Time-based recommendations
        st.subheader("⏰ Time-Based Optimization Windows")
        
        data_copy = data.copy()
        data_copy['Hour'] = data_copy.index.hour
        data_copy['Day'] = data_copy.index.day_name()
        
        hourly_availability = data_copy.groupby('Hour')['Renewable Availability (%)'].mean()
        hourly_prices = data_copy.groupby('Hour')['Energy Price ($/kWh)'].mean()
        daily_prices = data_copy.groupby('Day')['Energy Price ($/kWh)'].mean()
        
        optimal_hour_renewable = hourly_availability.idxmax()
        optimal_hour_price = hourly_prices.idxmin()
        worst_hour_price = hourly_prices.idxmax()
        optimal_day = daily_prices.idxmin()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"""
            **🌞 Best Window for Renewable Energy**
            - **Time**: {optimal_hour_renewable}:00 - {optimal_hour_renewable+1}:00
            - **Avg Renewable**: {hourly_availability[optimal_hour_renewable]:.1f}%
            - **Action**: Schedule batch jobs, backups, and ML training during this window
            - **Impact**: Reduces carbon footprint by maximizing green energy use
            """)
        with col2:
            st.success(f"""
            **💰 Cheapest Energy Window**
            - **Time**: {optimal_hour_price}:00 - {optimal_hour_price+1}:00
            - **Avg Price**: ${hourly_prices[optimal_hour_price]:.4f}/kWh
            - **Action**: Run energy-intensive operations (data migrations, builds)
            - **vs Peak**: {((hourly_prices[worst_hour_price] - hourly_prices[optimal_hour_price]) / hourly_prices[worst_hour_price] * 100):.0f}% cheaper than peak hour ({worst_hour_price}:00)
            """)
        
        st.info(f"📅 **Best Day**: {optimal_day} (avg ${daily_prices[optimal_day]:.4f}/kWh) — Plan weekly maintenance here")
        
        # Hourly pattern chart
        st.markdown("### 📊 Hourly Patterns — Decision Support")
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_availability.index, y=hourly_availability.values,
            name='Renewable %', marker_color='#8BC34A', opacity=0.7
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly_prices.index, y=hourly_prices.values * 500,  # Scale for visibility
            mode='lines+markers', name='Price (scaled)',
            line=dict(color='#FF9800', width=3), yaxis='y2'
        ))
        
        # Annotate optimal window
        fig_hourly.add_vrect(
            x0=optimal_hour_renewable - 0.5, x1=optimal_hour_renewable + 0.5,
            fillcolor="rgba(76,175,80,0.2)", line_width=0,
            annotation_text="OPTIMAL", annotation_position="top"
        )
        fig_hourly.add_vrect(
            x0=worst_hour_price - 0.5, x1=worst_hour_price + 0.5,
            fillcolor="rgba(244,67,54,0.2)", line_width=0,
            annotation_text="AVOID", annotation_position="top"
        )
        
        fig_hourly.update_layout(
            title="Hourly Renewable Availability & Energy Price — Green zone = schedule here, Red zone = avoid",
            xaxis_title="Hour of Day", yaxis_title="Renewable Availability (%)",
            yaxis2=dict(title="Energy Price ($/kWh)", overlaying='y', side='right',
                       tickvals=[v*500 for v in [0.05, 0.10, 0.15, 0.20]],
                       ticktext=['$0.05', '$0.10', '$0.15', '$0.20']),
            height=550, barmode='overlay'
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Equipment recommendations
        if not st.session_state.equipment_inventory.empty:
            st.subheader("⚙️ Equipment Optimization Opportunities")
            inventory = st.session_state.equipment_inventory.copy()
            
            server_inventory = inventory[inventory['Category'] == 'Servers']
            if not server_inventory.empty:
                avg_server_util = server_inventory['Utilization (%)'].mean()
                if avg_server_util < 60:
                    potential_savings_kw = server_inventory.apply(
                        lambda r: r['Power per Unit (W)'] * r['Count'] * (1 - r['Utilization (%)']/100) / 1000, axis=1
                    ).sum()
                    st.warning(f"""
                    **🔄 Server Consolidation Opportunity**
                    - Current avg server utilization: {avg_server_util:.1f}%
                    - Wasted capacity power: {potential_savings_kw:.1f} kW
                    - Annual cost of waste: ${potential_savings_kw * 8760 * energy_price_per_kwh:,.0f}
                    - **Recommendation**: Virtualize and consolidate to reach 70%+ utilization
                    """)
        
        # PUE Improvement
        st.subheader("🏢 Infrastructure Efficiency")
        if actual_pue > 1.4:
            improvement_potential = (actual_pue - 1.2) * total_energy
            st.warning(f"""
            **🌡️ PUE Improvement Path**
            - Current PUE: {actual_pue} | Industry best: 1.1-1.2
            - Excess energy overhead: {improvement_potential:,.0f} kWh/year
            - Cost of inefficiency: ${improvement_potential * energy_price_per_kwh:,.0f}/year
            
            **Priority Actions (by ROI):**
            1. Hot/cold aisle containment → PUE improvement: 0.1-0.3
            2. Raise cooling setpoint to 80°F (ASHRAE A1) → immediate 5-10% cooling savings
            3. Variable speed drives on CRAC units → 20-30% fan energy reduction
            4. Free cooling economizer hours → climate-dependent, 2000-5000 hrs/year
            """)
        else:
            st.success(f"✅ **Excellent PUE**: {actual_pue} — Operating at industry best-practice levels")
        
        # Water conservation
        st.subheader("💧 Water Conservation")
        st.info(f"""
        **Current Water Usage**: {total_water:,.0f} liters/year
        
        **Reduction Strategies:**
        1. Air-side economizers: Reduce water cooling by 60-80% in suitable climates
        2. Closed-loop dry coolers: Eliminate evaporative water loss
        3. Rainwater harvesting: Offset 15-25% of makeup water needs
        4. Water-side economizer optimization: Raise condenser water temp setpoint
        """)
        
        # Best Practices
        st.subheader("📋 LBNL Best Practices Checklist")
        best_practices = {
            "IT Equipment": [
                "Virtualize servers to achieve >60% utilization (current industry avg: 12-18%)",
                "Enable CPU power management (C-states, P-states) in BIOS",
                "Decommission zombie/comatose servers (est. 20-30% of installed base)",
                "Implement workload-aware scheduling aligned with renewable availability",
                "Right-size VMs and containers to actual resource consumption"
            ],
            "Cooling Systems": [
                "Implement hot/cold aisle containment (ROI: 6-18 months)",
                "Raise supply air temperature to 80°F per ASHRAE TC 9.9 A1 envelope",
                "Deploy variable speed drives on all cooling equipment",
                "Install blanking panels and seal cable openings",
                "Use CFD modeling to identify and eliminate hot spots"
            ],
            "Power Distribution": [
                "Deploy high-efficiency UPS (>96% at partial load)",
                "Right-size UPS to actual load (avoid oversizing penalty)",
                "Implement rack-level power monitoring (per-outlet metering)",
                "Minimize power conversion steps (AC→DC→AC losses)",
                "Consider 240V distribution to reduce I²R losses"
            ],
            "Operations & Management": [
                "Establish energy baseline and track PUE monthly",
                "Implement DCIM for real-time capacity and thermal monitoring",
                "Set department-level energy budgets with accountability",
                "Conduct annual energy audits (ASHRAE Level 2 minimum)",
                "Train operations staff on energy-efficient procedures"
            ]
        }
        
        for category, practices in best_practices.items():
            with st.expander(f"**{category}**", expanded=False):
                for i, practice in enumerate(practices, 1):
                    st.markdown(f"{i}. {practice}")
        
        # Export
        st.markdown("---")
        if st.button("📄 Export Recommendations Report"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = f"""
OPTENA - Data Center Energy Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
By PENTREST GLOBAL LLC
{'='*60}

FACILITY PERFORMANCE (2026 Data)
{'-'*40}
Total Energy Consumption: {total_energy:,.0f} kWh
Power Usage Effectiveness: {actual_pue}
Total Annual Cost: ${total_cost:,.2f}
Average Renewable Availability: {avg_renewable:.1f}%
Water Consumption: {total_water:,.0f} liters

OPTIMAL SCHEDULING WINDOWS
{'-'*40}
Best Hour for Renewables: {optimal_hour_renewable}:00 ({hourly_availability[optimal_hour_renewable]:.1f}%)
Cheapest Energy Hour: {optimal_hour_price}:00 (${hourly_prices[optimal_hour_price]:.4f}/kWh)
Best Day for Operations: {optimal_day}

OPTIMIZATION RESULTS
{'-'*40}
"""
            if st.session_state.optimization_results:
                r = st.session_state.optimization_results
                report += f"""Energy Savings: {r['energy_savings']:,.0f} kWh ({r['energy_savings']/r['baseline_energy']*100:.1f}%)
Cost Savings: ${r['cost_savings']:,.2f} ({r['cost_savings']/r['baseline_cost']*100:.1f}%)
CO₂ Reduction: {r['emissions_savings']:,.0f} kg ({r['emissions_savings']/r['baseline_emissions']*100:.1f}%)
"""
            report += f"""
EQUIPMENT INVENTORY
{'-'*40}
Total Equipment: {st.session_state.equipment_inventory['Count'].sum() if not st.session_state.equipment_inventory.empty else 0} units
Total IT Power: {calculate_equipment_power(st.session_state.equipment_inventory)} kW

{'='*60}
For detailed implementation guidance, refer to:
- LBNL Data Center Energy Efficiency Best Practices
- ASHRAE TC 9.9 Thermal Guidelines
- DOE Better Buildings Challenge
"""
            st.download_button(
                label="⬇️ Download Report", data=report,
                file_name=f"optena_report_{timestamp}.txt", mime="text/plain"
            )
    else:
        st.warning("📊 Please upload data to view recommendations")

# ============================================================
# TAB 5: DCPLOS — Cascade Failure & Risk-Aware Orchestration
# ============================================================

with tab5:
    st.header("🛡️ DCPLOS — Physical Layer Orchestration")
    st.markdown("**Data Center Physical Layer Orchestration System** — Cascade failure prediction, risk scoring, and risk-aware placement intelligence.")
    
    from dcplos_engine import create_demo_facility, ComponentType
    
    # Initialize DCPLOS engine in session state
    if 'dcplos_engine' not in st.session_state:
        st.session_state.dcplos_engine = None
    
    # Load demo facility button
    if st.button("🏗️ Load Demo Facility (40 racks, 58 components)", type="primary"):
        st.session_state.dcplos_engine = create_demo_facility()
        st.success("✅ Demo facility loaded: 8 rows × 5 racks, full power/cooling/network topology")
        st.rerun()
    
    if st.session_state.dcplos_engine:
        engine = st.session_state.dcplos_engine
        
        # Facility summary
        st.subheader("📋 Facility Overview")
        total_comps = len(engine.components)
        total_edges = engine.graph.number_of_edges()
        total_racks = sum(1 for c in engine.components.values() if c.component_type == ComponentType.RACK)
        total_load = sum(c.current_load_kw for c in engine.components.values() if c.component_type == ComponentType.RACK)
        
        ov1, ov2, ov3, ov4 = st.columns(4)
        with ov1:
            st.metric("Total Components", total_comps)
        with ov2:
            st.metric("Dependencies Modeled", total_edges)
        with ov3:
            st.metric("Racks", total_racks)
        with ov4:
            st.metric("Total IT Load", f"{total_load:.0f} kW")
        
        st.markdown("---")
        
        # === CASCADE SIMULATION ===
        st.subheader("💥 Cascade Failure Simulation")
        st.markdown("Select a component to simulate its failure and see the propagation chain.")
        
        # Component selector (exclude racks for cleaner UX)
        infrastructure_components = {cid: c for cid, c in engine.components.items() 
                                     if c.component_type != ComponentType.RACK}
        
        component_options = {f"{c.name} ({c.component_type.value}, {c.utilization*100:.0f}% loaded)": cid 
                           for cid, c in infrastructure_components.items()}
        
        sim_col1, sim_col2 = st.columns([2, 1])
        with sim_col1:
            selected_label = st.selectbox("Component to Fail", list(component_options.keys()))
        with sim_col2:
            degradation = st.slider("Degradation %", 25, 100, 100, 25)
        
        if st.button("⚡ Simulate Cascade", type="primary"):
            selected_id = component_options[selected_label]
            
            with st.spinner("Simulating cascade propagation..."):
                timeline, final_state = engine.simulate_cascade(selected_id, degradation)
            
            # Results
            st.markdown(f"### Cascade Results: {engine.components[selected_id].name} @ {degradation}% failure")
            
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Total Affected", final_state['total_affected'], 
                         help="Total components impacted by the cascade")
            with r2:
                st.metric("Racks Down", final_state['affected_racks'],
                         delta=f"of {total_racks} total", delta_color="inverse")
            with r3:
                st.metric("Cascade Duration", f"{final_state['cascade_duration_minutes']:.1f} min")
            with r4:
                pct_affected = final_state['affected_racks'] / total_racks * 100
                st.metric("Facility Impact", f"{pct_affected:.0f}%")
            
            # Timeline visualization
            st.markdown("### ⏱️ Cascade Timeline")
            
            for event in timeline:
                if event.severity == "failure":
                    icon = "🔴"
                elif event.severity == "critical":
                    icon = "🟠"
                else:
                    icon = "🟡"
                
                time_str = f"{event.time_seconds:.0f}s" if event.time_seconds < 60 else f"{event.time_seconds/60:.1f}min"
                st.markdown(f"{icon} **[{time_str}]** {event.description}")
            
            # Annotation
            st.markdown(f"""
            <div style="background: #FFEBEE; border-left: 4px solid #F44336; padding: 12px; margin: 16px 0; border-radius: 4px;">
                <strong style="color: #F44336; font-size: 15px;">⚠️ Cascade Impact Summary</strong><br>
                <span style="font-size: 14px;">A {degradation}% failure of <strong>{engine.components[selected_id].name}</strong> cascades to <strong>{final_state['total_affected']} components</strong> including <strong>{final_state['affected_racks']} racks</strong> within <strong>{final_state['cascade_duration_minutes']:.1f} minutes</strong>. 
                {'This represents a major facility event requiring immediate intervention.' if pct_affected > 20 else 'The impact is contained to a limited zone.'}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === RISK SCORES & CSPCF ===
        st.subheader("🎯 Cascade Risk Scores (CRS)")
        
        if st.button("📊 Compute All Risk Scores"):
            with st.spinner("Computing cascade risk for all components..."):
                risk_scores = engine.compute_cascade_risk_scores()
                st.session_state.dcplos_risk_scores = risk_scores
                cspcf = engine.identify_cspcf(threshold_multiplier=1.5)
                st.session_state.dcplos_cspcf = cspcf
            st.success("✅ Risk scores computed")
            st.rerun()
        
        if 'dcplos_risk_scores' in st.session_state:
            risk_scores = st.session_state.dcplos_risk_scores
            
            # Top risk components chart
            sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            risk_names = [engine.components[cid].name for cid, _ in sorted_risks]
            risk_values = [v for _, v in sorted_risks]
            
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                y=risk_names[::-1], x=risk_values[::-1],
                orientation='h',
                marker_color=['#F44336' if v > np.mean(risk_values)*1.5 else '#FF9800' if v > np.mean(risk_values) else '#4CAF50' 
                             for v in risk_values[::-1]]
            ))
            fig_risk.update_layout(
                title="Cascade Risk Score by Component (higher = more dangerous if it fails)",
                xaxis_title="CRS", yaxis_title="",
                height=450, margin=dict(l=200)
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.markdown("""
            <div style="background: #FFF3E0; border-left: 4px solid #FF9800; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                <strong style="color: #FF9800;">How to read CRS:</strong> Higher score = component failure would cascade to more critical systems. Red bars = critical single points of cascading failure (CSPCF). These should be prioritized for maintenance, redundancy upgrades, or load rebalancing.
            </div>
            """, unsafe_allow_html=True)
            
            # CSPCF Table
            if 'dcplos_cspcf' in st.session_state and st.session_state.dcplos_cspcf:
                st.markdown("### 🚨 Critical Single Points of Cascading Failure (CSPCF)")
                cspcf_data = st.session_state.dcplos_cspcf
                cspcf_df = pd.DataFrame(cspcf_data)
                cspcf_df = cspcf_df[['name', 'type', 'crs', 'blast_radius', 'affected_racks', 'cascade_duration_min', 'failure_probability']]
                cspcf_df.columns = ['Component', 'Type', 'CRS', 'Blast Radius', 'Racks Affected', 'Cascade (min)', 'P(Failure)']
                st.dataframe(cspcf_df, use_container_width=True, hide_index=True)
                
                st.markdown(f"""
                <div style="background: #FFEBEE; border-left: 4px solid #F44336; padding: 12px; margin: 16px 0; border-radius: 4px;">
                    <strong style="color: #F44336; font-size: 15px;">🚨 {len(cspcf_data)} Critical Vulnerabilities Identified</strong><br>
                    <span style="font-size: 14px;">These components pose disproportionate systemic risk. Their failure cascades to {cspcf_data[0]['blast_radius']}+ downstream components. 
                    <strong>Priority actions:</strong> Add redundancy, reduce load, or expedite maintenance for the top-ranked items.</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === RISK HEATMAP ===
        st.subheader("🗺️ Facility Risk Heatmap")
        
        if st.button("🗺️ Generate Risk Heatmap"):
            with st.spinner("Computing rack exposure scores..."):
                heatmap_data = engine.get_risk_heatmap_data()
                st.session_state.dcplos_heatmap = heatmap_data
            st.rerun()
        
        if 'dcplos_heatmap' in st.session_state:
            heatmap_data = st.session_state.dcplos_heatmap
            hm_df = pd.DataFrame(heatmap_data)
            
            # Create heatmap grid
            rows = sorted(hm_df['row'].unique())
            positions = sorted(hm_df['position'].unique())
            
            heatmap_matrix = np.zeros((len(rows), len(positions)))
            for _, row_data in hm_df.iterrows():
                r_idx = rows.index(row_data['row'])
                p_idx = positions.index(row_data['position'])
                heatmap_matrix[r_idx, p_idx] = row_data['exposure_score']
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_matrix,
                x=[f"Pos {p}" for p in positions],
                y=[f"Row {r}" for r in rows],
                colorscale='RdYlGn_r',  # Red=high risk, Green=low risk
                text=[[f"{heatmap_matrix[r][p]:.2f}" for p in range(len(positions))] for r in range(len(rows))],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Row %{y}, %{x}<br>Exposure Score: %{z:.3f}<extra></extra>"
            ))
            fig_heatmap.update_layout(
                title="Rack Cascade Exposure Score — Red = High Vulnerability, Green = Low Vulnerability",
                xaxis_title="Rack Position", yaxis_title="Row",
                height=450
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Identify highest risk zone
            max_row = hm_df.loc[hm_df['exposure_score'].idxmax()]
            st.markdown(f"""
            <div style="background: #FFEBEE; border-left: 4px solid #F44336; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                <strong style="color: #F44336;">Highest Risk Position: Row {int(max_row['row'])}, Position {int(max_row['position'])}</strong> (Exposure Score: {max_row['exposure_score']:.3f})<br>
                <span style="font-size: 13px;">Avoid placing critical workloads here. This position has the highest cascade exposure from upstream infrastructure vulnerabilities.</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Safest positions
            safest = hm_df.nsmallest(5, 'exposure_score')
            st.markdown(f"""
            <div style="background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                <strong style="color: #4CAF50;">Safest Positions for Critical Workloads:</strong><br>
                <span style="font-size: 13px;">{', '.join([f"Row {int(r['row'])}-Pos {int(r['position'])}" for _, r in safest.iterrows()])} — Lowest cascade exposure, most infrastructure redundancy upstream.</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === RISK-AWARE PLACEMENT ===
        st.subheader("📍 Risk-Aware Workload Placement")
        st.markdown("Find optimal positions for new workloads considering capacity AND cascade exposure.")
        
        pl_col1, pl_col2, pl_col3 = st.columns(3)
        with pl_col1:
            workload_kw = st.number_input("Workload Power (kW)", min_value=5, max_value=100, value=25, step=5)
        with pl_col2:
            criticality = st.selectbox("Workload Criticality", ["critical", "standard", "low"])
        with pl_col3:
            num_needed = st.number_input("Positions Needed", min_value=1, max_value=20, value=3, step=1)
        
        if st.button("📍 Find Optimal Placement", type="primary"):
            with st.spinner("Analyzing placement options against cascade risk..."):
                recommendations = engine.recommend_placement(workload_kw, criticality, num_needed)
            
            if recommendations:
                st.markdown(f"### Placement Recommendations ({workload_kw}kW, {criticality} workload)")
                
                rec_df = pd.DataFrame(recommendations[:10])
                display_cols = ['rank', 'name', 'deliverable_kw', 'cascade_exposure', 'risk_increase', 'headroom_after', 'binding_constraint', 'recommendation']
                st.dataframe(rec_df[display_cols], use_container_width=True, hide_index=True)
                
                # Top pick annotation
                top = recommendations[0]
                st.markdown(f"""
                <div style="background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 12px; margin: 16px 0; border-radius: 4px;">
                    <strong style="color: #4CAF50; font-size: 15px;">✅ Best Position: {top['name']} (Row {top['row']}, Position {top['position']})</strong><br>
                    <span style="font-size: 14px;">Deliverable capacity: {top['deliverable_kw']}kW | Headroom after placement: {top['headroom_after']}kW | Cascade exposure: {top['cascade_exposure']:.3f}<br>
                    Binding constraint: <strong>{top['binding_constraint']}</strong> | This position minimizes systemic risk for a {criticality} workload.</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Worst choice for contrast
                if len(recommendations) > 5:
                    worst = recommendations[-1]
                    st.markdown(f"""
                    <div style="background: #FFEBEE; border-left: 4px solid #F44336; padding: 10px; margin-bottom: 16px; border-radius: 4px;">
                        <strong style="color: #F44336;">❌ Worst Available: {worst['name']}</strong> — Cascade exposure: {worst['cascade_exposure']:.3f} (highest risk position that technically has capacity). Avoid for {criticality} workloads.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ No rack positions can accommodate {workload_kw}kW with current constraints. Consider unlock moves or infrastructure upgrades.")
        
        st.markdown("---")
        
        # === UNLOCK MOVES ===
        st.subheader("🔓 Unlock Moves (Capacity Defragmentation)")
        st.markdown("Identify workload relocations that free capacity in constrained zones — like defragmenting a disk.")
        
        if st.button("🔓 Identify Unlock Moves"):
            with st.spinner("Analyzing relocation opportunities..."):
                moves = engine.identify_unlock_moves()
                yield_data = engine.compute_facility_yield(workload_kw)
            
            if moves:
                st.markdown(f"### {len(moves)} Unlock Moves Identified")
                
                moves_df = pd.DataFrame(moves)
                display_move_cols = ['source_rack', 'source_load_kw', 'dest_rack', 'capacity_freed_kw', 'constrained_zone', 'benefit_ratio']
                moves_df.columns = [c.replace('_', ' ').title() for c in moves_df.columns]
                st.dataframe(moves_df, use_container_width=True, hide_index=True)
                
                total_freeable = sum(m['capacity_freed_kw'] for m in moves)
                st.markdown(f"""
                <div style="background: #E3F2FD; border-left: 4px solid #1976D2; padding: 12px; margin: 16px 0; border-radius: 4px;">
                    <strong style="color: #1976D2; font-size: 15px;">💡 Total Capacity Recoverable: {total_freeable:.1f} kW</strong><br>
                    <span style="font-size: 14px;">By executing {len(moves)} workload relocations, you can free {total_freeable:.1f}kW of capacity in constrained cooling/power zones. 
                    This is equivalent to deploying {int(total_freeable/workload_kw)} additional {workload_kw}kW racks without any infrastructure upgrades.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ No constrained zones detected — all infrastructure zones are operating within comfortable margins.")
            
            # Facility yield summary
            st.markdown("### 📊 Facility Yield Analysis")
            y1, y2, y3 = st.columns(3)
            with y1:
                st.metric("Deployable Now", f"{yield_data['deployable_without_orchestration']} racks",
                         help=f"Positions that can accept {workload_kw}kW today without changes")
            with y2:
                st.metric("Deployable After Moves", f"{yield_data['deployable_with_orchestration']} racks",
                         delta=f"+{yield_data['additional_racks_from_moves']} from unlock moves")
            with y3:
                st.metric("Capacity Gain", f"+{yield_data['capacity_gain_pct']}%",
                         help="Additional capacity unlocked through orchestration")
    
    else:
        st.info("👆 Click 'Load Demo Facility' to initialize the DCPLOS engine with a 40-rack facility model.")
        st.markdown("""
        **DCPLOS provides:**
        - 💥 **Cascade Failure Simulation** — "What happens if X fails?" with full propagation timeline
        - 🎯 **Cascade Risk Scores** — Every component scored for systemic danger
        - 🚨 **CSPCF Detection** — Critical single points of cascading failure identified automatically
        - 🗺️ **Risk Heatmap** — Visual map showing cascade vulnerability at every rack position
        - 🛡️ **Risk-Aware Placement** — "Deploy here, not there" based on cascade exposure
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>OPTENA | AI-Powered Data Center Energy Optimization</p>",
    unsafe_allow_html=True
)
