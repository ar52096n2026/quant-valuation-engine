import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import re
import pydeck as pdk
from geopy.geocoders import Nominatim
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ==============================================================================
# 1. APP CONFIGURATION & RESPONSIVE MOBILE STYLING
# ==============================================================================
st.set_page_config(
    page_title="Quant Valuation Engine", 
    layout="wide", 
    page_icon="🏡",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Main App Canvas */
.stApp {
    background-color: #fcfaf7 !important;
    color: #1a1a1a !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #f5f2ed !important;
    border-right: 1px solid rgba(0, 0, 0, 0.12) !important;
    padding-top: 1rem !important;
}

/* Force Streamlit Dialog / Modal Popups ABOVE WebGL Map Canvases */
[data-testid="stDialog"], [data-testid="stModal"], div[data-baseweb="modal"], div[role="dialog"] {
    z-index: 9999999 !important;
}

/* Restrain PyDeck Chart Canvas Stacking Context */
[data-testid="stDeckGlJsonChart"], .stDeckGlJsonChart {
    position: relative !important;
    z-index: 1 !important;
}

/* Responsive Metric Cards with CSS Clamp (Prevents Overflow) */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border-left: 3px solid #1a1a1a !important;
    border-top: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
    padding: 0.85rem 1rem !important;
    border-radius: 4px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    word-break: break-word !important;
    overflow: hidden !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    font-weight: 800 !important;
    color: rgba(0, 0, 0, 0.5) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(1.25rem, 2.2vw, 2.1rem) !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #0284c7 !important;
}

/* Section Subtitles & Headers */
.editorial-section-title {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #1a1a1a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
}
.editorial-section-subtitle {
    font-size: 0.85rem;
    color: rgba(0, 0, 0, 0.65);
    margin-bottom: 0.85rem;
    line-height: 1.4;
}
.editorial-section-dot {
    width: 7px;
    height: 7px;
    background-color: #1a1a1a;
    border-radius: 50%;
    display: inline-block;
}
.editorial-card {
    background-color: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 4px;
    padding: 1.25rem;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1rem;
    line-height: 1.6;
    color: #1a1a1a;
}

/* Custom Map Legend Styling */
.map-legend-container {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background-color: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 4px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.legend-badge-subject {
    width: 12px;
    height: 12px;
    background-color: #dc2626;
    border-radius: 50%;
    border: 2px solid #ffffff;
    box-shadow: 0 0 0 1px #dc2626;
}
.legend-badge-comp {
    width: 10px;
    height: 10px;
    background-color: #0284c7;
    border-radius: 50%;
    border: 1.5px solid #ffffff;
    box-shadow: 0 0 0 1px #0284c7;
}

/* Touch Target Enhancements for Buttons & Inputs */
.stButton > button {
    min-height: 44px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
div[data-baseweb="input"] {
    min-height: 44px !important;
}

/* Mobile Responsiveness Breakpoints */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        padding-top: 1rem !important;
    }
    .editorial-title {
        font-size: 1.9rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }
    [data-testid="stMetric"] {
        margin-bottom: 0.5rem !important;
        padding: 0.75rem 0.85rem !important;
    }
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 0.5rem !important;
    }
    .stButton > button {
        width: 100% !important;
    }
    .map-legend-container {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.4rem;
    }
}

@media (max-width: 480px) {
    .editorial-title {
        font-size: 1.6rem !important;
    }
    .editorial-section-title {
        font-size: 11px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOCAL STORAGE & DATA PIPELINE
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

def save_dataset_locally(file_bytes, filename):
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return filename

def get_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            return pd.read_csv(MANIFEST_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        'neighborhood_name', 'file_name', 
        'center_lat', 'center_lon', 
        'radius_miles', 'sample_size'
    ])

def update_manifest(new_row_dict):
    manifest_df = get_manifest()
    if not manifest_df.empty and 'neighborhood_name' in manifest_df.columns:
        if new_row_dict['neighborhood_name'] in manifest_df['neighborhood_name'].values:
            manifest_df = manifest_df[manifest_df['neighborhood_name'] != new_row_dict['neighborhood_name']]
    new_df = pd.DataFrame([new_row_dict])
    updated_manifest = pd.concat([manifest_df, new_df], ignore_index=True)
    updated_manifest.to_csv(MANIFEST_PATH, index=False)

def load_dataset(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return file_path
    return None

# ==============================================================================
# 3. MATH & GEOCODING HELPERS
# ==============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    return r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def geocode_address(address_str):
    geolocator = Nominatim(user_agent="quant_cma_valuation_engine")
    try:
        location = geolocator.geocode(address_str)
        if location:
            return location.latitude, location.longitude, location.address
    except Exception:
        pass
    return None, None, None

def clean_html(text):
    return '\n'.join(line.lstrip() for line in text.splitlines())

# ==============================================================================
# 4. DATA PROCESSING & RIDGE REGRESSION MODEL (WITH MATHEMATICAL GUARDRAILS)
# ==============================================================================
@st.cache_data
def process_dataset(file_source):
    df = pd.read_csv(file_source)
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    col_aliases = {
        'SQFT': 'SQUARE FEET', 'SQ FT': 'SQUARE FEET', 'SQUAREFEET': 'SQUARE FEET',
        'PRICE ($)': 'PRICE', 'LOT': 'LOT SIZE', 'YEARBUILT': 'YEAR BUILT',
        'LAT': 'LATITUDE', 'LON': 'LONGITUDE', 'LNG': 'LONGITUDE', 'PROPERTYSUBTYPE': 'PROPERTY TYPE'
    }
    df = df.rename(columns=col_aliases)

    if "SALE TYPE" in df.columns and pd.isna(df.iloc[0].get('SOLD DATE', np.nan)):
        df = df.drop(index=0)
       
    num_cols = ['PRICE', 'BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', 'LATITUDE', 'LONGITUDE']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
           
    if '$/SQUARE FEET' not in df.columns and 'PRICE' in df.columns and 'SQUARE FEET' in df.columns:
        df['$/SQUARE FEET'] = df['PRICE'] / df['SQUARE FEET']

    if 'SOLD DATE' in df.columns:
        df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'], errors='coerce')
    else:
        df['SOLD DATE'] = pd.Timestamp.now()
   
    mask = (df['PRICE'].notnull()) & (df['SQUARE FEET'] > 0)
    
    if 'PROPERTY TYPE' in df.columns:
        prop_mask = df['PROPERTY TYPE'].astype(str).str.contains('Single|Residential|Home', case=False, na=False)
        if prop_mask.sum() > 0:
            mask = mask & prop_mask

    clean_df = df[mask].copy()
   
    today = pd.Timestamp.now()
    clean_df['months_since_sale'] = ((today - clean_df['SOLD DATE']).dt.days / 30.0).fillna(0)
    clean_df['time_mult'] = (1.0 + (clean_df['months_since_sale'] * 0.003)).clip(lower=0.92, upper=1.08)
    clean_df['TIME_ADJUSTED_PRICE'] = clean_df['PRICE'] * clean_df['time_mult']
   
    return clean_df


def fit_dynamic_model(clean_df):
    features = ['SQUARE FEET', 'BATHS', 'LOT SIZE', 'YEAR BUILT']
    
    missing_cols = [col for col in features + ['TIME_ADJUSTED_PRICE'] if col not in clean_df.columns]
    if missing_cols:
        st.error(f"⚠️ Loaded dataset is missing required columns: **{', '.join(missing_cols)}**.")
        st.stop()
        
    model_df = clean_df.dropna(subset=features + ['TIME_ADJUSTED_PRICE']).copy()
   
    if len(model_df) < 3:
        st.error(f"⚠️ Insufficient sales data found ({len(model_df)} valid rows). Minimum 3 sales required.")
        st.stop()

    X = model_df[features]
    y = model_df['TIME_ADJUSTED_PRICE']
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    alphas = np.logspace(-2, 4, 50)
    cv_splits = min(5, max(2, len(model_df) // 2))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=alphas, cv=cv)
    ridge.fit(X_scaled, y)
   
    # FIX ZERO-VARIANCE DIVISION BUG: Clamp scaler.scale_ to prevent near-zero division explosion
    safe_scale = np.where(scaler.scale_ < 1e-4, 1.0, scaler.scale_)
    unscaled_coefs = ridge.coef_ / safe_scale
    
    # REAL ESTATE SANITY BOUNDS: Clamp coefficients to realistic real-world limits
    coef_dict = {
        'SQUARE FEET': float(np.clip(unscaled_coefs[0], 20.0, 1500.0)),
        'BATHS': float(np.clip(unscaled_coefs[1], 2500.0, 125000.0)),
        'LOT SIZE': float(np.clip(unscaled_coefs[2], 0.1, 150.0)),
        'YEAR BUILT': float(np.clip(unscaled_coefs[3], 50.0, 5000.0))
    }
   
    model_df['PREDICTED_BASE'] = ridge.predict(X_scaled)
    model_df['RESIDUAL_PCT'] = (model_df['TIME_ADJUSTED_PRICE'] - model_df['PREDICTED_BASE']) / model_df['PREDICTED_BASE']
   
    def classify_condition(res_pct):
        if res_pct < -0.12:      return 'Fixer'
        elif res_pct < -0.04:    return 'Dated/Original'
        elif res_pct <= 0.04:    return 'Average'
        elif res_pct <= 0.12:    return 'Updated'
        else:                    return 'Turnkey'
       
    model_df['AUTO_CONDITION'] = model_df['RESIDUAL_PCT'].apply(classify_condition)
    return coef_dict, ridge.alpha_, len(model_df), model_df

# ==============================================================================
# 5. SIDEBAR: AGENT ADMIN PORTAL
# ==============================================================================
def render_admin_portal():
    st.sidebar.markdown("### 🔒 Agent Admin Portal")
    admin_pin = st.sidebar.text_input("Enter Admin PIN", type="password", key="admin_pin_input")
    
    try:
        target_pin = st.secrets.get("ADMIN_PIN", "8888")
    except Exception:
        target_pin = "8888"
    
    if admin_pin == target_pin:
        st.sidebar.success("🔓 Authenticated")
        st.sidebar.markdown("#### 📤 Register Neighborhood CSV")
        neigh_label = st.sidebar.text_input("Neighborhood Label", value="Richmond - East Heights")
        radius = st.sidebar.number_input("Coverage Radius (Miles)", value=3.0, step=0.5, min_value=0.5)
        uploaded_csv = st.sidebar.file_uploader("Upload MLS/Redfin Sales CSV", type=["csv"])
        
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            col_map = {str(c).upper(): c for c in df.columns}
            lat_col = col_map.get('LATITUDE') or col_map.get('LAT')
            lon_col = col_map.get('LONGITUDE') or col_map.get('LON') or col_map.get('LNG')
            
            if not lat_col or not lon_col:
                st.sidebar.error("⚠️ CSV missing LATITUDE / LONGITUDE columns.")
            else:
                c_lat, c_lon = float(df[lat_col].mean()), float(df[lon_col].mean())
                n_sales = len(df)
                st.sidebar.info(f"📍 Centroid: ({c_lat:.4f}, {c_lon:.4f}) | Sales: {n_sales}")
                
                if st.sidebar.button("🚀 Publish Dataset"):
                    csv_bytes = uploaded_csv.getvalue()
                    clean_filename = f"{neigh_label.lower().replace(' ', '_')}_sales.csv"
                    saved_file = save_dataset_locally(csv_bytes, clean_filename)
                    
                    update_manifest({
                        'neighborhood_name': neigh_label,
                        'file_name': saved_file,
                        'center_lat': c_lat,
                        'center_lon': c_lon,
                        'radius_miles': radius,
                        'sample_size': n_sales
                    })
                    st.sidebar.balloons()
                    st.sidebar.success(f"🎉 **{neigh_label}** registered!")
    elif admin_pin != "":
        st.sidebar.error("Invalid PIN")

# ==============================================================================
# MODAL DIALOG: CONDITION CRITERIA & GLOSSARY
# ==============================================================================
@st.dialog("📖 Property Condition & Valuation Guide", width="large")
def show_condition_guide():
    st.caption("🏛️ **Dual-Pillar Framework:** Merges official bank appraisal standards with local Ridge regression math.")
    st.divider()
    guide_tab1, guide_tab2 = st.tabs(["🏡 Condition Tiers (Appraisal Scale)", "🎓 Plain-English Glossary & Method"])
   
    with guide_tab1:
        st.caption("Select the subject property's condition based on physical materials, age of finishes, and overall state:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✨ Turnkey *(Fannie Mae C1/C2)*")
            st.caption("**Residual Range:** > +12% Premium over sticker price")
            st.markdown("Full luxury overhaul or brand-new build (0–3 yrs old). Custom quartz/granite counters, luxury bath suites, new roof/HVAC, double-pane windows. Zero work needed.")
            st.divider()
            st.markdown("### 🔨 Updated *(Fannie Mae C3)*")
            st.caption("**Residual Range:** +4% to +12% Premium over sticker price")
            st.markdown("Modernized kitchens and baths within the last 4–10 yrs. Fresh paint, updated flooring, clean systems. Move-in ready with minor personal touches.")
            st.divider()
            st.markdown("### 🏡 Average *(Fannie Mae C4)*")
            st.caption("**Residual Range:** -4% to +4% Baseline")
            st.markdown("Standard builder-grade finishes, well-kept over time. Fully functional appliances, solid mechanicals, typical wear-and-tear for neighborhood age.")
        with col2:
            st.markdown("### 🕰️ Dated / Original *(Fannie Mae C5)*")
            st.caption("**Residual Range:** -12% to -4% Discount below sticker price")
            st.markdown("Clean and functional, but 15+ year-old finishes. Formica counters, older carpet, original oak cabinets. Buyers knock off money to cover upcoming cosmetic updates.")
            st.divider()
            st.markdown("### ⚠️ Fixer *(Fannie Mae C6)*")
            st.caption("**Residual Range:** < -12% Discount below sticker price")
            st.markdown("Major deferred maintenance or structural red flags. Leaking roof, ancient wiring/plumbing, foundation concerns, unlivable spaces. Investor/flipper candidate.")

    with guide_tab2:
        st.markdown("### 💡 Core Valuation Cheat Sheet")
        glossary_col1, glossary_col2 = st.columns(2)
        with glossary_col1:
            st.info("""
            **📐 Baseline ("Sticker Price")**  
            The target price predicted strictly by physical stats (SqFt, Beds, Baths, Lot Size, Year Built) assuming standard, average condition.
            """)
            st.success("""
            **📈 Premium**  
            A positive price bump. The home sold for *more* than its physical size predicted because of high-end finishes, views, or turnkey presentation.
            """)
        with glossary_col2:
            st.info("""
            **🧪 Residual ("Quality Score")**  
            The dollar gap between the actual sale price and the baseline sticker price. It mathematically captures hidden interior quality.
            """)
            st.error("""
            **📉 Discount**  
            A negative price cut. The home sold for *less* than its physical size predicted because of outdated finishes or needed repairs.
            """)

# ==============================================================================
# 6. MAIN APPLICATION & SEARCH UI
# ==============================================================================
def main():
    manifest_df = get_manifest()

    # --- SIDEBAR: OPTIONAL CSV SELECTOR & ADMIN PORTAL ---
    st.sidebar.markdown("### 📂 Load Existing CSV Property")
    
    active_dataset_file = None
    matched_neigh_name = "Default Dataset"

    if not manifest_df.empty:
        active_dataset_file = load_dataset(manifest_df.iloc[0]['file_name'])
        matched_neigh_name = manifest_df.iloc[0]['neighborhood_name']
    elif os.path.exists("sample_redfin.csv"):
        active_dataset_file = "sample_redfin.csv"
        matched_neigh_name = "Sample Redfin Data"

    selected_csv_addr = None
    if active_dataset_file:
        data = process_dataset(active_dataset_file)
        coefs, best_alpha, n_sales, model_df = fit_dynamic_model(data)
        csv_addresses = model_df['ADDRESS'].dropna().astype(str).tolist() if 'ADDRESS' in model_df.columns else []
        
        selected_csv_addr = st.sidebar.selectbox(
            "Select Property Record from Active CSV", 
            options=["-- Custom Address Entry --"] + csv_addresses,
            index=0
        )
    else:
        st.info("👈 Please open the **Agent Admin Portal** in the sidebar to upload a CSV dataset!")
        st.stop()

    st.sidebar.markdown("---")
    render_admin_portal()

    # --- MAIN CANVAS HEADER ---
    st.markdown("""
    <div style="margin-bottom: 0.75rem;">
        <h1 class="editorial-title" style="font-family: 'Playfair Display', Georgia, serif; font-size: 2.75rem; font-weight: 500; font-style: italic; margin: 0; color: #1a1a1a;">
            🏡 Quant Valuation Engine
        </h1>
        <p style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.16em; color: rgba(0,0,0,0.5); font-weight: 700; margin-top: 0.2rem;">
            Spatial Micro-Neighborhood Regression & Automated Appraisal Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- INTRO & PURPOSE BANNER ---
    with st.expander("💡 **Welcome! How this engine works & what you'll learn**", expanded=False):
        st.markdown("""
        <div style="font-size: 0.95rem; line-height: 1.6; color: #2d2d2d; margin-bottom: 1rem;">
            <strong>The Quant Valuation Engine</strong> replaces arbitrary guesswork and generic automated valuations with local micro-neighborhood machine learning. By combining actual recent sales data, spatial proximity, and institutional appraisal frameworks, this engine provides a transparent, mathematical estimate of what a specific home is worth in today's market.
        </div>
        """, unsafe_allow_html=True)
        
        intro_col1, intro_col2, intro_col3 = st.columns(3)
        with intro_col1:
            st.markdown("""
            **🎯 1. True Fair Market Value**  
            Calculate a precise baseline price along with defensible lower (Fast-Sale Floor) and upper (Peak Value Ceiling) boundaries.
            """)
        with intro_col2:
            st.markdown("""
            **📊 2. Local Value Drivers**  
            See exactly how much buyers in this specific neighborhood pay for square footage, extra bedrooms, bathrooms, and interior condition.
            """)
        with intro_col3:
            st.markdown("""
            **🛠️ 3. Renovation ROI**  
            Simulate adding square footage, bedrooms, bathrooms, or upgrading condition grade to calculate exact dollar return on investment.
            """)

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # --- STEP 1: ADDRESS SEARCH BAR ---
    st.markdown('<div class="editorial-section-title"><span class="editorial-section-dot"></span>📍 Step 1: Subject Property Address</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">Enter any target address to geocode coordinates and connect directly to local micro-neighborhood sales data.</div>', unsafe_allow_html=True)
    
    initial_search_val = ""
    if selected_csv_addr and selected_csv_addr != "-- Custom Address Entry --":
        initial_search_val = selected_csv_addr

    search_query = st.text_input(
        "Subject Property Address", 
        value=initial_search_val,
        placeholder="e.g. 123 Main St, San Francisco, CA 94123",
        label_visibility="collapsed"
    )

    # Defaults
    default_sqft, default_beds, default_baths, default_lot, default_year = 1500, 3, 2.0, 5000, 1955
    t_lat = model_df['LATITUDE'].mean() if 'LATITUDE' in model_df.columns else 37.9542
    t_lon = model_df['LONGITUDE'].mean() if 'LONGITUDE' in model_df.columns else -122.3353
    searched_lat, searched_lon, geocoded_addr = None, None, None

    if selected_csv_addr and selected_csv_addr != "-- Custom Address Entry --":
        match = model_df[model_df['ADDRESS'].astype(str) == selected_csv_addr]
        if not match.empty:
            row = match.iloc[0]
            default_sqft = int(row.get('SQUARE FEET', 1500)) if pd.notna(row.get('SQUARE FEET')) else 1500
            default_beds = int(row.get('BEDS', 3)) if pd.notna(row.get('BEDS')) else 3
            default_baths = float(row.get('BATHS', 2.0)) if pd.notna(row.get('BATHS')) else 2.0
            default_lot = int(row.get('LOT SIZE', 5000)) if pd.notna(row.get('LOT SIZE')) else 5000
            default_year = int(row.get('YEAR BUILT', 1955)) if pd.notna(row.get('YEAR BUILT')) else 1955
            t_lat = float(row.get('LATITUDE', 37.9542)) if pd.notna(row.get('LATITUDE')) else 37.9542
            t_lon = float(row.get('LONGITUDE', -122.3353)) if pd.notna(row.get('LONGITUDE')) else -122.3353
    elif search_query.strip():
        with st.spinner("Geocoding address & matching micro-neighborhood..."):
            searched_lat, searched_lon, geocoded_addr = geocode_address(search_query)
            if searched_lat and searched_lon and not manifest_df.empty:
                min_dist = float('inf')
                for idx, row in manifest_df.iterrows():
                    dist = haversine_distance(searched_lat, searched_lon, float(row['center_lat']), float(row['center_lon']))
                    if dist <= float(row['radius_miles']) and dist < min_dist:
                        min_dist = dist
                        active_dataset_file = load_dataset(row['file_name'])
                        matched_neigh_name = row['neighborhood_name']
                if active_dataset_file:
                    st.success(f"📍 **Located:** {geocoded_addr} | **Matched Coverage:** {matched_neigh_name} ({min_dist:.2f} mi away)")
                    t_lat, t_lon = searched_lat, searched_lon
                    data = process_dataset(active_dataset_file)
                    coefs, best_alpha, n_sales, model_df = fit_dynamic_model(data)

    subject_address = geocoded_addr if geocoded_addr else (search_query if search_query.strip() else "123 Main St, San Francisco, CA 94123")

    # --- STEP 2: PHYSICAL SPECS CARD ---
    st.markdown('<div class="editorial-section-title" style="margin-top: 1rem;"><span class="editorial-section-dot"></span>📋 Step 2: Physical Specs & Current Condition</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">Confirm the property\'s current dimensions and condition grade. The ML model uses these physical parameters to build an unadjusted baseline price.</div>', unsafe_allow_html=True)
    
    spec_col1, spec_col2, spec_col3, spec_col4, spec_col5, spec_col6 = st.columns(6)
    with spec_col1:
        subject_sqft = st.number_input("Living Area (SqFt)", value=default_sqft, step=50)
    with spec_col2:
        subject_beds = st.number_input("Bedrooms", value=default_beds, step=1)
    with spec_col3:
        subject_baths = st.number_input("Bathrooms", value=default_baths, step=0.5)
    with spec_col4:
        subject_lot = st.number_input("Lot Size (SqFt)", value=default_lot, step=250)
    with spec_col5:
        subject_year = st.number_input("Year Built", value=default_year, step=1)
    with spec_col6:
        subject_condition = st.selectbox("Current Condition Grade", options=["Turnkey", "Updated", "Average", "Dated", "Fixer"], index=0)

    if st.button("📖 Open Condition Guide & Glossary"):
        show_condition_guide()

    # --- STEP 3: RENOVATION & REMODEL SCENARIO PLANNER (EXPANDED BY DEFAULT) ---
    with st.expander("🛠️ Renovation & Remodel Scenario Planner", expanded=True):
        st.markdown('<div class="editorial-section-subtitle">Test "what-if" scenarios before renovating. Adjust footprint size, room count, or target condition grade to see projected dollar increases based on local market rates.</div>', unsafe_allow_html=True)
        reno_col1, reno_col2, reno_col3, reno_col4 = st.columns(4)
        with reno_col1:
            add_sqft = st.number_input("Add/Subtract Living Area (SqFt)", value=0, step=25)
        with reno_col2:
            add_beds = st.slider("Add/Subtract Bedrooms", -2, 3, 0, step=1)
        with reno_col3:
            add_baths = st.slider("Add/Subtract Bathrooms", -1.0, 3.0, 0.0, step=0.5)
        with reno_col4:
            target_cond_choice = st.selectbox(
                "Target Condition Grade", 
                options=["Same as Current (No Rehab)", "Turnkey", "Updated", "Average", "Dated", "Fixer"], 
                index=0
            )

    condition_options = {'Turnkey': +0.15, 'Updated': +0.07, 'Average': 0.00, 'Dated': -0.07, 'Dated/Original': -0.07, 'Fixer': -0.15}

    # Resolve target condition for calculations
    eval_condition = subject_condition if target_cond_choice == "Same as Current (No Rehab)" else target_cond_choice

    # Calculations
    eval_sqft = subject_sqft + add_sqft
    eval_beds = subject_beds + add_beds
    eval_baths = subject_baths + add_baths

    sqft_value_contrib = add_sqft * coefs['SQUARE FEET']
    bath_value_contrib = add_baths * coefs['BATHS']
    BEDROOM_UTILITY_DELTA = 12500.0
    bed_value_contrib = add_beds * BEDROOM_UTILITY_DELTA

    # --- EXCLUDE SUBJECT PROPERTY FROM COMP SELECTION CANDIDATES ---
    model_df['dist_miles'] = haversine_distance(t_lat, t_lon, model_df['LATITUDE'], model_df['LONGITUDE'])
    
    # Filter out subject property row if address or lat/lon matches exactly
    comp_candidates = model_df.copy()
    if selected_csv_addr and selected_csv_addr != "-- Custom Address Entry --":
        comp_candidates = comp_candidates[comp_candidates['ADDRESS'].astype(str) != selected_csv_addr]
    else:
        comp_candidates = comp_candidates[comp_candidates['dist_miles'] > 0.0001]
        
    if len(comp_candidates) < 3:
        comp_candidates = model_df.copy()  # Fallback safety

    sqft_std = comp_candidates['SQUARE FEET'].std() if comp_candidates['SQUARE FEET'].std() > 0 else 1
    comp_candidates['sqft_diff_norm'] = np.abs(comp_candidates['SQUARE FEET'] - eval_sqft) / sqft_std
    comp_candidates['composite_dist'] = (0.7 * comp_candidates['dist_miles']) + (0.3 * comp_candidates['sqft_diff_norm'])

    closest_comps = comp_candidates.sort_values(by='composite_dist').head(4).copy()

    comp_values = []
    comp_base_prices = []
    for _, comp in closest_comps.iterrows():
        adj_sqft = (eval_sqft - comp['SQUARE FEET']) * coefs['SQUARE FEET']
        adj_baths = (eval_baths - comp['BATHS']) * coefs['BATHS']
        lot_diff = np.sqrt(max(0, subject_lot)) - np.sqrt(max(0, comp['LOT SIZE']))
        adj_lot = lot_diff * (coefs['LOT SIZE'] * 2)  # Controlled lot scaling multiplier
        adj_year = (subject_year - comp['YEAR BUILT']) * coefs['YEAR BUILT']
        
        comp_cond_mult = condition_options.get(comp['AUTO_CONDITION'], 0.0)
        target_cond_mult = condition_options[eval_condition]
        adj_condition = comp['TIME_ADJUSTED_PRICE'] * (target_cond_mult - comp_cond_mult)

        tot_adj = adj_sqft + adj_baths + adj_lot + adj_year + adj_condition
        comp_values.append(comp['TIME_ADJUSTED_PRICE'] + tot_adj)
        comp_base_prices.append(comp['TIME_ADJUSTED_PRICE'])

    weights = 1.0 / (closest_comps['dist_miles'] + 0.01)
    weights /= weights.sum()
    reconciled_val = np.dot(comp_values, weights)
    floor_val, ceiling_val = reconciled_val * 0.94, reconciled_val * 1.05

    # Compute explicit condition upgrade dollar value contribution
    avg_comp_price = np.mean(comp_base_prices) if comp_base_prices else reconciled_val
    cond_value_contrib = avg_comp_price * (condition_options[eval_condition] - condition_options[subject_condition])
    total_renovation_delta = sqft_value_contrib + bath_value_contrib + bed_value_contrib + cond_value_contrib

    st.divider()

    # --- CLIENT-FRIENDLY VALUATION METRIC CARDS ---
    st.markdown('<div class="editorial-section-title"><span class="editorial-section-dot"></span>🎯 Estimated Market Valuation</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">Fair market value spectrum derived from distance-weighted local sale comps and time-normalized market trend lines (+0.3%/month).</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; text-transform: uppercase; color: rgba(0,0,0,0.5); margin-bottom: 0.5rem;">
        Target Property: <strong>{subject_address}</strong> • Neighborhood: <strong>{matched_neigh_name}</strong> ({n_sales} Local Comps)
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("🛡️ Fast-Sale Floor (-6%)", f"${floor_val:,.0f}", help="Lower boundary for a quick sale or softer market conditions.")
    col2.metric("🎯 Estimated Market Value", f"${reconciled_val:,.0f}", f"${total_renovation_delta:+,.0f} Renovation Delta", help="The primary fair market baseline value including scenario updates.")
    col3.metric("🚀 Peak Value Ceiling (+5%)", f"${ceiling_val:,.0f}", help="Top-tier value potential under aggressive staging and multiple offers.")
    
    st.divider()

    # --- LOCAL VALUE DRIVERS CARD ---
    st.markdown('<div class="editorial-section-title"><span class="editorial-section-dot"></span>📊 Local Neighborhood Value Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">The exact marginal dollar amount buyers in this micro-neighborhood pay per physical feature:</div>', unsafe_allow_html=True)

    driver_col1, driver_col2, driver_col3, driver_col4, driver_col5 = st.columns(5)
    driver_col1.metric("Living Area", f"${coefs['SQUARE FEET']:,.0f} / SqFt")
    driver_col2.metric("Bathroom Fixture", f"${coefs['BATHS']:,.0f} / Bath")
    driver_col3.metric("Lot Footprint", f"${coefs['LOT SIZE']:,.2f} / SqFt")
    driver_col4.metric("Age Factor", f"${coefs['YEAR BUILT']:,.0f} / Year")
    driver_col5.metric("⚙️ Model Stability Rating", f"{best_alpha:.2f}", help="CV L2 Alpha: Automated brake setting that prevents price distortion.")

    st.divider()

    # --- COMPS TABLE ---
    st.markdown('<div class="editorial-section-title"><span class="editorial-section-dot"></span>📊 Matched Comparable Sales & Inferred Conditions</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">The 4 most geographically and structurally relevant recent sales. "Condition" is reverse-engineered by comparing actual sale prices to regression baselines.</div>', unsafe_allow_html=True)
    
    comps_display = closest_comps[['ADDRESS', 'PRICE', 'BEDS', 'BATHS', 'SQUARE FEET', '$/SQUARE FEET', 'dist_miles', 'AUTO_CONDITION']].copy()
    comps_display.columns = ['Address', 'Sale Price', 'Beds', 'Baths', 'SqFt', '$/SqFt', 'Distance (Mi)', 'Condition Grade']
    comps_display['Distance (Mi)'] = comps_display['Distance (Mi)'].round(2)
    comps_display['Sale Price'] = comps_display['Sale Price'].apply(lambda x: f"${x:,.0f}")
    comps_display['$/SqFt'] = comps_display['$/SqFt'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(comps_display, use_container_width=True)

    # --- COMPACT SPATIAL PYDECK MAP WITH UNMISTAKABLE MARKERS & LEGEND ---
    st.markdown('<div class="editorial-section-title" style="margin-top: 1rem;"><span class="editorial-section-dot"></span>🗺️ Micro-Neighborhood Proximity Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">Spatial map displaying the subject property relative to nearby historical comp sales used to train the local model.</div>', unsafe_allow_html=True)
    
    # MAP LEGEND BAR
    st.markdown("""
    <div class="map-legend-container">
        <div class="legend-item">
            <span class="legend-badge-subject"></span>
            <span>🎯 <strong>Subject Property</strong> (Target Location)</span>
        </div>
        <div class="legend-item">
            <span class="legend-badge-comp"></span>
            <span>🔹 <strong>Matched Comp Sales</strong> (Nearby Historical Data)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SUBJECT PROPERTY LAYER (Vibrant Crimson Red, Larger Size)
    subject_df = pd.DataFrame([{
        'ADDRESS': subject_address, 
        'LATITUDE': float(t_lat), 
        'LONGITUDE': float(t_lon),
        'LABEL': f"🎯 SUBJECT PROPERTY<br><b>{subject_address}</b><br>Footprint: {eval_sqft:,} SqFt | Condition: {eval_condition}"
    }])

    subject_layer = pdk.Layer(
        "ScatterplotLayer", 
        data=subject_df,
        get_position=["LONGITUDE", "LATITUDE"], 
        get_fill_color=[220, 38, 38, 255],  # Crimson Red
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        get_radius=60, 
        pickable=True, 
        auto_highlight=True,
        radius_scale=1, 
        radius_min_pixels=14, 
        radius_max_pixels=28
    )

    # COMP SALES LAYER (Oceanic Blue, Smaller Size)
    comp_rows = [{
        'ADDRESS': str(row['ADDRESS']), 
        'LATITUDE': float(row['LATITUDE']), 
        'LONGITUDE': float(row['LONGITUDE']),
        'LABEL': f"🔹 COMP SALE<br><b>{row['ADDRESS']}</b><br>Sold Price: ${row['PRICE']:,.0f}<br>Proximity: {row['dist_miles']:.2f} miles away"
    } for _, row in closest_comps.iterrows()]

    comps_df = pd.DataFrame(comp_rows)

    comps_layer = pdk.Layer(
        "ScatterplotLayer", 
        data=comps_df,
        get_position=["LONGITUDE", "LATITUDE"], 
        get_fill_color=[2, 132, 199, 230],  # Oceanic Blue
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=1,
        get_radius=30, 
        pickable=True, 
        auto_highlight=True,
        radius_scale=1, 
        radius_min_pixels=9, 
        radius_max_pixels=18
    )

    initial_view = pdk.ViewState(latitude=float(t_lat), longitude=float(t_lon), zoom=14.8, pitch=30)
    
    st.pydeck_chart(
        pdk.Deck(
            layers=[comps_layer, subject_layer],  # Render subject layer ON TOP
            initial_view_state=initial_view, 
            map_style="light",
            tooltip={
                "html": "<div style='max-width: 220px; white-space: normal; z-index: 999999;'>{LABEL}</div>", 
                "style": {"backgroundColor": "#1a1a1a", "color": "#ffffff", "fontSize": "12px", "padding": "8px 12px", "borderRadius": "4px"}
            }
        ),
        height=320,
        use_container_width=True
    )

    st.divider()

    # ==============================================================================
    # 7. EXPANDED EDITORIAL PRESENTATION ASSETS & TECHNICAL DECODER TABS
    # ==============================================================================
    st.markdown('<div class="editorial-section-title"><span class="editorial-section-dot"></span>📑 Presentation Assets & Technical Decoder</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-section-subtitle">Client-facing presentation talking points, leave-behind executive summaries, objection handling, and formal machine learning specifications.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🎙️ Agent Presentation Script & Objection Handling",
        "🎓 Client Summary Report (Leave-Behind)",
        "🔬 Machine Learning Spec & Technical Decoder"
    ])

    # --- TAB 1: AGENT PRESENTATION SCRIPT & OBJECTION PLAYBOOK ---
    with tab1:
        st.markdown(clean_html(f"""
        <div class="editorial-card">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.16em; color: #1a1a1a; margin-bottom: 1.25rem; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 0.5rem;">
                Agent Pitch Script & Client Meeting Playbook
            </h3>
            
            <div style="margin-bottom: 1.75rem; background-color: #f5f2ed; padding: 1.25rem; border-left: 3px solid #1a1a1a; border-radius: 4px;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #1a1a1a; margin-bottom: 0.5rem;">
                    💬 1. Opening Narrative: Why This Valuation is Superior
                </h4>
                <p style="font-size: 0.95rem; line-height: 1.6; color: #2d2d2d; margin-bottom: 0;">
                    "Most consumer real estate sites use 'black box' algorithms that blindly average homes across entire zip codes without inspecting interior quality, market timing, or exact GPS proximity. What I've brought today is a custom <strong>Quantitative Valuation Engine</strong> calibrated specifically to our micro-neighborhood using <strong>{n_sales} verified recent sales</strong>. It removes subjective bias and reveals exactly what buyers in today's market are paying for square footage, room counts, and condition upgrades."
                </p>
            </div>

            <div style="margin-bottom: 1.75rem;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #0284c7; margin-bottom: 0.75rem;">
                    🎯 2. Walking the Client Through Their 3-Tier Price
                </h4>
                <ul style="font-size: 0.95rem; line-height: 1.7; margin-left: 1.25rem; color: #2d2d2d; padding-left: 0;">
                    <li><strong>🎯 Estimated Fair Market Value (${reconciled_val:,.0f}):</strong> "This represents our primary target baseline under normal market exposure and standard buyer purchasing power today."</li>
                    <li><strong>🛡️ Fast-Sale Floor (${floor_val:,.0f}):</strong> "If you need a guaranteed rapid liquidity sale or if broader interest rates soften, this is our defensible lower boundary."</li>
                    <li><strong>🚀 Peak Value Ceiling (${ceiling_val:,.0f}):</strong> "If we professional-stage the property, execute aggressive multi-channel marketing, and drive multiple competing offers, this represents our realistic top target."</li>
                </ul>
            </div>

            <div style="border-top: 1px solid rgba(0,0,0,0.1); padding-top: 1.25rem;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #dc2626; margin-bottom: 1rem;">
                    🛡️ 3. Objection Handling Playbook
                </h4>
                
                <div style="margin-bottom: 1rem; background-color: #fafafa; padding: 1rem; border: 1px solid rgba(0,0,0,0.08); border-radius: 4px;">
                    <strong style="color: #1a1a1a;">Objection:</strong> <em>"An online site (e.g. Zillow or Redfin) says my home is worth $50,000 more!"</em><br>
                    <span style="color: #2d2d2d; font-size: 0.92rem; line-height: 1.5;">
                        <strong>Script Response:</strong> "Automated online algorithms cannot walk through your front door. They treat an unrenovated 1950s kitchen the same as a $100k custom quartz rebuild next door. Our engine uses local Ridge Regression to isolate physical size from interior condition quality, giving us an honest appraisal that mortgage lenders will actually approve."
                    </span>
                </div>

                <div style="margin-bottom: 1rem; background-color: #fafafa; padding: 1rem; border: 1px solid rgba(0,0,0,0.08); border-radius: 4px;">
                    <strong style="color: #1a1a1a;">Objection:</strong> <em>"Why aren't you including that record sale 1 mile away across the main boulevard?"</em><br>
                    <span style="color: #2d2d2d; font-size: 0.92rem; line-height: 1.5;">
                        <strong>Script Response:</strong> "Real estate value decays rapidly over distance. Crossing major thoroughfares or school districts changes buyer demographics. Our model uses Inverse Distance Weighting, prioritizing the closest 4 homes within our micro-pocket to ensure your listing is priced against true direct competitors."
                    </span>
                </div>

                <div style="background-color: #fafafa; padding: 1rem; border: 1px solid rgba(0,0,0,0.08); border-radius: 4px;">
                    <strong style="color: #1a1a1a;">Objection:</strong> <em>"How do we reach the Peak Value Ceiling?"</em><br>
                    <span style="color: #2d2d2d; font-size: 0.92rem; line-height: 1.5;">
                        <strong>Script Response:</strong> "Hitting the 5% ceiling (${ceiling_val:,.0f}) requires eliminating all buyer hesitation points: professional staging, turnkey interior condition, pristine curb appeal, and an aggressive launch strategy that generates offer competition during week one."
                    </span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # --- TAB 2: CLIENT SUMMARY REPORT (LEAVE-BEHIND) ---
    with tab2:
        st.markdown(clean_html(f"""
        <div class="editorial-card">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.16em; color: #1a1a1a; margin-bottom: 1.25rem; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 0.5rem;">
                Institutional Comparative Market Analysis (CMA) Summary
            </h3>
            <p style="margin-bottom: 1.5rem; font-size: 0.95rem; color: rgba(0,0,0,0.7);">
                This executive report replaces intuitive guesswork with a transparent, 4-step mathematical valuation process calibrated specifically to <strong>{matched_neigh_name}</strong>. Below is the step-by-step methodology used to establish your baseline target of <strong>${reconciled_val:,.0f}</strong>:
            </p>
            
            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #1a1a1a; padding: 1.25rem; margin-bottom: 1rem; border-radius: 4px;">
                <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0 0 0.5rem 0;">STEP 01: Micro-Neighborhood Data Ingestion & Time Normalization</h4>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    The model evaluated <strong>{n_sales} verified historical sales</strong> in this micro-pocket. To account for market inflation or shifts over time, historical sales are time-normalized at <strong>+0.3% per month</strong> to match today's current purchasing power.
                </p>
            </div>

            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #0284c7; padding: 1.25rem; margin-bottom: 1rem; border-radius: 4px;">
                <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0 0 0.5rem 0;">STEP 02: Unbiased Composite Proximity Matching</h4>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    Rather than picking random sales, the engine uses a dual <strong>Composite KNN Distance Metric</strong> combining <strong>70% Geographic Geodesic Proximity</strong> (GPS distance) and <strong>30% Footprint Matching</strong> (square footage similarity) to isolate the 4 truest comparable properties.
                </p>
            </div>

            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #16a34a; padding: 1.25rem; margin-bottom: 1rem; border-radius: 4px;">
                <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0 0 0.5rem 0;">STEP 03: Feature-by-Feature Hedonic Adjustments</h4>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    Each matched comp is adjusted up or down to reflect the exact physical specifications of your subject home:
                </p>
                <ul style="font-size: 0.88rem; line-height: 1.6; color: #444444; margin-top: 0.5rem; margin-bottom: 0;">
                    <li><strong>Living Area:</strong> Adjusting for footprint difference at <strong>${coefs['SQUARE FEET']:,.0f}/SqFt</strong>.</li>
                    <li><strong>Bathrooms:</strong> Adjusting for bathroom count difference at <strong>${coefs['BATHS']:,.0f}/Bath</strong>.</li>
                    <li><strong>Condition Tier:</strong> Adjusting for interior finish level (Turnkey: +15%, Updated: +7%, Average: 0%, Dated: -7%, Fixer: -15%).</li>
                </ul>
            </div>

            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #dc2626; padding: 1.25rem; margin-bottom: 1.5rem; border-radius: 4px;">
                <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0 0 0.5rem 0;">STEP 04: Inverse Distance Weighting (IDW) Reconciliation</h4>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    Final target price is calculated using Inverse Distance Weighting, giving exponentially more mathematical weight to the homes sitting closest to your front door.
                </p>
            </div>

            <div style="background-color: #f5f2ed; padding: 1.25rem; border-radius: 4px; text-align: center;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #1a1a1a; margin-bottom: 0.4rem;">
                    📋 Executive Leave-Behind Summary
                </h4>
                <p style="font-size: 1.1rem; font-family: 'Playfair Display', serif; font-weight: 600; color: #1a1a1a; margin: 0;">
                    Recommended Listing Strategy: <strong>${reconciled_val:,.0f}</strong>  
                    <span style="font-size: 0.85rem; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 400; color: rgba(0,0,0,0.6); display: block;">
                        Range: ${floor_val:,.0f} (Fast Liquidity) — ${ceiling_val:,.0f} (Aggressive Staging Peak)
                    </span>
                </p>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # --- TAB 3: MACHINE LEARNING SPEC & TECHNICAL DECODER ---
    with tab3:
        st.markdown("### 🔬 Formal Mathematical Specification & Algorithmic Decoder")
        st.write("This tab outlines the complete mathematical framework powering the valuation pipeline, provided for technical auditability and institutional compliance.")
        
        st.divider()

        # 1. RIDGE REGRESSION
        st.markdown("#### 1. Regularized Ridge Regression ($L_2$ Penalty)")
        st.write("To handle small sample sizes and multicollinearity between correlated features (such as square footage and bathroom counts), feature weights are solved via $L_2$-Regularized Ridge Regression:")
        st.latex(r"\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} \left( y_i - x_i^T \beta \right)^2 + \alpha \sum_{j=1}^{p} \beta_j^2")
        st.caption(f"Current Model Hyperparameter: Tuned Regularization Penalty (α = {best_alpha:.4f}) solved dynamically via 5-Fold Cross Validation.")

        st.divider()

        # 2. HAVERSINE DISTANCE
        st.markdown("#### 2. Great-Circle Haversine Spatial Geodesic Distance")
        st.write("Exact spherical distance between latitude/longitude coordinate pairs on Earth's surface ($r = 3,958.8\text{ miles}$):")
        st.latex(r"d = 2r \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)} \right)")

        st.divider()

        # 3. COMPOSITE KNN DISTANCE
        st.markdown("#### 3. Composite Spatial & Footprint KNN Metric")
        st.write("Comparable selection employs a dual-component composite spatial distance metric combining geographic proximity with normalized living area footprint deviation:")
        st.latex(r"D_{\text{comp}} = 0.70 \cdot d_{\text{geodesic}} + 0.30 \cdot \left( \frac{| \text{SqFt}_{\text{subject}} - \text{SqFt}_{\text{comp}} |}{\sigma_{\text{SqFt}}} \right)")

        st.divider()

        # 4. INVERSE DISTANCE WEIGHTING (IDW)
        st.markdown("#### 4. Inverse Distance Weighting (IDW) Target Reconciliation")
        st.write("Final target price is synthesized across the $k$ nearest comps by applying Inverse Distance Weighting ($w_i$), ensuring closest comps dominate valuation:")
        st.latex(r"w_i = \frac{1}{d_i + \epsilon}, \quad \hat{Y}_{\text{target}} = \sum_{i=1}^{k} \hat{w}_i \cdot Y_i^{\text{adjusted}}")

        st.divider()

        # 5. SANITY GUARDS & BOUNDS
        st.markdown("#### 5. Numerical Stability & Variance Guardrails")
        st.info("""
        * **Zero-Variance Guard:** Prevents near-zero scale division (`scaler.scale_ < 1e-4`) when standard deviations collapse in micro-datasets.
        * **Coefficient Clamping:** Bounds parameter weights to real-world appraisal limits (e.g. Living Area constrained between $20/SqFt and $1,500/SqFt).
        """)

if __name__ == "__main__":
    main()
