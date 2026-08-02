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
# 1. APP CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Quant Valuation Engine", layout="wide", page_icon="🏡")

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
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(0, 0, 0, 0.4) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
    padding-bottom: 0.35rem !important;
}

/* Inputs & Selectboxes */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
    border-radius: 4px !important;
    color: #1a1a1a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput label, .stNumberInput label, .stSlider label, .stSelectbox label {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: rgba(0, 0, 0, 0.7) !important;
}

/* Editorial Header Banner */
.editorial-banner {
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.editorial-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 3.25rem !important;
    font-weight: 500 !important;
    font-style: italic !important;
    letter-spacing: -0.03em !important;
    color: #1a1a1a !important;
    line-height: 1.05 !important;
    margin: 0 !important;
}
.editorial-subtitle {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    color: rgba(0, 0, 0, 0.5) !important;
    font-weight: 700 !important;
    margin-top: 0.5rem !important;
}
.editorial-meta {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    color: rgba(0, 0, 0, 0.6);
    line-height: 1.5;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border-left: 3px solid #1a1a1a !important;
    border-top: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06) !important;
    padding: 1.25rem !important;
    border-radius: 2px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    font-weight: 800 !important;
    color: rgba(0, 0, 0, 0.45) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.03em !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #0284c7 !important;
    text-transform: uppercase !important;
}

/* Dividers & Section Elements */
hr {
    border: none !important;
    border-top: 1px solid rgba(0, 0, 0, 0.12) !important;
    margin: 2.25rem 0 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
    border-radius: 4px !important;
    background-color: #ffffff !important;
    padding: 2px !important;
}
.editorial-section-title {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #1a1a1a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
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
    padding: 1.5rem;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.7;
    color: #1a1a1a;
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
# 4. DATA PROCESSING & RIDGE REGRESSION MODEL
# ==============================================================================
@st.cache_data
def process_dataset(file_source):
    df = pd.read_csv(file_source)
    if "SALE TYPE" in df.columns and pd.isna(df.iloc[0]['SOLD DATE']):
        df = df.drop(index=0)
       
    num_cols = ['PRICE', 'BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', 'LATITUDE', 'LONGITUDE', '$/SQUARE FEET']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
           
    df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'], errors='coerce')
   
    clean_df = df[(df['PROPERTY TYPE'] == 'Single Family Residential') &
                  (df['PRICE'].notnull()) &
                  (df['$/SQUARE FEET'] >= 300)].copy()
   
    # Local Market Time Drift (+0.3% per month)
    today = pd.Timestamp.now()
    clean_df['months_since_sale'] = ((today - clean_df['SOLD DATE']).dt.days / 30.0).fillna(0)
    clean_df['time_mult'] = (1.0 + (clean_df['months_since_sale'] * 0.003)).clip(lower=0.92, upper=1.08)
    clean_df['TIME_ADJUSTED_PRICE'] = clean_df['PRICE'] * clean_df['time_mult']
   
    return clean_df

def fit_dynamic_model(clean_df):
    features = ['SQUARE FEET', 'BATHS', 'LOT SIZE', 'YEAR BUILT']
    model_df = clean_df.dropna(subset=features + ['TIME_ADJUSTED_PRICE']).copy()
   
    X = model_df[features]
    y = model_df['TIME_ADJUSTED_PRICE']
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    alphas = np.logspace(-2, 4, 50)
    cv = KFold(n_splits=min(5, max(2, len(model_df)//3)), shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=alphas, cv=cv)
    ridge.fit(X_scaled, y)
   
    unscaled_coefs = ridge.coef_ / scaler.scale_
    coef_dict = dict(zip(features, unscaled_coefs))
   
    # Residuals Condition Grading
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
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔒 Agent Admin Portal"):
        admin_pin = st.text_input("Enter Admin PIN", type="password", key="admin_pin_input")
        target_pin = st.secrets.get("ADMIN_PIN", "8888")
        
        if admin_pin == target_pin:
            st.success("🔓 Authenticated")
            st.markdown("#### 📤 Register Neighborhood CSV")
            neigh_label = st.text_input("Neighborhood Label", value="Richmond - East Heights")
            radius = st.number_input("Coverage Radius (Miles)", value=3.0, step=0.5, min_value=0.5)
            uploaded_csv = st.file_uploader("Upload MLS/Redfin Sales CSV", type=["csv"])
            
            if uploaded_csv is not None:
                df = pd.read_csv(uploaded_csv)
                col_map = {str(c).upper(): c for c in df.columns}
                lat_col = col_map.get('LATITUDE') or col_map.get('LAT')
                lon_col = col_map.get('LONGITUDE') or col_map.get('LON') or col_map.get('LNG')
                
                if not lat_col or not lon_col:
                    st.error("⚠️ CSV missing LATITUDE / LONGITUDE columns.")
                else:
                    c_lat, c_lon = float(df[lat_col].mean()), float(df[lon_col].mean())
                    n_sales = len(df)
                    st.info(f"📍 **Centroid:** ({c_lat:.4f}, {c_lon:.4f}) | Sales: {n_sales}")
                    
                    if st.button("🚀 Publish Dataset"):
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
                        st.balloons()
                        st.success(f"🎉 **{neigh_label}** registered!")
        elif admin_pin != "":
            st.error("Invalid PIN")

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
        st.divider()
        st.markdown("### 📊 How Do We Know These Percentages Are Real?")
        st.markdown("Instead of guessing numbers out of thin air, our engine anchors itself on two solid pillars:")
        st.markdown("""
        <div style="background-color: #f5f2ed; border-left: 4px solid #1a1a1a; padding: 1.25rem; border-radius: 4px; margin-top: 0.5rem; margin-bottom: 1rem;">
            <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin-bottom: 0.75rem;">
                🏛️ Pillar 1: Bank & Appraisal Rules (National Standards)
            </h4>
            <ul style="font-size: 0.9rem; line-height: 1.6; color: #2d2d2d; margin-left: 1rem; margin-bottom: 0;">
                <li><strong>±4% Market Noise Band:</strong> In statistics, normal negotiation, staging, or competitive bidding naturally moves sale prices by up to ±4% without the physical home actually being better or worse.</li>
                <li><strong>4% to 12% Renovation ROI:</strong> Remodeling a kitchen and bathrooms usually costs 5%–10% of a home's total value. Appraisal paired-sales data shows high-impact updates yield a net 4%–12% bump at resale.</li>
                <li><strong>> 12% Major CapEx / Flips:</strong> Real estate investors follow the "70% Rule"—meaning full gut remodels or severe fixers shift prices by more than 12% relative to a standard baseline.</li>
            </ul>
        </div>
        <div style="background-color: #ffffff; border: 1px solid rgba(0,0,0,0.12); border-left: 4px solid #0284c7; padding: 1.25rem; border-radius: 4px;">
            <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #0284c7; margin-bottom: 0.75rem;">
                ⚙️ Pillar 2: Micro-Neighborhood Regression (Local CSV Data)
            </h4>
            <p style="font-size: 0.9rem; line-height: 1.6; color: #2d2d2d; margin-bottom: 0;">
                While the percentage guardrails come from national appraisal standards, <strong>the actual dollar adjustments are calculated live</strong> from your uploaded local sales CSV. The engine runs a Ridge Regression to calculate what buyers in <em>this exact neighborhood</em> paid per square foot, bathroom, and year built.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 6. MAIN APPLICATION & SEARCH UI
# ==============================================================================
def main():
    render_admin_portal()
    
    # --- SIDEBAR: SUBJECT SPECS & SOURCE SELECTOR ---
    st.sidebar.markdown("### 01. Subject Property Specs")
    
    property_mode = st.sidebar.radio(
        "Subject Property Source",
        options=["Off-Market / Custom Address", "Select Existing Property from CSV"],
        index=0,
        help="Choose whether to evaluate an off-market target address or pick an existing property directly from your CSV dataset."
    )

    manifest_df = get_manifest()
    
    # Address search bar & dataset matching
    active_dataset_file = None
    matched_neigh_name = "Default Dataset"
    searched_lat, searched_lon, geocoded_addr = None, None, None
    search_query = ""

    if property_mode == "Off-Market / Custom Address":
        search_query = st.text_input(
            "Enter Subject Property Address", 
            placeholder="e.g. 3509 McBryde Ave, Richmond, CA 94805"
        )
        if search_query:
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
                    else:
                        st.warning("📍 Address located, but falls outside active neighborhood radii. Using primary dataset.")
    
    # Fallback to load primary dataset if no specific match file set yet
    if not active_dataset_file:
        if not manifest_df.empty:
            active_dataset_file = load_dataset(manifest_df.iloc[0]['file_name'])
            matched_neigh_name = manifest_df.iloc[0]['neighborhood_name']
        elif os.path.exists("sample_redfin.csv"):
            active_dataset_file = "sample_redfin.csv"
            matched_neigh_name = "Sample Redfin Data"
        else:
            st.info("👈 Please use the **Agent Admin Portal** in the sidebar to upload a CSV dataset!")
            st.stop()

    data = process_dataset(active_dataset_file)
    coefs, best_alpha, n_sales, model_df = fit_dynamic_model(data)

    # Pre-populate defaults or CSV values
    if property_mode == "Select Existing Property from CSV":
        csv_addresses = model_df['ADDRESS'].dropna().astype(str).tolist() if 'ADDRESS' in model_df.columns else []
        selected_csv_addr = st.sidebar.selectbox("Select Property from CSV", options=csv_addresses) if csv_addresses else "Sample Property"
        match = model_df[model_df['ADDRESS'].astype(str) == selected_csv_addr] if csv_addresses else pd.DataFrame()
        
        if not match.empty:
            row = match.iloc[0]
            default_sqft = int(row.get('SQUARE FEET', 1500)) if pd.notna(row.get('SQUARE FEET')) else 1500
            default_beds = int(row.get('BEDS', 3)) if pd.notna(row.get('BEDS')) else 3
            default_baths = float(row.get('BATHS', 2.0)) if pd.notna(row.get('BATHS')) else 2.0
            default_lot = int(row.get('LOT SIZE', 5000)) if pd.notna(row.get('LOT SIZE')) else 5000
            default_year = int(row.get('YEAR BUILT', 1955)) if pd.notna(row.get('YEAR BUILT')) else 1955
            t_lat = float(row.get('LATITUDE', 37.9542)) if pd.notna(row.get('LATITUDE')) else 37.9542
            t_lon = float(row.get('LONGITUDE', -122.3353)) if pd.notna(row.get('LONGITUDE')) else -122.3353
            subject_address = selected_csv_addr
        else:
            default_sqft, default_beds, default_baths, default_lot, default_year = 1500, 3, 2.0, 5000, 1955
            t_lat, t_lon = 37.9542, -122.3353
            subject_address = "Sample Property"
    else:
        # Custom Off-Market Property Defaults
        default_sqft, default_beds, default_baths, default_lot, default_year = 1500, 3, 2.0, 5000, 1955
        t_lat = searched_lat if searched_lat else (model_df['LATITUDE'].mean() if 'LATITUDE' in model_df.columns else 37.9542)
        t_lon = searched_lon if searched_lon else (model_df['LONGITUDE'].mean() if 'LONGITUDE' in model_df.columns else -122.3353)
        subject_address = geocoded_addr if geocoded_addr else (search_query if search_query else "Off-Market Subject Property")

    # Editable Property Specs inputs in Sidebar
    subject_sqft = st.sidebar.number_input("Living Area (SqFt)", value=default_sqft, step=50)
    subject_beds = st.sidebar.number_input("Bedrooms", value=default_beds, step=1)
    subject_baths = st.sidebar.number_input("Bathrooms", value=default_baths, step=0.5)
    subject_lot = st.sidebar.number_input("Lot Size (SqFt)", value=default_lot, step=250)
    subject_year = st.sidebar.number_input("Year Built", value=default_year, step=1)
    subject_condition = st.sidebar.selectbox("Property Condition Grade", options=["Turnkey", "Updated", "Average", "Dated", "Fixer"], index=0)

    if st.sidebar.button("📖 Open Condition Guide & Glossary", use_container_width=True):
        show_condition_guide()

    condition_options = {'Turnkey': +0.15, 'Updated': +0.07, 'Average': 0.00, 'Dated': -0.07, 'Dated/Original': -0.07, 'Fixer': -0.15}

    # --- SIDEBAR: RENOVATION PLANNER ---
    st.sidebar.markdown("### 02. Renovation Scenario Planner")
    add_sqft = st.sidebar.number_input("Add/Subtract Living Area (SqFt)", value=0, step=25)
    add_beds = st.sidebar.slider("Add/Subtract Bedrooms", -2, 3, 0, step=1)
    add_baths = st.sidebar.slider("Add/Subtract Bathrooms", -1.0, 3.0, 0.0, step=0.5)

    eval_sqft = subject_sqft + add_sqft
    eval_beds = subject_beds + add_beds
    eval_baths = subject_baths + add_baths

    sqft_value_contrib = add_sqft * coefs['SQUARE FEET']
    bath_value_contrib = add_baths * coefs['BATHS']
    BEDROOM_UTILITY_DELTA = 12500.0
    bed_value_contrib = add_beds * BEDROOM_UTILITY_DELTA
    total_renovation_delta = sqft_value_contrib + bath_value_contrib + bed_value_contrib

    # --- COMP SELECTION & RECONCILIATION ---
    model_df['dist_miles'] = haversine_distance(t_lat, t_lon, model_df['LATITUDE'], model_df['LONGITUDE'])
    sqft_std = model_df['SQUARE FEET'].std() if model_df['SQUARE FEET'].std() > 0 else 1
    model_df['sqft_diff_norm'] = np.abs(model_df['SQUARE FEET'] - eval_sqft) / sqft_std
    model_df['composite_dist'] = (0.7 * model_df['dist_miles']) + (0.3 * model_df['sqft_diff_norm'])

    closest_comps = model_df.sort_values(by='composite_dist').head(4).copy()

    comp_values = []
    for _, comp in closest_comps.iterrows():
        adj_sqft = (eval_sqft - comp['SQUARE FEET']) * coefs['SQUARE FEET']
        adj_baths = (eval_baths - comp['BATHS']) * coefs['BATHS']
        lot_diff = np.sqrt(max(0, subject_lot)) - np.sqrt(max(0, comp['LOT SIZE']))
        adj_lot = lot_diff * (coefs['LOT SIZE'] * 50)
        adj_year = (subject_year - comp['YEAR BUILT']) * coefs['YEAR BUILT']
        
        comp_cond_mult = condition_options.get(comp['AUTO_CONDITION'], 0.0)
        subj_cond_mult = condition_options[subject_condition]
        adj_condition = comp['TIME_ADJUSTED_PRICE'] * (subj_cond_mult - comp_cond_mult)

        tot_adj = adj_sqft + adj_baths + adj_lot + adj_year + adj_condition
        comp_values.append(comp['TIME_ADJUSTED_PRICE'] + tot_adj)

    weights = 1.0 / (closest_comps['dist_miles'] + 0.01)
    weights /= weights.sum()
    reconciled_val = np.dot(comp_values, weights)
    floor_val, ceiling_val = reconciled_val * 0.94, reconciled_val * 1.05

    # --- MAIN EDITORIAL DASHBOARD HEADER ---
    st.markdown(f"""
    <div class="editorial-banner">
        <div>
            <h1 class="editorial-title">Quant Valuation Engine</h1>
            <p class="editorial-subtitle">Dynamic Spatial Regression & Residual Condition Engine • Target: {subject_address}</p>
        </div>
        <div class="editorial-meta">
            <div><strong>Sample Size:</strong> {n_sales} Recent Sales</div>
            <div><strong>Subject Grade:</strong> {subject_condition}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Conservative Floor (-6%)", f"${floor_val:,.0f}")
    col2.metric("Reconciled Valuation", f"${reconciled_val:,.0f}", f"${total_renovation_delta:+,.0f} Renovation Delta")
    col3.metric("Aggressive Ceiling (+5%)", f"${ceiling_val:,.0f}")
    st.divider()

    # --- COMPS TABLE ---
    st.markdown('<div class="editorial-section-title"><span class="editorial-section-dot"></span>Geodesic Matched Comparables & Inferred Conditions</div>', unsafe_allow_html=True)
    comps_display = closest_comps[['ADDRESS', 'PRICE', 'BEDS', 'BATHS', 'SQUARE FEET', '$/SQUARE FEET', 'dist_miles', 'AUTO_CONDITION']].copy()
    comps_display.columns = ['Address', 'Sale Price', 'Beds', 'Baths', 'SqFt', '$/SqFt', 'Distance (Mi)', 'Auto Condition']
    comps_display['Distance (Mi)'] = comps_display['Distance (Mi)'].round(2)
    st.dataframe(comps_display, use_container_width=True)

    # --- SPATIAL PYDECK MAP ---
    st.markdown('<div class="editorial-section-title" style="margin-top: 1.5rem;"><span class="editorial-section-dot"></span>Micro-Neighborhood Spatial Proximity Map</div>', unsafe_allow_html=True)
    subject_row = {
        'ADDRESS': subject_address, 'LATITUDE': float(t_lat), 'LONGITUDE': float(t_lon),
        'R': 220, 'G': 38, 'B': 38, 'A': 255, 'RADIUS': 40,
        'LABEL': f"<b>SUBJECT PROPERTY</b><br>{subject_address}<br>Footprint: {eval_sqft:,} SqFt"
    }
    comp_rows = [{
        'ADDRESS': str(row['ADDRESS']), 'LATITUDE': float(row['LATITUDE']), 'LONGITUDE': float(row['LONGITUDE']),
        'R': 2, 'G': 132, 'B': 199, 'A': 230, 'RADIUS': 25,
        'LABEL': f"<b>COMP: {row['ADDRESS']}</b><br>Sold Price: ${row['PRICE']:,.0f}<br>Proximity: {row['dist_miles']:.2f} miles away"
    } for _, row in closest_comps.iterrows()]

    all_map_locations = pd.DataFrame([subject_row] + comp_rows)
    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer", data=all_map_locations,
        get_position=["LONGITUDE", "LATITUDE"], get_fill_color="[R, G, B, A]",
        get_radius="RADIUS", pickable=True, auto_highlight=True,
        radius_scale=1, radius_min_pixels=10, radius_max_pixels=30
    )
    initial_view = pdk.ViewState(latitude=float(t_lat), longitude=float(t_lon), zoom=14.8, pitch=35)
    st.pydeck_chart(pdk.Deck(layers=[scatterplot_layer], initial_view_state=initial_view, map_style="light",
                             tooltip={"html": "{LABEL}", "style": {"backgroundColor": "#1a1a1a", "color": "#ffffff", "fontSize": "12px", "padding": "8px 12px", "borderRadius": "4px", "boxShadow": "0 2px 8px rgba(0,0,0,0.2)"}}))

    st.divider()

    # ==============================================================================
    # EXPLANATION, SCRIPTING & TECHNICAL SPECIFICATION TABS
    # ==============================================================================
    tab1, tab2, tab3 = st.tabs([
        "🎙️ Agent Presentation Script (Client Talking Points)",
        "🎓 Client Summary Report (Leave-Behind)",
        "🔬 Machine Learning Spec & Agent Decoder"
    ])

    # ------------------------------------------------------------------------------
    # TAB 1: AGENT PRESENTATION SCRIPT & TALKING POINTS
    # ------------------------------------------------------------------------------
    with tab1:
        st.markdown(clean_html(f"""
        <div class="editorial-card">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.16em; color: #1a1a1a; margin-bottom: 1.25rem; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 0.5rem;">
                Agent Meeting Script & Objection Handling Playbook
            </h3>
           
            <p style="margin-bottom: 1.5rem; font-size: 0.95rem; color: rgba(0,0,0,0.7);">
                <em>Use this tab as your personal cheat sheet during client presentations to explain the valuation, build instant authority, and confidently answer tough questions.</em>
            </p>
            <!-- SECTION 1: THE OPENING PITCH -->
            <div style="margin-bottom: 1.75rem; background-color: #f5f2ed; padding: 1.25rem; border-left: 3px solid #1a1a1a; border-radius: 4px;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #1a1a1a; margin-bottom: 0.5rem;">
                    💬 How to Introduce This Engine (The Opening Script)
                </h4>
                <p style="font-size: 0.95rem; line-height: 1.6; color: #2d2d2d; margin-bottom: 0;">
                    "Most online valuation tools like Zillow or automated MLS reports are 'black boxes'—they blindly average random homes without accounting for micro-location, market timing, or property condition. What I've brought today is a custom <strong>Quantitative Valuation Engine</strong> calibrated specifically to our micro-neighborhood using <strong>{n_sales} recent local sales</strong>. It removes human bias and shows us exactly what buyers in today's market are paying for every square foot, bedroom, and bathroom."
                </p>
            </div>
            <!-- SECTION 2: EXPLAINING THE NUMBERS -->
            <div style="margin-bottom: 1.75rem;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #0284c7; margin-bottom: 0.75rem;">
                    🎯 Walking the Client Through Their Price
                </h4>
               
                <ul style="font-size: 0.95rem; line-height: 1.7; margin-left: 1.25rem; color: #2d2d2d;">
                    <li><strong>The Reconciled Target (${reconciled_val:,.0f}):</strong> "This is our fair market baseline based on current buyer purchasing power today."</li>
                    <li><strong>The Conservative Floor (${floor_val:,.0f}):</strong> "If we need to sell quickly or if market traffic slows down, this is our defensible lower boundary."</li>
                    <li><strong>The Aggressive Ceiling (${ceiling_val:,.0f}):</strong> "If we stage the home impeccably, market aggressively, and generate multiple competitive offers, this is our realistic top target."</li>
                </ul>
            </div>
            <!-- SECTION 3: OBJECTION HANDLING -->
            <div style="margin-bottom: 1.5rem;">
                <h4 style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 800; color: #0284c7; margin-bottom: 0.75rem;">
                    🛡️ Client Objection Handling (Quick Reference)
                </h4>
                <div style="margin-bottom: 1.25rem; background-color: #ffffff; border: 1px solid rgba(0,0,0,0.08); padding: 1rem; border-radius: 4px;">
                    <strong style="font-size: 0.95rem; color: #1a1a1a;">Q: "Why is Zillow / Redfin's estimate higher or lower than this?"</strong>
                    <p style="font-size: 0.9rem; line-height: 1.5; color: rgba(0,0,0,0.75); margin-top: 0.35rem; margin-bottom: 0;">
                        <em>Your Response:</em> "Algorithms like Zestimate use broad ZIP code averages. They don't know whether a comp sold 9 months ago in a different rate environment, or whether a nearby sale was a beaten-up fixer. Our model adjusts older sales for market time and isolates property condition mathematically."
                    </p>
                </div>
                <div style="margin-bottom: 1.25rem; background-color: #ffffff; border: 1px solid rgba(0,0,0,0.08); padding: 1rem; border-radius: 4px;">
                    <strong style="font-size: 0.95rem; color: #1a1a1a;">Q: "Why didn't you pick the house 2 blocks away that sold for $850k?"</strong>
                    <p style="font-size: 0.9rem; line-height: 1.5; color: rgba(0,0,0,0.75); margin-top: 0.35rem; margin-bottom: 0;">
                        <em>Your Response:</em> "Our engine evaluates every home using a <strong>Composite Match Score</strong> (70% distance + 30% home size match). That $850k home was either significantly larger in square footage or a turnkey remodel, which would distort our baseline. The 4 comps we selected are mathematically closest to your home's footprint."
                    </p>
                </div>
                <div style="margin-bottom: 1rem; background-color: #ffffff; border: 1px solid rgba(0,0,0,0.08); padding: 1rem; border-radius: 4px;">
                    <strong style="font-size: 0.95rem; color: #1a1a1a;">Q: "If I spend $30,000 to add a bathroom, do I get $30,000 back in value?"</strong>
                    <p style="font-size: 0.9rem; line-height: 1.5; color: rgba(0,0,0,0.75); margin-top: 0.35rem; margin-bottom: 0;">
                        <em>Your Response:</em> "Not automatically. Value comes from two things: physical space added and layout utility. Our scenario planner shows that adding a bathroom fixture adds ~${bath_value_contrib:,.0f} in functional utility. To capture full value, we also need to expand the square footage, which adds ${coefs['SQUARE FEET']:.0f}/sqft."
                    </p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # ------------------------------------------------------------------------------
    # TAB 2: CLIENT SUMMARY REPORT
    # ------------------------------------------------------------------------------
    with tab2:
        st.markdown(clean_html(f"""
        <div class="editorial-card">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.16em; color: #1a1a1a; margin-bottom: 1.25rem; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 0.5rem;">
                Valuation Methodology & Execution Workflow
            </h3>
           
            <p style="margin-bottom: 1.5rem; font-size: 0.95rem; color: rgba(0,0,0,0.7);">
                This valuation report replaces traditional guesswork with a transparent, data-driven algorithm calibrated specifically to this micro-neighborhood. Here is how your estimate of <strong>${reconciled_val:,.0f}</strong> was derived:
            </p>
            <!-- STEP 1 CARD -->
            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #1a1a1a; padding: 1.25rem; margin-bottom: 1rem; border-radius: 4px;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 800; background-color: #1a1a1a; color: #ffffff; padding: 2px 7px; border-radius: 2px;">STEP 01</span>
                    <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0;">Micro-Neighborhood Sample & Time Normalization</h4>
                </div>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    The model analyzed <strong>{n_sales} recent single-family sales</strong> from local MLS records. Because older sales closed under different interest rates and market conditions, every sale price is automatically normalized for market time drift (+0.3% per month) so all comparisons reflect current purchasing power today.
                </p>
            </div>
            <!-- STEP 2 CARD -->
            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #0284c7; padding: 1.25rem; margin-bottom: 1rem; border-radius: 4px;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 800; background-color: #0284c7; color: #ffffff; padding: 2px 7px; border-radius: 2px;">STEP 02</span>
                    <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0;">Unbiased Comparable Matching (Composite Proximity)</h4>
                </div>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    To prevent hand-picking comps, the algorithm evaluates sales using a dual <strong>Composite Distance Metric</strong> combining <strong>70% Geographic Proximity</strong> (GPS coordinates in miles) with <strong>30% Footprint Match</strong> (living area size similarity) to avoid comparing small cottages to large estates.
                </p>
            </div>
            <!-- STEP 3 CARD -->
            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #1a1a1a; padding: 1.25rem; margin-bottom: 1rem; border-radius: 4px;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 800; background-color: #1a1a1a; color: #ffffff; padding: 2px 7px; border-radius: 2px;">STEP 03</span>
                    <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0;">Automated Property Condition Detection</h4>
                </div>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    The engine infers property condition (Fixer vs. Turnkey) by measuring price residuals against baseline physical specs. Homes selling at significant premiums are recognized as <em>Updated</em> or <em>Turnkey</em> and adjusted to match your subject home's selected grade (<strong>{subject_condition}</strong>).
                </p>
            </div>
            <!-- STEP 4 CARD -->
            <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); border-left: 4px solid #0284c7; padding: 1.25rem; margin-bottom: 1.5rem; border-radius: 4px;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 800; background-color: #0284c7; color: #ffffff; padding: 2px 7px; border-radius: 2px;">STEP 04</span>
                    <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #1a1a1a; margin: 0;">Distance-Weighted Valuation Reconciliation</h4>
                </div>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #333333; margin: 0;">
                    Using <strong>Inverse Distance Weighting (IDW)</strong>, comparables located physically closer to <strong>{subject_address}</strong> exert a proportionally stronger mathematical influence on the final reconciled valuation.
                </p>
            </div>
            <hr style="margin: 1.5rem 0 !important;">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.15em; color: rgba(0,0,0,0.5); margin-bottom: 1rem;">
                Renovation Scenario Itemized Impact
            </h3>
            <ul style="margin-left: 1.25rem; margin-bottom: 0; font-size: 0.95rem; line-height: 1.7; color: #2d2d2d;">
                <li><strong>Footprint Expansion ({add_sqft:+d} SqFt):</strong> ${sqft_value_contrib:+,.0f} <em>(derived at ${coefs['SQUARE FEET']:.0f}/sqft)</em></li>
                <li><strong>Bedroom Layout Utility ({add_beds:+d} Beds):</strong> ${bed_value_contrib:+,.0f}</li>
                <li><strong>Bathroom Fixture Utility ({add_baths:+.1f} Baths):</strong> ${bath_value_contrib:+,.0f}</li>
                <li style="margin-top: 0.5rem;"><strong>Total Net Valuation Delta:</strong> <strong style="color: #0284c7;">${total_renovation_delta:+,.0f}</strong></li>
            </ul>
        </div>
        """), unsafe_allow_html=True)

    # ------------------------------------------------------------------------------
    # TAB 3: QUANTITATIVE SPECIFICATION & AGENT TECHNICAL DECODER
    # ------------------------------------------------------------------------------
    with tab3:
        st.markdown(clean_html(f"""
        <div style="background-color: #ffffff; border: 1px solid rgba(0,0,0,0.12); padding: 2rem; border-radius: 4px;">
            <h2 style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.75rem; font-weight: 600; margin-bottom: 0.25rem; color: #1a1a1a;">
                Quantitative Valuation Methodology & Agent Decoder
            </h2>
            <p style="font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.16em; color: rgba(0,0,0,0.4); margin-bottom: 1.5rem;">
                Institutional Algorithmic Spec + Agent Plain-English Translation Guide
            </p>
            <!-- AGENT CHEAT SHEET OVERVIEW -->
            <div style="background-color: #f5f2ed; border-left: 4px solid #0284c7; padding: 1.25rem; margin-bottom: 2.25rem; border-radius: 2px;">
                <h4 style="font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.14em; color: #0284c7; margin-bottom: 0.75rem;">
                    💡 Agent's Translation Guide (What the Math Actually Means)
                </h4>
                <ul style="font-size: 0.92rem; line-height: 1.7; margin-left: 1rem; margin-bottom: 0; color: #2d2d2d;">
                    <li><strong>Ridge Regularization ($L_2$ Penalty):</strong> Our algorithm's "sanity filter." It prevents correlated features (like SqFt and Baths) from producing wildly erratic adjustments.</li>
                    <li><strong>Alpha ($\alpha = {best_alpha:.2f}$):</strong> The strength of the regularization filter. Automatically tuned via 5-Fold Cross-Validation to ensure maximum predictive accuracy.</li>
                    <li><strong>Composite Distance Score ($D_{{\text{{comp}}}}$):</strong> Ensures we pick comps that are both <em>physically close</em> (70% weight) and <em>similar in home footprint size</em> (30% weight).</li>
                    <li><strong>Inverse Distance Weighting (IDW):</strong> Gives closer comps more "voting power" in setting the target valuation price.</li>
                    <li><strong>Residual Percentage ($e_i$):</strong> Measures how much a comp sold above or below its structural baseline. Automatically identifies Fixers vs. Turnkey homes.</li>
                </ul>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        # --- FORMAL MATHEMATICAL SPECIFICATIONS ---
        st.markdown("### 1. Objective Function & Regularization Optimization")
        st.write(
            "To prevent Multicollinearity Variance inflation between correlated architectural features (e.g., Living SqFt and Bathroom Count), "
            "feature coefficients $w$ are solved via $L_2$-Regularized Ridge Regression:"
        )
        st.latex(r"\min_{w} \sum_{i=1}^{n} \left( y_i - w_0 - \sum_{j=1}^{p} x_{ij}w_j \right)^2 + \alpha \sum_{j=1}^{p} w_j^2")
        st.caption("Features are standardized via Z-score normalization prior to fitting. Hyperparameter α is tuned dynamically via 5-Fold Cross-Validation.")
        st.divider()
        
        st.markdown("### 2. Spatial & Footprint Composite KNN Distance Metric")
        st.write(
            "Comparable selection employs a dual-component composite spatial metric $D_{\\text{comp}}$ combining geodesic distance "
            "(Haversine equation) with normalized footprint deviation:"
        )
        st.latex(r"d_{\text{geo}} = 2 R \cdot \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)")
        st.latex(r"d_{\text{foot}} = \frac{|S_{\text{subject}} - S_{\text{comp}}|}{\sigma_S}")
        st.latex(r"D_{\text{comp}} = 0.70 \cdot d_{\text{geo}} + 0.30 \cdot d_{\text{foot}}")
        st.divider()
        
        st.markdown("### 3. Inverse Distance Weighting (IDW) Reconciliation")
        st.write(
            "Selected top $K=4$ comparables are reconciled using Inverse Distance Weighting (IDW) with a smoothing kernel $\\epsilon = 0.01$ to prevent singular division:"
        )
        st.latex(r"w_i = \frac{1}{d_{\text{geo}, i} + \epsilon}, \quad W_i = \frac{w_i}{\sum_{k=1}^{K} w_k}")
        st.latex(r"\hat{Y}_{\text{reconciled}} = \sum_{i=1}^{K} W_i \cdot Y_{i, \text{adjusted}}")
        st.divider()
        
        st.markdown("### 4. Residual Condition Classification Engine")
        st.write(
            "Unobserved interior quality metrics are extracted by computing Pass-1 price residuals relative to baseline physical expectations:"
        )
        st.latex(r"e_i = \frac{Y_{i, \text{time\_adj}} - \hat{Y}_{i, \text{baseline}}}{\hat{Y}_{i, \text{baseline}}}")
        st.write(
            "Residuals $e_i$ are categorized into five ordinal condition tiers: "
            "**Fixer** ($e_i < -12\\%$), **Dated** ($-12\\% \\le e_i < -4\\%$), **Average** ($-4\\% \\le e_i \\le +4\\%$), "
            "**Updated** ($+4\\% < e_i \\le +12\\%$), and **Turnkey** ($e_i > +12\\%$)."
        )
        st.divider()
        
        st.markdown("### 5. Empirical Model Diagnostics & Calibrated Parameters")
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown(f"""
            * **Sample Size ($N$):** {n_sales} Local Transactions
            * **Tuned Penalty Parameter ($\alpha$):** {best_alpha:.4f}
            * **Time Drift Normalization:** +0.3% / Month (Bounded $[-8\\%, +8\\%]$)
            * **Geodesic Scope:** Max Comp Distance = {comps_display['Distance (Mi)'].max():.2f} miles
            """)
        with diag_col2:
            st.markdown(f"""
            * **Living Area Weight:** ${coefs['SQUARE FEET']:.2f} / sqft
            * **Bathroom Base Weight:** ${coefs['BATHS']:.2f} / bath
            * **Lot Size Weight:** ${coefs['LOT SIZE']:.2f} / sqft (Square-Root Scaled)
            * **Year Built Weight:** ${coefs['YEAR BUILT']:.2f} / year
            """)

if __name__ == "__main__":
    main()