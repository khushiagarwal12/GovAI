# File: GovAI/app/main_app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
from pathlib import Path
import streamlit as st
import pandas as pd
from thefuzz import process
from app.gemini_helpers import make_prompt_from_df, call_gemini_for_analysis

# ---------------------------
# Paths & Constants
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # GovAI/app
ROOT_DIR = os.path.dirname(BASE_DIR)                         # GovAI
DEFAULT_CSV_PATH = os.path.join(ROOT_DIR, "data", "cleaned_mortality_final.csv")
TIRANGA_PATH = os.path.join(BASE_DIR, "tiranga.jpeg")
ADMIN_PASSWORD = "admin123"

# ---------------------------
# Page config & styling (flat UI)
# ---------------------------
st.set_page_config(page_title="GovAI", layout="wide")

# Load tiranga image as base64 (if present)
tiranga_b64 = None
if os.path.exists(TIRANGA_PATH):
    with open(TIRANGA_PATH, "rb") as f:
        tiranga_b64 = base64.b64encode(f.read()).decode()

# Minimal CSS: title with background image; remove card boxes
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #ffffff; }}
    .govai-title {{
        text-align: center;
        font-size: 2.4rem;
        color: #ffffff;
        font-weight: 800;
        margin: 0 0 8px 0;
        padding: 48px 12px;
        border-radius: 12px;
        background-color: #1E3A8A;
        {"background-image: url('data:image/png;base64," + tiranga_b64 + "'); background-size: cover; background-position: center;" if tiranga_b64 else ""}
    }}
    .govai-subtitle {{
        text-align: center;
        color: #1E293B;
        font-size: 1rem;
        margin-bottom: 20px;
    }}
    /* remove Streamlit default card look for main containers */
    .streamlit-expanderHeader {{}}
    .block-container {{ padding: 24px 36px; }}
    .section-header {{
        color: #0f172a;
        font-weight: 700;
        margin: 8px 0 6px 0;
        font-size: 1.05rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header area (title + subtitle)
st.markdown(f"<div class='govai-title'>GovAI Dashboard</div>", unsafe_allow_html=True)
st.markdown(f"<div class='govai-subtitle'>AI-powered analytics for city mortality and health trends</div>", unsafe_allow_html=True)

# ---------------------------
# Session-state-backed DataFrame (persist while session runs)
# ---------------------------
if "df" not in st.session_state:
    # Load default CSV from sibling data/ folder
    if not os.path.exists(DEFAULT_CSV_PATH):
        st.error(f"❌ Default dataset not found at: {DEFAULT_CSV_PATH}")
        st.stop()
    try:
        st.session_state.df = pd.read_csv(DEFAULT_CSV_PATH)
    except Exception as e:
        st.error(f"❌ Failed to read default CSV: {e}")
        st.stop()

# convenience reference
df = st.session_state.df

# ---------------------------
# Cleaning helpers
# ---------------------------
def unify_city_names(series: pd.Series, threshold: int = 90) -> pd.Series:
    """Map variant city names to canonical using fuzzy matching across observed names."""
    unique_names = list(series.dropna().unique())
    canonical = {}
    for name in unique_names:
        best_match = None
        if canonical:
            result = process.extractOne(name, canonical.keys(), score_cutoff=threshold)
            if result:
                best_match = result[0]
        canonical[name] = best_match if best_match else name
    return series.map(canonical)

def clean_numeric(series: pd.Series) -> pd.Series:
    """Extract first numeric group and convert to float (keeps NaN if no match)."""
    return series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)

# Ensure idempotent cleaning for base df
if "City Name" in df.columns:
    df["City Name"] = (
        df["City Name"].astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[-_]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )
    df["City Name"] = unify_city_names(df["City Name"])

for col in ["No. of Deaths - Total", "Total No. of Live Births"]:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

# Save cleaned df back to session state
st.session_state.df = df

# ---------------------------
# Tabs: Dashboard & Admin
# ---------------------------
tab_dashboard, tab_admin = st.tabs(["Dashboard", "Admin"])

# ---------------------------
# DASHBOARD
# ---------------------------
with tab_dashboard:
    # Simple filter area (flat, no boxes)
    st.markdown("<div class='section-header'>🎯 Filters</div>", unsafe_allow_html=True)

    # Validate required columns exist
    if "City Name" not in df.columns or "Year" not in df.columns:
        st.error("Dataset missing required columns: 'City Name' and/or 'Year'. Upload a dataset that contains these columns.")
        st.stop()

    cities = sorted(df["City Name"].dropna().unique())
    years = sorted(df["Year"].dropna().unique())

    col1, col2 = st.columns([1, 1])
    with col1:
        sel_cities = st.multiselect("🏙️ Select Cities", cities, default=cities[:5])
    with col2:
        sel_years = st.multiselect("📅 Select Years", years, default=years[-3:])

    if not sel_cities:
        sel_cities = cities[:5]
    if not sel_years:
        sel_years = years[-3:]

    filtered = st.session_state.df[st.session_state.df["City Name"].isin(sel_cities) & st.session_state.df["Year"].isin(sel_years)]

    # Layout: left table / stats, right AI panel
    col_left, col_right = st.columns([1.2, 1.8])

    # LEFT: data preview + metrics (flat)
    with col_left:
        st.markdown("<div class='section-header'>📊 Filtered Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(filtered.head(20), use_container_width=True)

        st.markdown("<div class='section-header'>📈 Summary Statistics</div>", unsafe_allow_html=True)
        total_deaths = int(filtered["No. of Deaths - Total"].sum(skipna=True) or 0) if "No. of Deaths - Total" in filtered.columns else 0
        avg_births = int(filtered["Total No. of Live Births"].mean(skipna=True) or 0) if "Total No. of Live Births" in filtered.columns else 0
        m1, m2 = st.columns(2)
        m1.metric("🕊️ Total Deaths", f"{total_deaths:,}")
        m2.metric("👶 Avg. Births", f"{avg_births:,}")

    # RIGHT: Gemini AI analysis
    with col_right:
        st.markdown("<div class='section-header'>🧠 Gemini AI Analysis</div>", unsafe_allow_html=True)
        if st.button("🚀 Generate AI Insights"):
            with st.spinner("Generating insights..."):
                try:
                    sample_df = filtered.sample(min(len(filtered), 20), random_state=42) if len(filtered) > 0 else filtered
                    prompt = make_prompt_from_df(sample_df)
                    parsed_json, raw_text = call_gemini_for_analysis(prompt)
                except Exception as e:
                    st.error(f"Error while calling Gemini: {e}")
                    parsed_json, raw_text = None, None

            if not parsed_json:
                st.warning("⚠️ No valid JSON response from Gemini or parsing failed.")
                if raw_text:
                    st.text_area("Raw AI output (truncated)", raw_text[:4000], height=200)
            else:
                st.markdown("#### 📘 AI Summary")
                st.write(parsed_json.get("summary", "No summary."))

                st.markdown("#### 🔍 Key Interpretations")
                for it in parsed_json.get("interpretations", []):
                    if isinstance(it, dict):
                        text = it.get("text", "")
                    else:
                        text = str(it)
                    st.markdown(f"- {text}")

                st.markdown("#### ⚠️ Top Risks")
                top_risks = parsed_json.get("top_risks", [])
                if isinstance(top_risks, list) and top_risks:
                    st.dataframe(pd.DataFrame(top_risks), use_container_width=True)
                else:
                    st.write("No major risks identified.")

                st.markdown("#### ✅ Recommendations")
                for rec in parsed_json.get("recommendations", []):
                    if isinstance(rec, dict):
                        action = rec.get("action", rec.get("text", ""))
                        dept = rec.get("department", "N/A")
                        urgency = rec.get("urgency", "N/A")
                        rationale = rec.get("rationale", "")
                        st.markdown(f"- **{action}** (Dept: {dept}, Urgency: {urgency}) — {rationale}")
                    else:
                        st.markdown(f"- {rec}")

# ---------------------------
# ADMIN
# ---------------------------
with tab_admin:
    st.markdown("<div class='section-header'>🔐 Admin — Upload Additional CSVs</div>", unsafe_allow_html=True)
    pwd = st.text_input("Enter Admin Password", type="password")
    if pwd == ADMIN_PASSWORD:
        st.success("✅ Access granted")
        uploaded_files = st.file_uploader("Upload CSV files to merge with default dataset", accept_multiple_files=True, type=["csv"])
        if uploaded_files:
            merged = 0
            for upload in uploaded_files:
                try:
                    new_df = pd.read_csv(upload)

                    # Normalize if City Name present
                    if "City Name" in new_df.columns:
                        new_df["City Name"] = (
                            new_df["City Name"].astype(str)
                            .str.strip()
                            .str.lower()
                            .str.replace(r"[-_]", " ", regex=True)
                            .str.replace(r"\s+", " ", regex=True)
                            .str.title()
                        )
                        new_df["City Name"] = unify_city_names(new_df["City Name"])

                    for col in ["No. of Deaths - Total", "Total No. of Live Births"]:
                        if col in new_df.columns:
                            new_df[col] = clean_numeric(new_df[col])

                    # Append to session DataFrame
                    st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
                    merged += 1
                    st.success(f"Merged: {upload.name} (rows: {len(new_df)})")
                except Exception as e:
                    st.error(f"Failed to merge {upload.name}: {e}")

            if merged:
                st.info(f"✅ {merged} file(s) merged. Use the Dashboard tab to view merged data.")
        else:
            st.info("No files uploaded. Upload CSVs to merge them with the default dataset.")
    elif pwd:
        st.error("❌ Incorrect password")
