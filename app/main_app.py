# File: GovAI/app/main_app.py
import sys
import os
from pathlib import Path
import base64

# ensure app package can import sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from thefuzz import process
from app.gemini_helpers import make_prompt_from_df, call_gemini_for_analysis

# ---------------------------
# Paths & constants
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # GovAI/app
ROOT_DIR = os.path.dirname(BASE_DIR)                     # GovAI
DEFAULT_CSV_PATH = os.path.join(ROOT_DIR, "data", "cleaned_mortality_final.csv")
TIRANGA_PATH = os.path.join(BASE_DIR, "tiranga.jpeg")
ADMIN_PASSWORD = "admin123"

# ---------------------------
# Page config & minimal styling (flat UI + faded tiranga watermark)
# ---------------------------
st.set_page_config(page_title="GovAI", layout="wide")

# load tiranga as base64 (if exists)
tiranga_b64 = None
if os.path.exists(TIRANGA_PATH):
    with open(TIRANGA_PATH, "rb") as f:
        tiranga_b64 = base64.b64encode(f.read()).decode()

# minimal CSS: title with faded watermark; black title text; thin separators for sections
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #ffffff; color: #0f172a; }}
    .govai-title-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 12px 0 6px 0;
        padding: 28px 14px;
        border-radius: 8px;
        {"background-image: linear-gradient(rgba(255,255,255,0.15), rgba(255,255,255,0.85)), url('data:image/png;base64," + tiranga_b64 + "');" if tiranga_b64 else "background-color: #ffffff;"}
        background-size: cover;
        background-position: center;
    }}
    .govai-title {{
        font-size: 2.4rem;
        color: #000000;     /* BLACK title text */
        font-weight: 800;
        margin: 0;
        padding: 0 6px;
    }}
    .govai-subtitle {{
        text-align: center;
        color: #0f172a;
        font-size: 1rem;
        margin-bottom: 18px;
    }}

    /* Thin black divider underneath section headers (Option C) */
    .section-header {{
        font-weight: 700;
        color: #0f172a;
        margin: 8px 0 4px 0;
        font-size: 1.05rem;
    }}
    .section-divider {{
        border: none;
        border-top: 1px solid #000000;
        margin: 4px 0 14px 0;
        width: 100%;
    }}

    /* remove default card shadows and heavy padding where possible */
    .block-container {{ padding: 18px 36px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header area (title + subtitle)
st.markdown(f"<div class='govai-title-wrapper'><h1 class='govai-title'>GovAI Dashboard</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='govai-subtitle'>AI-powered analytics for city mortality and health trends</div>", unsafe_allow_html=True)

# ---------------------------
# Load default dataset into session_state (so uploads/merges persist)
# ---------------------------
if "df" not in st.session_state:
    if not os.path.exists(DEFAULT_CSV_PATH):
        st.error(f"❌ Default dataset not found at: {DEFAULT_CSV_PATH}")
        st.stop()
    try:
        st.session_state.df = pd.read_csv(DEFAULT_CSV_PATH)
    except Exception as e:
        st.error(f"❌ Error reading default dataset: {e}")
        st.stop()

# convenience alias
df = st.session_state.df

# ---------------------------
# Cleaning helpers
# ---------------------------
def unify_city_names(series: pd.Series, threshold: int = 90) -> pd.Series:
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
    # extract first numeric group and convert to float; keeps NaN if missing
    return series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)

# apply idempotent cleaning if required columns exist
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

# save cleaned df back to session
st.session_state.df = df

# ---------------------------
# Tabs: Dashboard & Admin
# ---------------------------
tab_dashboard, tab_admin = st.tabs(["Dashboard", "Admin"])

# ---------------------------
# DASHBOARD
# ---------------------------
with tab_dashboard:
    # Filters section (flat header + thin line)
    st.markdown("<div class='section-header'>🎯 Filters</div>", unsafe_allow_html=True)
    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # validate essential columns
    if "City Name" not in df.columns or "Year" not in df.columns:
        st.error("Dataset missing required columns: 'City Name' and/or 'Year'. Upload a dataset that contains these columns.")
    else:
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

    # create filtered dataframe (safe fallback)
    if "City Name" in df.columns and "Year" in df.columns:
        filtered = st.session_state.df[st.session_state.df["City Name"].isin(sel_cities) & st.session_state.df["Year"].isin(sel_years)]
    else:
        filtered = st.session_state.df.copy()

    # Data preview + stats / AI analysis layout
    col_left, col_right = st.columns([1.2, 1.8])

    # LEFT: data preview and metrics
    with col_left:
        st.markdown("<div class='section-header'>📊 Filtered Data Preview</div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        st.dataframe(filtered.head(20), use_container_width=True)

        st.markdown("<div class='section-header'>📈 Summary Statistics</div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

        total_deaths = int(filtered["No. of Deaths - Total"].sum(skipna=True) or 0) if "No. of Deaths - Total" in filtered.columns else 0
        avg_births = int(filtered["Total No. of Live Births"].mean(skipna=True) or 0) if "Total No. of Live Births" in filtered.columns else 0
        m1, m2 = st.columns(2)
        m1.metric("🕊️ Total Deaths", f"{total_deaths:,}")
        m2.metric("👶 Avg. Births", f"{avg_births:,}")

    # RIGHT: Gemini AI panel
    with col_right:
        st.markdown("<div class='section-header'>🧠 Gemini AI Analysis</div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

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
                    text = it.get("text") if isinstance(it, dict) else str(it)
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
    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    pwd = st.text_input("Enter Admin Password", type="password")
    if pwd == ADMIN_PASSWORD:
        st.success("✅ Access granted")
        uploaded_files = st.file_uploader("Upload CSV files to merge with default dataset", accept_multiple_files=True, type=["csv"])
        if uploaded_files:
            merged = 0
            for upload in uploaded_files:
                try:
                    new_df = pd.read_csv(upload)

                    # Normalize city names if present
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

                    # append into session dataframe
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
