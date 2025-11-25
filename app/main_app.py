import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from thefuzz import process
from app.gemini_helpers import make_prompt_from_df, call_gemini_for_analysis

# ---------------------------
# Page Setup & Styling
# ---------------------------
st.set_page_config(page_title="GovAI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f7fafc; }
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #2563EB;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    .dashboard-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .section-header {
        color: #2563EB;
        font-size: 1.3rem;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1 class='main-title'>GovAI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>AI-powered analytics for city mortality and health trends</p>", unsafe_allow_html=True)

# ---------------------------
# SAFE DEFAULT DATA LOADING
# ---------------------------
DEFAULT_DATA_PATH = "data/cleaned_mortality_final.csv"

if not os.path.exists(DEFAULT_DATA_PATH):
    st.error(f"❌ Default dataset not found at: {DEFAULT_DATA_PATH}")
    st.stop()

# load default dataset
try:
    df = pd.read_csv(DEFAULT_DATA_PATH)
except Exception as e:
    st.error(f"❌ Error reading default dataset: {e}")
    st.stop()

# ---------------------------
# Cleaning Helpers
# ---------------------------
def unify_city_names(series, threshold=90):
    unique_names = list(series.dropna().unique())
    canonical = {}

    for name in unique_names:
        best_match = None
        if canonical:
            result = process.extractOne(name, canonical.keys(), score_cutoff=threshold)
            if result is not None:
                best_match = result[0]
        canonical[name] = best_match if best_match else name

    return series.map(canonical)

def clean_numeric(series):
    # extract first numeric group (integer/float), convert to float, keep NaN if missing
    return (
        series.astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

# ---------------------------
# DEFAULT DATA CLEANING (idempotent)
# ---------------------------
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

# ---------------------------
# Tabs: Dashboard + Admin
# ---------------------------
tab_dashboard, tab_admin = st.tabs(["📊 Dashboard", "🔐 Admin"])

# ---------------------------
# DASHBOARD TAB
# ---------------------------
with tab_dashboard:
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🎯 Filters</div>", unsafe_allow_html=True)

    # guard: required columns
    if "City Name" not in df.columns or "Year" not in df.columns:
        st.error("Dataset missing required columns: 'City Name' and/or 'Year'.")
    else:
        cities = sorted(df["City Name"].dropna().unique())
        years = sorted(df["Year"].dropna().unique())

        col1, col2 = st.columns([1, 1])
        with col1:
            sel_cities = st.multiselect("🏙️ Select Cities", cities, default=cities[:5])
        with col2:
            sel_years = st.multiselect("📅 Select Years", years, default=years[-3:])

    st.markdown("</div>", unsafe_allow_html=True)

    # filtered view (if filters exist)
    if "City Name" in df.columns and "Year" in df.columns:
        if not sel_cities:
            sel_cities = cities[:5]  # fallback if user cleared selection
        if not sel_years:
            sel_years = years[-3:]

        filtered = df[df["City Name"].isin(sel_cities) & df["Year"].isin(sel_years)]
    else:
        filtered = df.copy()

    # layout columns
    col_left, col_right = st.columns([1.2, 1.8])

    # LEFT COLUMN - Data + Stats
    with col_left:
        st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Filtered Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(filtered.head(20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 Summary Statistics</div>", unsafe_allow_html=True)

        total_deaths = 0
        avg_births = 0
        if "No. of Deaths - Total" in filtered.columns:
            total_deaths = int(filtered["No. of Deaths - Total"].sum(skipna=True) or 0)
        if "Total No. of Live Births" in filtered.columns:
            avg_births = int(filtered["Total No. of Live Births"].mean(skipna=True) or 0)

        c1, c2 = st.columns(2)
        c1.metric("🕊️ Total Deaths", f"{total_deaths:,}")
        c2.metric("👶 Avg. Births", f"{avg_births:,}")

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN - Gemini Analysis
    with col_right:
        st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🧠 Gemini AI Analysis</div>", unsafe_allow_html=True)

        if st.button("🚀 Generate AI Insights"):
            with st.spinner("Generating insights..."):
                try:
                    sample_df = filtered.sample(min(len(filtered), 20), random_state=42) if len(filtered) > 0 else filtered
                    prompt = make_prompt_from_df(sample_df)
                    parsed_json, raw_text = call_gemini_for_analysis(prompt)
                except Exception as e:
                    st.error(f"Error while calling Gemini API: {e}")
                    parsed_json, raw_text = None, None

            if not parsed_json:
                st.warning("⚠️ No valid response received or parsing failed.")
                if raw_text:
                    st.text_area("Raw AI output (truncated)", raw_text[:4000], height=200)
            else:
                st.markdown("#### 📘 AI Summary")
                st.write(parsed_json.get("summary", "No summary available."))

                st.markdown("#### 🔍 Key Interpretations")
                for it in parsed_json.get("interpretations", []):
                    text = it.get("text") if isinstance(it, dict) else str(it)
                    st.markdown(f"- {text}")

                st.markdown("#### ⚠️ Top Risks")
                if isinstance(parsed_json.get("top_risks"), list) and parsed_json["top_risks"]:
                    st.dataframe(pd.DataFrame(parsed_json["top_risks"]), use_container_width=True)
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
                        st.markdown(f"- {str(rec)}")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# ADMIN TAB
# ---------------------------
with tab_admin:
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Admin Panel — Upload Additional CSVs", unsafe_allow_html=True)

    ADMIN_PASSWORD = "admin123"
    password = st.text_input("Enter Admin Password", type="password")

    if password == ADMIN_PASSWORD:
        st.success("✅ Access granted")

        uploaded_files = st.file_uploader("Upload CSV files to merge (optional)", accept_multiple_files=True, type=["csv"])
        if uploaded_files:
            merge_count = 0
            for file in uploaded_files:
                try:
                    new_df = pd.read_csv(file)

                    # normalize city names if present
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

                    # merge into the main dataframe
                    df = pd.concat([df, new_df], ignore_index=True)
                    merge_count += 1
                    st.success(f"Merged: {file.name} (rows: {len(new_df)})")
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")

            if merge_count:
                st.info(f"✅ {merge_count} file(s) merged. Dashboard will use merged data now.")
        else:
            st.info("No files uploaded. Upload CSVs to merge them with the default dataset.")

    elif password:
        st.error("❌ Incorrect password")

    st.markdown("</div>", unsafe_allow_html=True)
