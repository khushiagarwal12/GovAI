import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from app.gemini_helpers import make_prompt_from_df, call_gemini_for_analysis
from thefuzz import process

# ---------------------------
# Page Setup and Styling
# ---------------------------
st.set_page_config(page_title="GovAI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #f7fafc;
    }
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
# Load & Clean Data
# ---------------------------
df = pd.read_csv("data/cleaned_mortality_final.csv")

# --- Normalize city names ---
df["City Name"] = (
    df["City Name"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[-_]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

def unify_city_names(series, threshold=90):
    unique_names = list(series.unique())
    canonical = {}

    for name in unique_names:
        best_match, score = (None, 0)
        if canonical:
            result = process.extractOne(name, canonical.keys(), score_cutoff=threshold)
            if result is not None:
                best_match, score = result

        if best_match:
            canonical[name] = best_match
        else:
            canonical[name] = name

    return series.map(canonical)


df["City Name"] = unify_city_names(df["City Name"])

# --- Clean numeric columns ---
def clean_numeric(series):
    return (
        series.astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

for col in ["No. of Deaths - Total", "Total No. of Live Births"]:
    df[col] = clean_numeric(df[col])

# ---------------------------
# Filters Section
# ---------------------------
with st.container():
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🎯 Filters</div>", unsafe_allow_html=True)

    cities = sorted(df["City Name"].unique())
    years = sorted(df["Year"].unique())

    col1, col2 = st.columns([1, 1])
    with col1:
        sel_cities = st.multiselect("🏙️ Select Cities", cities, default=cities[:5])
    with col2:
        sel_years = st.multiselect("📅 Select Years", years, default=years[-3:])
    st.markdown("</div>", unsafe_allow_html=True)

filtered = df[df["City Name"].isin(sel_cities) & df["Year"].isin(sel_years)]

# ---------------------------
# Dashboard Layout
# ---------------------------
col_left, col_right = st.columns([1.2, 1.8])

# ---------------------------
# LEFT COLUMN - DATA + STATS
# ---------------------------
with col_left:
    with st.container():
        st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Filtered Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(filtered.head(15), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 Summary Statistics</div>", unsafe_allow_html=True)

        total_deaths = filtered["No. of Deaths - Total"].sum(skipna=True)
        total_deaths = int(total_deaths) if not pd.isna(total_deaths) else 0

        avg_births = filtered["Total No. of Live Births"].mean(skipna=True)
        avg_births = int(avg_births) if not pd.isna(avg_births) else 0

        c1, c2 = st.columns(2)
        c1.metric("🕊️ Total Deaths", f"{total_deaths:,}")
        c2.metric("👶 Avg. Births", f"{avg_births:,}")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# RIGHT COLUMN - GEMINI ANALYSIS
# ---------------------------
with col_right:
    with st.container():
        st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🧠 Gemini AI Analysis</div>", unsafe_allow_html=True)

        if st.button("🚀 Generate AI Insights"):
            with st.spinner("Generating insights... please wait ⏳"):
                try:
                    sampled_df = filtered.sample(min(len(filtered), 20), random_state=42)
                    prompt = make_prompt_from_df(sampled_df)
                    j, raw = call_gemini_for_analysis(prompt)
                except Exception as e:
                    st.error(f"Error while calling Gemini API: {e}")
                    j, raw = None, None

            if not j:
                st.warning("⚠️ No valid response received.")
            else:
                st.markdown("#### 📘 AI Summary")
                st.write(j.get("summary", "No summary available."))

                st.markdown("#### 🔍 Key Interpretations")
                for it in j.get("interpretations", []):
                    st.markdown(f"- {it.get('text', '')}")

                st.markdown("#### ⚠️ Top Risks")
                if isinstance(j.get("top_risks"), list) and len(j["top_risks"]) > 0:
                    st.dataframe(pd.DataFrame(j["top_risks"]), width="stretch")
                else:
                    st.write("No major risks identified.")

                st.markdown("#### ✅ Recommendations")
                for rec in j.get("recommendations", []):
                    st.markdown(
                        f"- **{rec.get('action', rec.get('text', ''))}** "
                        f"(Dept: {rec.get('department', 'N/A')}, "
                        f"Urgency: {rec.get('urgency', 'N/A')}) — "
                        f"{rec.get('rationale', '')}"
                    )

        st.markdown("</div>", unsafe_allow_html=True)
