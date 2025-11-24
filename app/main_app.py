import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from app.gemini_helpers import make_prompt_from_df, call_gemini_for_analysis
from thefuzz import process
import base64

# -----------------------------------------------------------
# BASE DIR / RELATIVE PATHS
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIRANGA_PATH = os.path.join(BASE_DIR, "tiranga.jpeg")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_mortality_final.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="GovAI Dashboard",
    layout="wide",
)

# -----------------------------------------------------------
# Load background image
# -----------------------------------------------------------
try:
    with open(TIRANGA_PATH, "rb") as f:
        data = f.read()
        b64_data = base64.b64encode(data).decode()
except FileNotFoundError:
    b64_data = ""  # fallback if image missing

# -----------------------------------------------------------
# Inject CSS for header (SAFE)
# -----------------------------------------------------------
st.markdown(f"""
<style>
    .block-container {{
        padding-top: 15px !important;
        padding-left: 40px !important;
        padding-right: 40px !important;
    }}

    .main-title {{
        text-align: center;
        font-size: 2.4rem;
        color: white;
        font-weight: 800;
        margin-bottom: 5px;
        padding: 70px 0;
        {"background-image: url('data:image/png;base64," + b64_data + "');" if b64_data else ""}
        background-size: cover;
        background-position: center;
        border-radius: 15px;
        width: 100%;
    }}

    .sub-title {{
        text-align: center;
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }}

    .dashboard-box {{
        background: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }}

    .section-header {{
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>GovAI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>AI-powered analytics for city mortality and health trends</p>", unsafe_allow_html=True)

# -----------------------------------------------------------
# Load & Clean Original Data
# -----------------------------------------------------------
try:
    df = pd.read_csv(DEFAULT_CSV_PATH)
except FileNotFoundError:
    st.warning("Default CSV not found. Please upload a CSV in the Admin tab.")
    df = pd.DataFrame()  # empty dataframe

if not df.empty:
    # Normalize city names
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
            if canonical:
                result = process.extractOne(name, canonical.keys(), score_cutoff=threshold)
                if result:
                    canonical[name] = result[0]
                    continue
            canonical[name] = name
        return series.map(canonical)

    df["City Name"] = unify_city_names(df["City Name"])

    # Clean numeric columns
    def clean_numeric(series):
        return (
            series.astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
            .astype(float)
        )

    for col in ["No. of Deaths - Total", "Total No. of Live Births"]:
        df[col] = clean_numeric(df[col])

# -----------------------------------------------------------
# ADMIN TAB FOR CSV UPLOAD
# -----------------------------------------------------------
admin_password = "admin123"
tab1, tab2 = st.tabs(["Dashboard", "Admin"])

with tab2:
    st.markdown("<h2>🔐 Admin Panel</h2>", unsafe_allow_html=True)
    password = st.text_input("Enter Admin Password", type="password")

    if password == admin_password:
        st.success("✅ Access Granted")

        uploaded_file = st.file_uploader("Upload CSV for processing", type=["csv"])
        if uploaded_file:
            save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to {save_path}")

        # List uploaded files
        st.markdown("### Uploaded Files")
        uploaded_files = os.listdir(UPLOAD_FOLDER)
        if uploaded_files:
            files_to_delete = st.multiselect("Select files to delete", uploaded_files)
            if st.button("Delete Selected Files"):
                for file_name in files_to_delete:
                    os.remove(os.path.join(UPLOAD_FOLDER, file_name))
                st.success("Selected files deleted.")

            files_to_load = st.multiselect("Select files to merge with dashboard data", uploaded_files)
            if st.button("Merge Selected Files"):
                for file_name in files_to_load:
                    try:
                        new_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file_name))
                        if not new_df.empty and "City Name" in new_df.columns:
                            # Normalize city names
                            new_df["City Name"] = (
                                new_df["City Name"]
                                .astype(str)
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
                            df = pd.concat([df, new_df], ignore_index=True)
                            st.success(f"{file_name} merged successfully! Data now has {len(df)} rows.")
                        else:
                            st.warning(f"{file_name} is empty or missing 'City Name' column.")
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
        else:
            st.info("No uploaded files yet.")

    elif password:
        st.error("❌ Incorrect password")

# -----------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------
with tab1:
    if df.empty:
        st.info("No data available. Please upload CSVs in the Admin tab.")
    else:
        # Filters Section
        with st.container():
            st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🎯 Filters</div>", unsafe_allow_html=True)

            cities = sorted(df["City Name"].unique())
            years = sorted(df["Year"].unique())

            col1, col2 = st.columns([1, 1])
            with col1:
                sel_cities = st.multiselect("🏙 Select Cities", cities, default=cities[:5])
            with col2:
                sel_years = st.multiselect("📅 Select Years", years, default=years[-3:])

            st.markdown("</div>", unsafe_allow_html=True)

        filtered = df[df["City Name"].isin(sel_cities) & df["Year"].isin(sel_years)]

        # Dashboard Layout
        col_left, col_right = st.columns([1.2, 1.8])

        # LEFT COLUMN - DATA + STATS
        with col_left:
            with st.container():
                st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📊 Filtered Data Preview</div>", unsafe_allow_html=True)
                st.dataframe(filtered.head(15), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📈 Summary Statistics</div>", unsafe_allow_html=True)

                total_deaths = int(filtered["No. of Deaths - Total"].sum(skipna=True) or 0)
                avg_births = int(filtered["Total No. of Live Births"].mean(skipna=True) or 0)

                c1, c2 = st.columns(2)
                c1.metric("🕊 Total Deaths", f"{total_deaths:,}")
                c2.metric("👶 Avg. Births", f"{avg_births:,}")
                st.markdown("</div>", unsafe_allow_html=True)

        # RIGHT COLUMN - GEMINI ANALYSIS
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
                        st.warning("⚠ No valid response received.")
                    else:
                        # Filter business-relevant insights
                        def filter_business_insights(items):
                            filtered = []
                            for it in items:
                                text = it.get("text", "").lower() if "text" in it else ""
                                action = it.get("action", "").lower() if "action" in it else ""
                                if any(keyword in text for keyword in ["missing", "nan", "null", "data validation", "incomplete"]):
                                    continue
                                if any(keyword in action for keyword in ["missing", "nan", "null", "data validation", "incomplete"]):
                                    continue
                                filtered.append(it)
                            return filtered

                        key_interpretations = filter_business_insights(j.get("interpretations", []))
                        recommendations = filter_business_insights(j.get("recommendations", []))

                        st.markdown("#### 📘 AI Summary")
                        st.write(j.get("summary", "No summary available."))

                        st.markdown("#### 🔍 Key Interpretations")
                        for it in key_interpretations:
                            st.markdown(f"- {it.get('text', '')}")

                        st.markdown("#### ⚠ Top Risks")
                        if isinstance(j.get("top_risks"), list) and len(j["top_risks"]) > 0:
                            st.dataframe(pd.DataFrame(j["top_risks"]), use_container_width=True)
                        else:
                            st.write("No major risks identified.")

                        st.markdown("#### ✅ Recommendations")
                        for rec in recommendations:
                            st.markdown(
                                f"- *{rec.get('action', rec.get('text', ''))}* "
                                f"(Dept: {rec.get('department', 'N/A')}, "
                                f"Urgency: {rec.get('urgency', 'N/A')}) — "
                                f"{rec.get('rationale', '')}"
                            )
