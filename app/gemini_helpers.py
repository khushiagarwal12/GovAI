import os
import json
import pandas as pd
import asyncio
from datetime import datetime
from textwrap import shorten
import google.generativeai as genai

# ✅ Configure Gemini API (no Vertex AI)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# ---------- Utilities ----------
def df_top_stats(df, max_cities=6):
    """Compute lightweight numeric-only stats for speed."""
    numeric_cols = df.select_dtypes("number")
    stats = numeric_cols.describe().to_dict()
    sample = df.nlargest(max_cities, "No. of Deaths - Total", keep="first")
    return stats, sample.to_csv(index=False)


def make_prompt_from_df(df, context_instructions=None, max_rows=150):
    """Optimized prompt creation – smaller, lighter, faster."""
    df = df.copy()
    keep = [
        "City Name", "Year",
        "Total No. of Live Births", "No. of Deaths - Total",
        "No. of Deaths - Male", "No. of Deaths - Female",
        "No. of Deaths - Infants (0-1 year)",
        "No. of Deaths - Children (1-5 years)",
        "No. of Deaths - age above 5 years"
    ]
    df = df[[c for c in keep if c in df.columns]]

    n = len(df)
    # Instead of sending 20 full rows, send statistical slices
    if n > max_rows:
        top = df.nlargest(8, "No. of Deaths - Total")
        bottom = df.nsmallest(8, "No. of Deaths - Total")
        df = pd.concat([top, bottom])
        note = f"Full dataset: {n} rows (showing top/bottom 8).\n"
    else:
        note = f"Dataset rows: {n}.\n"

    # Simplify JSON stats to cut down token size
    stats, _ = df_top_stats(df)
    stats_json = json.dumps(stats, default=str)

    # Build compact CSV (trim decimals + column headers)
    data_blob = df.round(2).to_csv(index=False)

    instructions = context_instructions or (
        "You are an expert public health analyst. Analyze the provided dataset "
        "and return insights ONLY in this JSON format:\n"
        "{'summary': str, 'interpretations': [{'text': str}], "
        "'top_risks': [{'risk': str, 'severity': str, 'reason': str}], "
        "'recommendations': [{'action': str, 'department': str, 'urgency': str, 'rationale': str}], "
        "'confidence': float, 'metadata': {'source': 'Gemini', 'timestamp': str}}\n"
        "Output valid JSON only."
    )

    return (
        f"{instructions}\n\n{note}"
        f"DATA SAMPLE (CSV):\n{data_blob}\n\n"
        f"AGGREGATED_STATS: {stats_json}\n"
        "Respond with JSON only."
    )


async def _call_gemini_async(prompt, model=MODEL, temperature=0.3):
    """Stream Gemini output as it’s generated."""
    model_handle = genai.GenerativeModel(model)
    text_chunks = []

    # Use the streaming API
    with model_handle.generate_content_stream(
        prompt,
        generation_config={"temperature": temperature}
    ) as stream:
        for event in stream:
            if event.text:
                # Display partial text as it arrives (optional)
                placeholder = st.empty()
                ...
                placeholder.write(event.text)
                text_chunks.append(event.text)

    return "".join(text_chunks)



def _parse_gemini_output(text):
    """Robust, fast JSON parser with graceful fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        # fallback
        return {
            "summary": shorten(text, 400),
            "interpretations": [
                {"text": line.strip("- ").strip()} for line in text.split("\n") if len(line.strip()) > 25
            ][:5],
            "top_risks": [],
            "recommendations": [],
            "confidence": 0.0,
            "metadata": {
                "note": "Fallback non-JSON output",
                "timestamp": datetime.utcnow().isoformat()
            },
            "raw_text": text,
        }


def call_gemini_for_analysis(prompt, model=MODEL, temperature=0.3):
    """Call Gemini API safely and quickly (no streaming)."""
    try:
        model_handle = genai.GenerativeModel(model)

        # ✅ Use the correct call method depending on SDK version
        if hasattr(model_handle, "generate_content"):
            resp = model_handle.generate_content(prompt)
        else:
            # For older SDKs (fallback)
            resp = model_handle.generate_text(prompt)

        text = getattr(resp, "text", str(resp))
    except Exception as e:
        return {
            "summary": f"Error calling Gemini API: {e}",
            "interpretations": [],
            "top_risks": [],
            "recommendations": [],
            "confidence": 0.0,
            "metadata": {"source": "Gemini", "timestamp": datetime.utcnow().isoformat()},
        }, str(e)

    # --- Try JSON parsing ---
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start:end])
            except Exception:
                parsed = None

    # --- Fallback if not valid JSON ---
    if not parsed:
        parsed = {
            "summary": shorten(text, 400),
            "interpretations": [line.strip("- ").strip() for line in text.split("\n") if len(line.strip()) > 20][:5],
            "top_risks": [],
            "recommendations": [],
            "confidence": 0.0,
            "metadata": {"note": "Fallback: non-JSON output", "timestamp": datetime.utcnow().isoformat()},
            "raw_text": text,
        }

    # --- Normalize lists ---
    def normalize_list(lst, key_name="text"):
        if not isinstance(lst, list):
            return []
        cleaned = []
        for item in lst:
            if isinstance(item, dict):
                cleaned.append(item)
            else:
                cleaned.append({key_name: str(item)})
        return cleaned

    parsed["interpretations"] = normalize_list(parsed.get("interpretations", []), "text")
    parsed["recommendations"] = normalize_list(parsed.get("recommendations", []), "action")

    return parsed, text