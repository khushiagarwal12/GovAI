# File: GovAI/app/gemini_helpers.py
import os
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai

# Configure Gemini (no error if env var missing; code will handle it)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def df_top_stats(df: pd.DataFrame, max_cities: int = 6):
    numeric_cols = df.select_dtypes(include="number")
    stats = numeric_cols.describe().to_dict()
    sample = df.nlargest(max_cities, "No. of Deaths - Total", keep="first") if "No. of Deaths - Total" in df.columns else df.head(max_cities)
    return stats, sample.to_csv(index=False)


def make_prompt_from_df(df: pd.DataFrame, context_instructions: str = None, max_rows: int = 150) -> str:
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
    if n > max_rows:
        top = df.nlargest(8, "No. of Deaths - Total") if "No. of Deaths - Total" in df.columns else df.head(8)
        bottom = df.nsmallest(8, "No. of Deaths - Total") if "No. of Deaths - Total" in df.columns else df.tail(8)
        df = pd.concat([top, bottom])
        note = f"Full dataset: {n} rows (showing top/bottom 8).\n"
    else:
        note = f"Dataset rows: {n}.\n"

    stats, _ = df_top_stats(df)
    stats_json = json.dumps(stats, default=str)
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


def _parse_gemini_output(text: str):
    if not text:
        return {
            "summary": "",
            "interpretations": [],
            "top_risks": [],
            "recommendations": [],
            "confidence": 0.0,
            "metadata": {"timestamp": datetime.utcnow().isoformat()},
            "raw_text": ""
        }

    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # try to extract JSON substring
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    # fallback structure
    shortened = text[:400]
    lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
    interpretations = [{"text": l} for l in lines[:5] if len(l) > 20]

    return {
        "summary": shortened,
        "interpretations": interpretations,
        "top_risks": [],
        "recommendations": [],
        "confidence": 0.0,
        "metadata": {"note": "fallback parsed", "timestamp": datetime.utcnow().isoformat()},
        "raw_text": text,
    }


def call_gemini_for_analysis(prompt: str, model_name: str = MODEL, temperature: float = 0.3):
    """
    Non-streaming call to Gemini. Returns (parsed_dict_or_fallback, raw_text).
    If GEMINI_API_KEY is not set or call fails, returns a structured error fallback.
    """
    try:
        model_handle = genai.GenerativeModel(model_name)
        if hasattr(model_handle, "generate_content"):
            resp = model_handle.generate_content(prompt, generation_config={"temperature": temperature})
        else:
            resp = model_handle.generate_text(prompt)
        raw_text = getattr(resp, "text", str(resp) or "")
    except Exception as e:
        raw_text = f"Error calling Gemini API: {e}"
        parsed = {
            "summary": raw_text,
            "interpretations": [],
            "top_risks": [],
            "recommendations": [],
            "confidence": 0.0,
            "metadata": {"source": "gemini", "timestamp": datetime.utcnow().isoformat()},
            "raw_text": raw_text,
        }
        return parsed, raw_text

    parsed = _parse_gemini_output(raw_text)
    return parsed, raw_text
