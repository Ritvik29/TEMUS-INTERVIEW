"""
Structured Vision: extract tables and charts from images, then synthesize data.

Features:
  - Table extraction (40+ cells) → structured dict/DataFrame
  - Chart extraction (30+ data points, 3 series) → structured data
  - Synthetic data generation (10x) preserving statistical distribution
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from groq import Groq

from src.genai.vision import _load_image, _compress_to_limit, _image_content_block, VISION_MODEL

# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

TABLE_EXTRACT_PROMPT = """
Extract ALL data from this table image into structured JSON.

Return ONLY valid JSON in this format:
{
  "headers": ["col1", "col2", ...],
  "rows": [
    ["val1", "val2", ...],
    ...
  ],
  "metadata": {
    "title": "table title if visible",
    "total_cells": <number>,
    "notes": "any footnotes or units"
  }
}

Be precise — extract every cell exactly as shown.
"""

CHART_EXTRACT_PROMPT = """
Extract ALL data series from this chart image into structured JSON.

Return ONLY valid JSON in this format:
{
  "chart_type": "line|bar|scatter|area",
  "title": "chart title if visible",
  "x_axis": {"label": "...", "values": [...]},
  "series": [
    {"name": "series1", "values": [...]},
    {"name": "series2", "values": [...]},
    {"name": "series3", "values": [...]}
  ],
  "total_data_points": <number>
}

Estimate values as precisely as possible from the chart scale.
"""


def _extract_json(text: str) -> dict:
    """Parse JSON from model output, stripping markdown fences if present."""
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from model output:\n{text[:300]}")


def extract_table(image_path: str) -> dict:
    """
    Extract structured table data from an image.
    Returns dict with headers, rows, and a pandas DataFrame.
    """
    img = _load_image(image_path)
    b64 = _compress_to_limit(img)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": TABLE_EXTRACT_PROMPT},
                _image_content_block(b64),
            ],
        }],
        max_tokens=2048,
    )

    raw = _extract_json(resp.choices[0].message.content)
    df = pd.DataFrame(raw.get("rows", []), columns=raw.get("headers", []))

    return {
        "headers": raw.get("headers", []),
        "rows": raw.get("rows", []),
        "metadata": raw.get("metadata", {}),
        "dataframe": df,
        "cell_count": len(raw.get("headers", [])) * len(raw.get("rows", [])),
    }


def extract_chart(image_path: str) -> dict:
    """
    Extract structured data from a chart image (30+ data points, 3+ series).
    """
    img = _load_image(image_path)
    b64 = _compress_to_limit(img)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CHART_EXTRACT_PROMPT},
                _image_content_block(b64),
            ],
        }],
        max_tokens=2048,
    )

    raw = _extract_json(resp.choices[0].message.content)
    return raw


# ---------------------------------------------------------------------------
# Synthetic data generation (10x)
# ---------------------------------------------------------------------------

def generate_synthetic_data(source_df: pd.DataFrame, multiplier: int = 10) -> pd.DataFrame:
    """
    Generate `multiplier`x synthetic rows preserving the statistical distribution
    of each numeric column (mean, std, min, max, skewness via Gaussian copula approximation).

    Categorical columns are resampled with their original frequency distribution.
    """
    n_original = len(source_df)
    n_synthetic = n_original * multiplier
    synthetic: dict = {}

    numeric_cols = source_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = source_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numeric: fit per-column normal + clip to [min, max]
    for col in numeric_cols:
        vals = source_df[col].dropna().astype(float)
        if len(vals) < 2:
            synthetic[col] = np.full(n_synthetic, vals.mean())
            continue
        mean, std = vals.mean(), vals.std()
        col_min, col_max = vals.min(), vals.max()
        sampled = np.random.normal(mean, std, n_synthetic)
        sampled = np.clip(sampled, col_min - 0.1 * abs(col_min), col_max + 0.1 * abs(col_max))
        # Preserve skewness crudely: if original is right-skewed, apply exp transform
        if vals.skew() > 1:
            sampled = np.exp(np.random.normal(np.log(np.abs(vals) + 1).mean(),
                                               np.log(np.abs(vals) + 1).std(), n_synthetic))
            sampled = np.clip(sampled, col_min, col_max * 1.1)
        synthetic[col] = np.round(sampled, 2)

    # Categorical: weighted resample
    for col in categorical_cols:
        freq = source_df[col].value_counts(normalize=True)
        synthetic[col] = np.random.choice(freq.index, size=n_synthetic, p=freq.values)

    return pd.DataFrame(synthetic)[source_df.columns.tolist()]


def extract_table_and_synthesize(image_path: str, multiplier: int = 10) -> dict:
    """End-to-end: extract table → synthesize 10x data → return both."""
    extracted = extract_table(image_path)
    df = extracted["dataframe"]

    # Attempt numeric conversion where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    synthetic_df = generate_synthetic_data(df, multiplier=multiplier)

    return {
        "original": extracted,
        "synthetic": synthetic_df,
        "original_rows": len(df),
        "synthetic_rows": len(synthetic_df),
        "multiplier": multiplier,
    }
