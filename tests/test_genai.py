"""
GenAI tests — no API calls, no heavy models.
Tests cover: image encoding/compression, audio validation, synthetic data generation.
"""
import io
import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from src.genai.vision import (
    _compress_to_limit, _tile_image, _pil_to_b64,
    MAX_B64_BYTES, SUPPORTED_FORMATS, CONVERTIBLE_FORMATS, TILE_GRID,
)
from src.genai.audio import SUPPORTED_AUDIO
from src.genai.structured_vision import generate_synthetic_data, _extract_json


# ---------------------------------------------------------------------------
# Vision: image encoding and large-file handling
# ---------------------------------------------------------------------------

def _make_image(width: int, height: int) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_compress_to_limit_stays_within_api_limit():
    large_img = _make_image(3000, 3000)
    b64 = _compress_to_limit(large_img)
    assert len(b64) <= MAX_B64_BYTES


def test_tile_image_produces_correct_grid_count():
    img = _make_image(400, 400)
    tiles = _tile_image(img)
    expected = TILE_GRID[0] * TILE_GRID[1]
    assert len(tiles) == expected


def test_supported_and_convertible_formats_are_disjoint():
    overlap = SUPPORTED_FORMATS & CONVERTIBLE_FORMATS
    assert len(overlap) == 0, f"Formats in both sets: {overlap}"


# ---------------------------------------------------------------------------
# Audio: format validation
# ---------------------------------------------------------------------------

def test_supported_audio_formats_include_common_types():
    for fmt in [".mp3", ".wav", ".m4a", ".flac"]:
        assert fmt in SUPPORTED_AUDIO


def test_transcribe_raises_on_unsupported_format(tmp_path):
    from src.genai.audio import transcribe
    fake = tmp_path / "test.xyz"
    fake.write_bytes(b"fake audio data")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        transcribe(str(fake))


# ---------------------------------------------------------------------------
# Structured vision: synthetic data generation
# ---------------------------------------------------------------------------

def test_synthetic_data_has_correct_row_count():
    df = pd.DataFrame({"price": [100.0, 110.0, 95.0, 105.0], "volume": [1000, 1200, 800, 1100]})
    synthetic = generate_synthetic_data(df, multiplier=10)
    assert len(synthetic) == len(df) * 10


def test_synthetic_data_preserves_numeric_column_range():
    df = pd.DataFrame({"price": [50.0, 60.0, 55.0, 58.0, 52.0]})
    synthetic = generate_synthetic_data(df, multiplier=5)
    original_min, original_max = df["price"].min(), df["price"].max()
    # Synthetic values should stay roughly within range (with ±10% tolerance)
    assert synthetic["price"].min() >= original_min * 0.8
    assert synthetic["price"].max() <= original_max * 1.2


def test_extract_json_handles_markdown_fences():
    raw = '```json\n{"headers": ["A", "B"], "rows": [[1, 2]]}\n```'
    result = _extract_json(raw)
    assert result["headers"] == ["A", "B"]
