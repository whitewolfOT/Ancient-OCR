"""Tests for preprocessing/seg_filter.py — segmentation noise filtering."""

from __future__ import annotations

from preprocessing.seg_filter import LineFilterConfig, filter_lines


def _line(line_id="line_000", x=10, y=10, w=200, h=30):
    return {"id": line_id, "bbox": [x, y, w, h]}


def test_removes_short_lines():
    lines = [_line(h=5)]
    assert filter_lines(lines) == []


def test_removes_narrow_lines():
    lines = [_line(w=30)]
    assert filter_lines(lines) == []


def test_keeps_valid_lines():
    lines = [_line(w=200, h=30)]
    result = filter_lines(lines)
    assert len(result) == 1
    assert result[0]["id"] == "line_000"


def test_manual_lines_bypass_filter():
    lines = [_line(line_id="manual_001", w=5, h=5)]
    result = filter_lines(lines)
    assert len(result) == 1
    assert result[0]["id"] == "manual_001"


def test_max_lines_cap():
    lines = [_line(line_id=f"line_{i:03d}", y=i * 40) for i in range(80)]
    cfg = LineFilterConfig(max_lines_per_page=60)
    result = filter_lines(lines, cfg)
    assert len(result) == 60


def test_empty_input_returns_empty():
    assert filter_lines([]) == []


def test_malformed_line_does_not_raise():
    lines = [{"id": "broken"}, _line()]
    result = filter_lines(lines)
    assert len(result) == 1
    assert result[0]["id"] == "line_000"


def test_dict_shaped_bbox_supported():
    """generate_line_crops.py uses {'x','y','w','h'} bbox dicts, not lists."""
    lines = [{"id": "line_000", "bbox": {"x": 10, "y": 10, "w": 200, "h": 30}}]
    assert len(filter_lines(lines)) == 1
    lines_noise = [{"id": "line_001", "bbox": {"x": 10, "y": 10, "w": 200, "h": 3}}]
    assert filter_lines(lines_noise) == []
