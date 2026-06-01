"""Tests for ingest.region_cropper: crop_regions and stitch_results."""
import numpy as np

from ingest.region_cropper import crop_regions, stitch_results
from ocr_engine.schema import OCRResult, WordToken


def _white_image(h=200, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# crop_regions
# ---------------------------------------------------------------------------

def test_two_regions_produce_two_crops():
    img = _white_image(200, 400)
    regions = [
        {"x": 0, "y": 0, "w": 200, "h": 100, "type": "text_block", "region_id": "r0"},
        {"x": 200, "y": 100, "w": 200, "h": 100, "type": "text_block", "region_id": "r1"},
    ]
    crops = crop_regions(img, regions)
    assert len(crops) == 2


def test_crop_shapes_are_correct():
    img = _white_image(200, 400)
    regions = [
        {"x": 0, "y": 0, "w": 100, "h": 50, "type": "text_block", "region_id": "r0"},
        {"x": 100, "y": 50, "w": 150, "h": 80, "type": "text_block", "region_id": "r1"},
    ]
    crops = crop_regions(img, regions)
    assert crops[0]["image"].shape[:2] == (50, 100)
    assert crops[1]["image"].shape[:2] == (80, 150)


def test_empty_regions_returns_one_full_page_crop():
    img = _white_image(200, 400)
    crops = crop_regions(img, [])
    assert len(crops) == 1
    assert crops[0]["region_id"] == "full"
    assert crops[0]["image"].shape[:2] == (200, 400)


def test_crop_is_copy_not_view():
    img = _white_image(200, 400)
    crops = crop_regions(img, [{"x": 0, "y": 0, "w": 50, "h": 50,
                                 "type": "text_block", "region_id": "r0"}])
    crop = crops[0]["image"]
    # Mutate original; crop should be unaffected
    img[0, 0] = [0, 0, 0]
    assert crop[0, 0, 0] == 255


def test_crop_bbox_in_page_matches_region():
    img = _white_image(200, 400)
    regions = [{"x": 10, "y": 20, "w": 80, "h": 60,
                 "type": "text_block", "region_id": "r0"}]
    crops = crop_regions(img, regions)
    assert crops[0]["bbox_in_page"] == (10, 20, 80, 60)


def test_region_clamped_to_page_boundary():
    img = _white_image(100, 100)
    # Region extends beyond page right edge
    regions = [{"x": 80, "y": 0, "w": 100, "h": 50,
                 "type": "text_block", "region_id": "r0"}]
    crops = crop_regions(img, regions)
    # Should clamp w to 20 (100 - 80)
    assert len(crops) == 1
    assert crops[0]["image"].shape[1] == 20


# ---------------------------------------------------------------------------
# stitch_results
# ---------------------------------------------------------------------------

def _make_token(text="كتاب", x=10, y=20, w=30, h=15, conf=0.8, source="paddle"):
    return WordToken(
        text=text, confidence=conf,
        bbox=(x, y, w, h),
        page_index=0, source=source,
    )


def _make_result(tokens, x_off=0, y_off=0, conf=0.8):
    text = " ".join(t.text for t in tokens)
    return OCRResult(
        text=text,
        words=tokens,
        confidence=conf,
        page_index=0,
        source="paddle",
        raw={"crop_bbox_in_page": [x_off, y_off, 0, 0]},
    )


def test_stitch_two_regions_concatenates_words():
    tok0 = _make_token("كتاب", x=5, y=5, w=20, h=10)
    tok1 = _make_token("علم", x=5, y=5, w=20, h=10)
    r0 = _make_result([tok0], x_off=0, y_off=0)
    r1 = _make_result([tok1], x_off=200, y_off=100)
    stitched = stitch_results([r0, r1], page_index=0)
    assert len(stitched.words) == 2


def test_stitch_bbox_translation_adds_offset():
    tok = _make_token("كتاب", x=5, y=5, w=20, h=10)
    result = _make_result([tok], x_off=100, y_off=50)
    stitched = stitch_results([result], page_index=0)
    assert stitched.words[0].bbox == (105, 55, 20, 10)


def test_stitch_empty_returns_empty_ocr_result():
    stitched = stitch_results([], page_index=1)
    assert stitched.text == ""
    assert stitched.words == []
    assert stitched.confidence == 0.0
    assert stitched.page_index == 1


def test_stitch_source_is_ensemble():
    tok = _make_token()
    result = _make_result([tok])
    stitched = stitch_results([result], page_index=0)
    assert stitched.source == "ensemble"


def test_stitch_no_offset_keeps_original_bbox():
    tok = _make_token(x=10, y=20, w=30, h=15)
    result = _make_result([tok], x_off=0, y_off=0)
    stitched = stitch_results([result], page_index=0)
    assert stitched.words[0].bbox == (10, 20, 30, 15)
