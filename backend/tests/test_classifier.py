"""Classifier tests on synthetic PDFs built in memory."""

from __future__ import annotations

import pymupdf
import pytest
from pdf_builders import ENGLISH, add_digital, add_scanned
from pdf_builders import gibberish as _gibberish

from ocr_pipeline.classify import classify_document, classify_page, decide
from ocr_pipeline.classify.quality import quality_score
from ocr_pipeline.config import ClassifierSettings, Settings, load_settings
from ocr_pipeline.models import ClassificationReason as Reason
from ocr_pipeline.models import PageSignals, PageType


@pytest.fixture
def settings() -> ClassifierSettings:
    return ClassifierSettings()


@pytest.fixture
def doc():
    with pymupdf.open() as d:
        yield d


# --- page types -----------------------------------------------------------------


def test_digital_page_with_good_text_needs_no_ocr(doc, settings):
    add_digital(doc)
    c = classify_page(doc[0], settings)
    assert c.page_type == PageType.DIGITAL
    assert c.reason == Reason.TEXT_LAYER_OK
    assert not c.needs_ocr
    assert c.quality_score is not None and c.quality_score > 0.9


def test_scanned_page_without_text_layer_needs_ocr(doc, settings):
    add_scanned(doc)
    c = classify_page(doc[0], settings)
    assert c.page_type == PageType.SCANNED
    assert c.reason == Reason.NO_TEXT_LAYER
    assert c.needs_ocr
    assert c.signals.image_coverage > 0.99


def test_blank_page_is_empty(doc, settings):
    doc.new_page()
    c = classify_page(doc[0], settings)
    assert c.page_type == PageType.EMPTY
    assert c.reason == Reason.EMPTY_PAGE
    assert not c.needs_ocr


def test_digital_page_with_garbage_text_layer_needs_ocr(doc, settings):
    add_digital(doc, _gibberish())
    c = classify_page(doc[0], settings)
    assert c.page_type == PageType.DIGITAL
    assert c.reason == Reason.TEXT_LAYER_LOW_QUALITY
    assert c.needs_ocr


def test_scanned_page_with_good_ocr_layer_is_kept(doc, settings):
    add_scanned(doc, ocr_layer=ENGLISH)
    c = classify_page(doc[0], settings)
    assert c.signals.has_ocr_layer
    assert c.page_type == PageType.SCANNED
    assert c.reason == Reason.OCR_LAYER_OK
    assert not c.needs_ocr


def test_scanned_page_with_bad_ocr_layer_is_reocred(doc, settings):
    add_scanned(doc, ocr_layer=_gibberish())
    c = classify_page(doc[0], settings)
    assert c.signals.has_ocr_layer
    assert c.reason == Reason.OCR_LAYER_LOW_QUALITY
    assert c.needs_ocr


def test_scan_with_small_visible_stamp_is_scanned(doc, settings):
    add_scanned(doc, visible_text="Received 2026-11-12")
    c = classify_page(doc[0], settings)
    assert c.page_type == PageType.SCANNED
    assert c.reason == Reason.IMAGE_DOMINANT_SPARSE_TEXT
    assert c.needs_ocr


# --- threshold is user-controlled --------------------------------------------------


def test_lower_threshold_accepts_garbage_text(doc):
    add_digital(doc, _gibberish())
    c = classify_page(doc[0], ClassifierSettings(quality_threshold=0.2))
    assert c.reason == Reason.TEXT_LAYER_OK
    assert not c.needs_ocr
    assert c.threshold == 0.2


def test_ui_override_via_model_copy(doc):
    add_digital(doc, _gibberish())
    base = Settings().classifier
    strict = base.model_copy(update={"quality_threshold": 0.2})
    assert classify_page(doc[0], base).needs_ocr
    assert not classify_page(doc[0], strict).needs_ocr


# --- whole document --------------------------------------------------------------


def test_mixed_document_is_classified_per_page(tmp_path, settings):
    path = tmp_path / "mixed.pdf"
    with pymupdf.open() as d:
        add_digital(d)
        add_scanned(d)
        d.new_page()
        add_scanned(d, ocr_layer=ENGLISH)
        d.save(path)

    results = classify_document(path, settings)
    assert [r.page_number for r in results] == [1, 2, 3, 4]
    assert [r.page_type for r in results] == [
        PageType.DIGITAL,
        PageType.SCANNED,
        PageType.EMPTY,
        PageType.SCANNED,
    ]
    assert [r.needs_ocr for r in results] == [False, True, False, False]


# --- scoring (pure function) ------------------------------------------------------


def _signals(**overrides) -> PageSignals:
    base = dict(
        char_count=500,
        ocr_char_count=0,
        word_count=80,
        bad_char_ratio=0.0,
        dictionary_ratio=1.0,
        image_coverage=0.0,
        curve_vector_count=0,
        has_ocr_layer=False,
        ocr_model_probability=None,
    )
    return PageSignals(**(base | overrides))


def test_bad_characters_lower_the_score(settings):
    clean, _ = quality_score(_signals(), settings)
    dirty, components = quality_score(_signals(bad_char_ratio=0.2), settings)
    assert clean == pytest.approx(1.0)
    assert components["clean_chars"] == pytest.approx(0.0)
    assert dirty < clean


def test_too_many_bad_characters_force_ocr_even_with_good_words(settings):
    # Partly broken font encoding: remaining words are fine, score passes, gate fails it.
    c = decide(1, _signals(bad_char_ratio=0.2), settings)
    assert c.quality_score is not None and c.quality_score >= settings.quality_threshold
    assert c.reason == Reason.TEXT_LAYER_LOW_QUALITY
    assert c.needs_ocr


def test_few_bad_characters_are_tolerated(settings):
    c = decide(1, _signals(bad_char_ratio=0.02), settings)
    assert c.reason == Reason.TEXT_LAYER_OK
    assert not c.needs_ocr


def test_missing_components_are_dropped_and_weights_renormalised(settings):
    score, components = quality_score(_signals(dictionary_ratio=None), settings)
    assert set(components) == {"clean_chars"}
    assert score == pytest.approx(1.0)


def test_ocr_model_probability_is_used_when_present(settings):
    score, components = quality_score(_signals(ocr_model_probability=1.0), settings)
    assert components["image_text"] == pytest.approx(0.0)
    assert score == pytest.approx(0.8)  # (0.5*1 + 0.3*1 + 0.2*0) / 1.0


# --- settings ----------------------------------------------------------------------


def test_load_settings_from_toml(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text('[classifier]\nquality_threshold = 0.75\n[ocr]\nlanguage = "eng"\n')
    s = load_settings(cfg)
    assert s.classifier.quality_threshold == 0.75
    assert s.classifier.weights.dictionary == 0.5  # untouched default


def test_repo_default_config_is_valid():
    from pathlib import Path

    s = load_settings(Path(__file__).parents[1] / "config" / "default.toml")
    assert s.ocr.tessdata_path is not None


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError):
        ClassifierSettings(weights={"dictionary": 0, "clean_chars": 0, "image_text": 0})
