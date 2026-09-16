import unicodedata

import pytest

from vlm_handwriting.metrics import (
    character_error_rate,
    corpus_metrics,
    normalize_for_evaluation,
    word_error_rate,
)


def test_normalization_is_nfc_only() -> None:
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    assert normalize_for_evaluation(decomposed) == "Tiếng Việt"
    assert normalize_for_evaluation("Đông.") != normalize_for_evaluation("đông")
    assert normalize_for_evaluation("a  b") == "a  b"


def test_error_metrics() -> None:
    assert character_error_rate("abc", "axc") == pytest.approx(1 / 3)
    assert word_error_rate("xin chào bạn", "xin chào") == pytest.approx(1 / 3)
    assert corpus_metrics(["abc", "đúng"], ["axc", "đúng"]) == pytest.approx(
        {"cer": 1 / 7, "wer": 1 / 2, "exact_line_accuracy": 1 / 2}
    )


def test_corpus_metrics_rejects_unaligned_inputs() -> None:
    with pytest.raises(ValueError):
        corpus_metrics(["one"], [])
