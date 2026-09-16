"""Strict OCR text normalization and dependency-free error metrics."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence


def normalize_for_evaluation(text: str) -> str:
    """Apply the experiment's only allowed text normalization: Unicode NFC."""
    return unicodedata.normalize("NFC", str(text))


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    """Return Levenshtein distance using memory linear in the shorter sequence."""
    if len(reference) < len(hypothesis):
        hypothesis, reference = reference, hypothesis
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (reference_item != hypothesis_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[object], hypothesis: Sequence[object]) -> float:
    edits = edit_distance(reference, hypothesis)
    if not reference:
        return 0.0 if edits == 0 else 1.0
    return edits / len(reference)


def character_error_rate(reference: str, hypothesis: str) -> float:
    return error_rate(normalize_for_evaluation(reference), normalize_for_evaluation(hypothesis))


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_for_evaluation(reference).split()
    hypothesis_words = normalize_for_evaluation(hypothesis).split()
    return error_rate(reference_words, hypothesis_words)


def corpus_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """Compute corpus CER/WER and exact-line accuracy from aligned strings."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one aligned prediction is required")

    normalized = [
        (normalize_for_evaluation(reference), normalize_for_evaluation(hypothesis))
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    character_edits = sum(
        edit_distance(reference, hypothesis) for reference, hypothesis in normalized
    )
    character_count = sum(len(reference) for reference, _ in normalized)
    word_pairs = [(reference.split(), hypothesis.split()) for reference, hypothesis in normalized]
    word_edits = sum(edit_distance(reference, hypothesis) for reference, hypothesis in word_pairs)
    word_count = sum(len(reference) for reference, _ in word_pairs)
    exact_count = sum(reference == hypothesis for reference, hypothesis in normalized)

    return {
        "cer": character_edits / max(1, character_count),
        "wer": word_edits / max(1, word_count),
        "exact_line_accuracy": exact_count / len(normalized),
    }
