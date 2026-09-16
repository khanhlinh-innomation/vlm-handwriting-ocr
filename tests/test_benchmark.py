import pytest

from vlm_handwriting.benchmark import (
    evaluation_manifest_name,
    latency_summary,
    select_benchmark_rows,
)


def _rows(count: int) -> list[dict[str, str]]:
    return [{"relative_path": f"{index}.jpg"} for index in range(count)]


def test_one_and_smoke_use_the_same_frozen_selection() -> None:
    validation = _rows(682)
    test = _rows(201)
    one, one_split, one_tag = select_benchmark_rows(validation, test, "one")
    smoke, smoke_split, smoke_tag = select_benchmark_rows(validation, test, "smoke")
    assert one == smoke[:1]
    assert one_split == "validation_one"
    assert one_tag == "one"
    assert len(smoke) == 20
    assert smoke_split == "validation_smoke20"
    assert smoke_tag == "smoke20"


def test_test_mode_is_explicitly_gated() -> None:
    validation = _rows(682)
    test = _rows(201)
    with pytest.raises(PermissionError, match="test set"):
        select_benchmark_rows(validation, test, "test")

    selected_test, _, _ = select_benchmark_rows(validation, test, "test", allow_test=True)
    assert selected_test == test


def test_latency_summary_uses_nearest_rank_p90() -> None:
    assert latency_summary([1.0, 2.0, 3.0, 4.0, 5.0]) == {
        "latency_mean_sec": 3.0,
        "latency_p50_sec": 3.0,
        "latency_p90_sec": 5.0,
    }


def test_full_validation_does_not_require_test_access() -> None:
    assert evaluation_manifest_name("validation") == ("val.csv", "validation")
    with pytest.raises(PermissionError, match="test set"):
        evaluation_manifest_name("test")
    assert evaluation_manifest_name("test", allow_test=True) == ("test.csv", "test")
