import pytest

from vlm_handwriting.smoke import smoke_indices


def test_frozen_validation_smoke_indices() -> None:
    assert smoke_indices(682) == [
        54, 90, 118, 145, 158, 176, 181, 211, 213, 220,
        265, 292, 302, 321, 335, 355, 375, 388, 514, 580,
    ]


def test_smoke_size_is_validated() -> None:
    with pytest.raises(ValueError):
        smoke_indices(3, size=4)
