from src.main import add_coords
import pytest


def test_add_coords_simple():
    assert add_coords((1, 2), (3, 4)) == (4, 6)


def test_add_coords_mismatch():
    with pytest.raises(ValueError):
        add_coords((1, 2), (3,))


def test_add_coords_none():
    with pytest.raises(ValueError):
        add_coords(None, (1, 2))
