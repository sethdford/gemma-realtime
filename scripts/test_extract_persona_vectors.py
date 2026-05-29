#!/usr/bin/env python3
"""Tests for extract_persona_vectors.py pure helpers (no model needed)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import extract_persona_vectors as e


def test_normalize_unit_length():
    v = e.normalize_direction([3.0, 4.0])  # 3-4-5 triangle
    assert abs(v[0] - 0.6) < 1e-9 and abs(v[1] - 0.8) < 1e-9
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9


def test_normalize_zero_vector_is_zeros():
    assert e.normalize_direction([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_directions_mean_difference():
    # one layer, hidden=2. high mean=(2,0), low mean=(0,0) -> dir=(1,0)
    high = [[4.0, 0.0]]  # sum over 2 samples
    low = [[0.0, 0.0]]
    out = e.directions_from_sums(high, 2, low, 2)
    assert len(out) == 1
    assert abs(out[0][0] - 1.0) < 1e-9 and abs(out[0][1]) < 1e-9


def test_directions_opposite_sign():
    # high mean below low mean -> negative direction component
    out = e.directions_from_sums([[0.0]], 1, [[5.0]], 1)
    assert out[0][0] < 0


def test_directions_zero_count_raises():
    try:
        e.directions_from_sums([[1.0]], 0, [[0.0]], 1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_trait_pairs_well_formed():
    assert set(e.TRAIT_PAIRS) == {"formality", "verbosity", "warmth", "humor"}
    for trait, pairs in e.TRAIT_PAIRS.items():
        assert len(pairs) >= 8, f"{trait} too few pairs"
        for hi, lo in pairs:
            assert hi and lo and hi != lo


def main():
    tests = [test_normalize_unit_length, test_normalize_zero_vector_is_zeros,
             test_directions_mean_difference, test_directions_opposite_sign,
             test_directions_zero_count_raises, test_trait_pairs_well_formed]
    print("Testing extract_persona_vectors.py")
    print("=" * 60)
    p = f = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            p += 1
        except Exception as ex:  # noqa: BLE001
            print(f"✗ {t.__name__}: {type(ex).__name__}: {ex}")
            f += 1
    print("=" * 60)
    print(f"Results: {p} passed, {f} failed")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
