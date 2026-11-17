"""Unit tests for src/equivalence_main.py pure helpers."""
from __future__ import annotations


import numpy as np

from src.equivalence_main import canonical_labels, select_rows


def test_canonical_labels_is_partition_invariant():
    a = canonical_labels(np.array([7, 7, -1, 3, 3, 7]))
    b = canonical_labels(np.array([1, 1, -1, 0, 0, 1]))
    assert a.tolist() == [0, 0, -1, 1, 1, 0]
    assert a.tolist() == b.tolist()
    assert canonical_labels(np.array([-1, -1])).tolist() == [-1, -1]


def test_select_rows_caps_deterministically():
    assert select_rows(5, 0, 1).tolist() == [0, 1, 2, 3, 4]
    assert select_rows(5, 10, 1).tolist() == [0, 1, 2, 3, 4]
    picked = select_rows(1000, 10, 17)
    assert len(picked) == 10 and picked.tolist() == sorted(picked.tolist())
    assert picked.tolist() == select_rows(1000, 10, 17).tolist()
