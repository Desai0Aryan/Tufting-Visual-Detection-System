import numpy as np

from src.duct_fit import band_mask, surround_check

E = ((500.0, 400.0), (300.0, 260.0), 10.0)


def ring(n, r=1.3, start=0.0, span=2 * np.pi):
    a = start + np.linspace(0, span, n, endpoint=False)
    return np.c_[500 + r * 150 * np.cos(a), 400 + r * 130 * np.sin(a)]


def test_boxes_all_round_pass():
    ok, n = surround_check(E, ring(16))
    assert ok and n == 8


def test_boxes_on_one_side_fail():
    ok, n = surround_check(E, ring(16, span=np.pi / 2))
    assert not ok and n <= 3


def test_boxes_far_away_or_inside_do_not_count():
    ok, n = surround_check(E, np.vstack([ring(16, r=3.0), ring(16, r=0.5)]))
    assert not ok and n == 0


def test_band_is_between_inner_and_outer():
    m = band_mask((800, 1000), E, 0.8, 1.0)
    assert m[400, 500] == 0                 # centre
    assert m[400, 500 + 135] == 255         # 0.9 R along the (almost) major axis
    assert m[400, 500 + 200] == 0           # outside
