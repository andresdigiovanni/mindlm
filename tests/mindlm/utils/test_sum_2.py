import pytest

from mindlm.utils.sum_2 import sum_2


def test_sum_2_positive_numbers():
    assert sum_2(2, 3) == 5


def test_sum_2_negative_numbers():
    assert sum_2(-2, -3) == -5


def test_sum_2_mixed_numbers():
    assert sum_2(-2, 3) == 1
    assert sum_2(2, -3) == -1


def test_sum_2_with_zero():
    assert sum_2(0, 5) == 5
    assert sum_2(5, 0) == 5
    assert sum_2(0, 0) == 0


def test_sum_2_floats():
    assert sum_2(2.5, 3.5) == 6.0
    assert sum_2(1.1, 2.2) == pytest.approx(3.3)
