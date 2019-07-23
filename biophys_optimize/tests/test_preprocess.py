import pytest
import biophys_optimize.preprocess as preprocess
import numpy as np


def test_passive_fit_window_inputs():
    grand_up = np.zeros(3)
    grand_down_1 = np.zeros(4)
    grand_down_2 = np.zeros(3)
    t = np.zeros(5)

    pytest.raises(ValueError, preprocess.passive_fit_window,grand_up, grand_down_1, t)
    pytest.raises(ValueError, preprocess.passive_fit_window, grand_up, grand_down_2, t)


def test_passive_fit_window_zero_window():
    n_points = 500
    grand_up = np.ones(n_points)
    grand_down = np.ones(n_points) * 5
    t = np.arange(n_points)
    pytest.raises(RuntimeError, preprocess.passive_fit_window,
        grand_up, grand_down, t)


def test_passive_fit_window_full_window():
    n_points = 500
    grand_up = np.ones(n_points)
    grand_down = -grand_up
    t = np.arange(n_points)
    assert preprocess.passive_fit_window(grand_up, grand_down, t) == t[-1]


def test_passive_fit_window_full_window():
    n_points = 500
    grand_up = np.ones(n_points)
    grand_down = -grand_up
    escape = 300
    grand_down[escape:] = grand_down[escape:] * 2
    t = np.arange(n_points)
    assert preprocess.passive_fit_window(grand_up, grand_down, t) >= escape
    assert preprocess.passive_fit_window(grand_up, grand_down, t) < t[-1]