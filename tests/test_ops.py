from __future__ import annotations

import math

import numpy as np
import pytest

from autodiff import tensor, add, sub, mul, div, neg, abs_, exp, log, sin, cos, vjp
from tests.utils import scalar, assert_close


def finite_diff_scalar(f, x: float, eps: float = 1e-6) -> float:
    # Central difference: O(eps^2) truncation error
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


@pytest.mark.parametrize("xv,yv", [(2.0, 3.0), (-1.5, 0.25), (0.1, -0.2)])
def test_ops_scalar_forward_matches_python(xv: float, yv: float) -> None:
    x = tensor(xv)
    y = tensor(yv)

    assert (add(x, y)).arr.value == [xv + yv]
    assert (sub(x, y)).arr.value == [xv - yv]
    assert (mul(x, y)).arr.value == [xv * yv]
    assert (div(x, y)).arr.value == [xv / yv]


def test_ops_unary_forward_matches_math() -> None:
    x = tensor(0.123)
    assert neg(x).arr.value == [-0.123]
    assert abs_(tensor(-0.5)).arr.value == [0.5]
    assert exp(x).arr.value == [math.exp(0.123)]
    assert log(tensor(1.5)).arr.value == [math.log(1.5)]
    assert sin(x).arr.value == [math.sin(0.123)]
    assert cos(x).arr.value == [math.cos(0.123)]


@pytest.mark.parametrize("xv", [0.2, 1.1, -0.7])
def test_vjp_matches_finite_diff_for_sin_square(xv: float) -> None:
    # f(x) = sin(x^2), f'(x) = cos(x^2) * 2x
    x = scalar(xv)
    y = sin(x * x)

    grads = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=False)
    g_ad = grads[x].value[0]

    def f_num(t: float) -> float:
        return math.sin(t * t)

    g_fd = finite_diff_scalar(f_num, xv)

    assert abs(g_ad - g_fd) < 1e-4


@pytest.mark.parametrize("xv", [0.2, 1.1, 3.0])
def test_vjp_matches_closed_form_for_exp_log(xv: float) -> None:
    # f(x) = log(exp(x)) = x, derivative 1 (for all real x)
    x = scalar(xv)
    y = log(exp(x))
    grads = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=False)
    assert abs(grads[x].value[0] - 1.0) < 1e-10


def test_ops_array_forward_matches_numpy() -> None:
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[10.0, 20.0], [30.0, 40.0]])
    an = np.array(a.arr.value, dtype=float)
    bn = np.array(b.arr.value, dtype=float)

    assert_close((a + b), tensor((an + bn).tolist()))
    assert_close((a - b), tensor((an - bn).tolist()))
    assert_close((a * b), tensor((an * bn).tolist()))
    assert_close((a / b), tensor((an / bn).tolist()))
