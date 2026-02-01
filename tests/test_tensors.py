from __future__ import annotations

import math
import pytest
import autodiff
from utils import nested_map, assert_close, scalar
from autodiff import array, tensor, jvp, vjp


# -------------------------
# Forward semantics
# -------------------------

def test_forward_add_scalar() -> None:
    x = scalar(2.0)
    y = scalar(3.0)
    z = x + y
    assert z.arr.value == [5.0]


# -------------------------
# Shape semantics
# -------------------------

def test_array_flatten_shape() -> None:
    a = array([[1.0, 2.0], [3.0, 4.0]])
    f = a.flatten()
    assert f.shape == (4,)
    assert f.value == [1.0, 2.0, 3.0, 4.0]


# -------------------------
# Error semantics
# -------------------------

def test_matmul_shape_error() -> None:
    a = array([[1.0, 2.0]])          # shape (1,2)
    b = array([[1.0, 2.0]])          # shape (1,2) inner dims mismatch for @
    with pytest.raises(ValueError):
        _ = a @ b


# -------------------------
# Differentiation semantics
# -------------------------

def test_vjp_scalar_square() -> None:
    x = scalar(3.0)
    y = x * x  # y = x^2
    grads = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=False)
    # dy/dx = 2x = 6
    assert grads[x].value == [6.0]


def test_jvp_scalar_square() -> None:
    x = scalar(3.0)
    y = x * x

    # direction v = 1 => JVP = 2x * 1 = 6
    tan = jvp(y, inputs={x: x.arr}, directions={x: array(1.0)}, push_forward=False)
    assert tan.value == [6.0]


def test_vjp_pullback_closure() -> None:
    x = scalar(2.0)
    y = x * x

    pb = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=True)
    grads = pb(None)  # None => cotangent=ones
    assert grads[x].value == [4.0]
