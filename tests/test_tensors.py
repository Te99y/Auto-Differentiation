from __future__ import annotations

import math

import pytest

from autodiff import array, tensor, jvp, vjp
from tests.utils import scalar, nested_map, assert_close


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

def test_tensor_flatten_shape() -> None:
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    f = a.flatten()
    assert f.shape == (4,)
    assert f.arr.value == [1.0, 2.0, 3.0, 4.0]


# -------------------------
# Error semantics
# -------------------------

def test_matmul_shape_error() -> None:
    a = tensor([[1.0, 2.0]])          # shape (1,2)
    b = tensor([[1.0, 2.0]])          # shape (1,2) inner dims mismatch for @
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


@pytest.mark.parametrize("s", [3.0, [3.0], [1.0, 2.0, 3.0], [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]])
def test_jvp_scalar_unary_elementwise(s) -> None:
    x = tensor(s)

    # y = x*x, direction v = 1 => JVP = 2x * 1
    tan = jvp(x * x, inputs=None, directions={x: array(1.0)})
    assert tan.value == nested_map(lambda v: v * 2, s)

    # y = e^x, direction v = 1 => JVP = e^x * 1 = 6
    tan = jvp(x.exp(), inputs=None, directions={x: array(1.0)})
    assert tan.value == nested_map(lambda v: math.exp(v), s)


@pytest.mark.parametrize(
    "s1, s2",
    [(3.0, 4.1),
     ([3.0], [4.1]),
     ([1.0, 2.0, 3.0], [4.1, 5.2, 6.3]),
     ([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], [[4.1, 5.2, 6.3], [-4.1, -5.2, -6.3]])]
)
def test_jvp_scalar_binary_elementwise(s1, s2) -> None:
    x1, x2 = tensor(s1), tensor(s2)

    # y = x1 * x2, direction v = 1 => JVP = x1 * 1 + x2 * 1
    tan = jvp(x1 * x2, inputs=None, directions={x1: array(1.0), x2: array(1.0)})
    assert tan.value == nested_map(lambda v1, v2: v1+v2, s1, s2)

    # y = x1 / x2, direction v = 1 => JVP = 1/x2 * 1 + -x1*x2^-2 * 1
    tan = jvp(x1 / x2, inputs=None, directions={x1: array(1.0), x2: array(1.0)})
    assert_close(tan.value, nested_map(lambda v1, v2: 1.0/v2 - v1/(v2*v2), s1, s2))


def test_jvp_1d_vector() -> None:
    x = tensor([1.0, 2.0, 3.0])
    y = x * x

    # direction v = 1 => JVP = 2x * 1 = 6
    tan = jvp(y, inputs={x: x.arr}, directions={x: array(1.0)})
    assert tan.value == [2.0, 4.0, 6.0]


def test_vjp_pullback_closure() -> None:
    x = scalar(2.0)
    y = x * x

    pb = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=True)
    grads = pb(None)  # None => cotangent=ones
    assert grads[x].value == [4.0]
