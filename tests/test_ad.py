from __future__ import annotations

import pytest

from autodiff import array, tensor, jvp, vjp
from tests.utils import scalar


def test_jvp_requires_directions_when_not_push_forward() -> None:
    x = scalar(2.0)
    y = x * x
    with pytest.raises(ValueError):
        _ = jvp(y, inputs={x: x.arr}, directions=None, push_forward=False)


def test_jvp_push_forward_requires_directions_none() -> None:
    x = scalar(2.0)
    y = x * x
    with pytest.raises(ValueError):
        _ = jvp(y, inputs={x: x.arr}, directions={x: array(1.0)}, push_forward=True)


def test_jvp_push_forward_returns_callable() -> None:
    x = scalar(2.0)
    y = x * x
    fn = jvp(y, inputs={x: x.arr}, directions=None, push_forward=True)
    out = fn({x: array(1.0)})
    assert out.value == [4.0]  # 2x * 1


def test_jvp_missing_root_key_raises_keyerror() -> None:
    x = scalar(2.0)
    y = x * x
    fn = jvp(y, inputs={x: x.arr}, directions=None, push_forward=True)
    with pytest.raises(KeyError):
        _ = fn({})  # missing x


def test_vjp_pullback_requires_cotangent_none() -> None:
    x = scalar(2.0)
    y = x * x
    with pytest.raises(ValueError):
        _ = vjp(y, inputs={x: x.arr}, cotangent=array(1.0), pull_back=True)


def test_vjp_pullback_returns_callable() -> None:
    x = scalar(3.0)
    y = x * x
    pb = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=True)
    grads = pb(None)
    assert grads[x].value == [6.0]


def test_vjp_cotangent_shape_mismatch_raises() -> None:
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x @ x.transpose()
    # y shape (2,2), provide cotangent (2,)
    with pytest.raises(ValueError):
        _ = vjp(y, inputs={x: x.arr}, cotangent=array([1.0, 1.0]), pull_back=False)
