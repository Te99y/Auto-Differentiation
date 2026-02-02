from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from autodiff import array, vjp, jvp
from tests.utils import scalar


def test_compare_vjp_with_jax_grad_scalar() -> None:
    # f(x) = sin(x^2)
    x = scalar(1.234)
    y = (x * x).sin()

    grads = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=False)
    g_ad = grads[x].value[0]

    def f_jax(t):
        return jnp.sin(t * t)

    g_jax = jax.grad(f_jax)(1.234)
    assert abs(g_ad - float(g_jax)) < 1e-6


def test_compare_jvp_with_jax_jvp_scalar() -> None:
    x = scalar(0.7)
    y = (x * x).sin()

    tan = jvp(y, inputs={x: x.arr}, directions={x: array(1.0)}, push_forward=False)
    t_ad = tan.value[0]

    def f_jax(t):
        return jnp.sin(t * t)

    (_, t_jax) = jax.jvp(f_jax, (0.7,), (1.0,))
    assert abs(t_ad - float(t_jax)) < 1e-6
