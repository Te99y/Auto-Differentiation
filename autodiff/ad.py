from __future__ import annotations

from typing import overload, Callable, Mapping, Literal

from .array import array
from .tensor import tensor

JVPFn = Callable[[Mapping[tensor, array]], array]
VJPFn = Callable[[array | None], dict[tensor, array]]


@overload
def jvp(
    f: tensor,
    inputs: Mapping[tensor, array] | None,
    directions: None,
    *,
    push_forward: Literal[True],
) -> JVPFn: ...


@overload
def jvp(
    f: tensor,
    inputs: Mapping[tensor, array] | None,
    directions: Mapping[tensor, array],
    *,
    push_forward: Literal[False] = False,
) -> array: ...


def jvp(
    f: tensor,
    inputs: Mapping[tensor, array] | None,
    directions: Mapping[tensor, array] | None,
    *,
    push_forward: bool = False,
) -> array | JVPFn:
    """
    Compute the Jacobian–vector product (JVP) for a computation graph.

    Parameters
    ----------
    f : tensor
        Output tensor (root of the computation graph).
    inputs : Mapping[tensor, array] or None
        If provided, assigns primal values to root tensors before propagation.
    directions : Mapping[tensor, array] or None
        Tangent vectors for root tensors. Required unless push_forward=True.
    push_forward : bool, default=False
        If True, return a function that accepts directions and computes the JVP.

    Returns
    -------
    array or Callable[[Mapping[tensor, array]], array]
        If push_forward=False, returns the tangent (JVP) at f.
        If push_forward=True, returns a callable that maps directions to the JVP.

    Raises
    ------
    ValueError
        If arguments are inconsistent with push_forward.
    """
    if push_forward and directions is not None:
        raise ValueError("directions must be None when push_forward=True")
    if (not push_forward) and directions is None:
        raise ValueError("directions must be provided when push_forward=False")

    visited: set[tensor] = set()
    order: list[tensor] = []
    roots: list[tensor] = []
    f.topo(visited, order, roots)

    if inputs is not None:
        for root in roots:
            root.arr = array(inputs[root])
        for t in order:
            t._prop_val()

    if push_forward:
        def f_jvp(_directions: Mapping[tensor, array]) -> array:
            for root in roots:
                root.tangent = array(_directions[root])
            for t in order:
                t._prop_tan()
            return f.tangent
        return f_jvp

    # directions is not None here
    for root in roots:
        root.tangent = array(directions[root])
    for t in order:
        t._prop_tan()
    return f.tangent


@overload
def vjp(
    f: tensor,
    inputs: Mapping[tensor, array] | None,
    cotangent: array | None,
    *,
    pull_back: Literal[False] = False,
) -> dict[tensor, array]: ...


@overload
def vjp(
    f: tensor,
    inputs: Mapping[tensor, array] | None,
    cotangent: None,
    *,
    pull_back: Literal[True],
) -> VJPFn: ...


def vjp(
    f: tensor,
    inputs: Mapping[tensor, array] | None,
    cotangent: array | None,
    *,
    pull_back: bool = False,
) -> dict[tensor, array] | VJPFn:
    """
    Compute the vector–Jacobian product (VJP) for a computation graph.

    Parameters
    ----------
    f : tensor
        Output tensor (root of the computation graph).
    inputs : Mapping[tensor, array] or None
        If provided, assigns primal values to root tensors before propagation.
    cotangent : array or None
        Output cotangent with the same shape as f. If None, uses ones.
        Must be None when pull_back=True.
    pull_back : bool, default=False
        If True, return a function that maps cotangents to gradients at the roots.

    Returns
    -------
    dict[tensor, array] or Callable[[array | None], dict[tensor, array]]
        If pull_back=False, returns the gradients w.r.t. the roots for the given cotangent.
        If pull_back=True, returns a closure that accepts a cotangent (or None for ones)
        and returns gradients w.r.t. roots.

    Raises
    ------
    ValueError
        If arguments are inconsistent with pull_back, or if cotangent shape mismatches f.
    """
    if pull_back and cotangent is not None:
        raise ValueError("cotangent must be None when pull_back=True")
    if cotangent is not None and cotangent.shape != f.shape:
        raise ValueError(f'Cotangent shape {cotangent.shape} does not match output shape {f.shape}.')

    visited: set[tensor] = set()
    order: list[tensor] = []
    roots: list[tensor] = []
    f.topo(visited, order, roots)

    if inputs is not None:
        for root in roots:
            root.arr = array(inputs[root])
        for t in order:
            t._prop_val()

    if pull_back:
        def f_vjp(_cotangent: array | None) -> dict[tensor, array]:
            if _cotangent is not None and _cotangent.shape != f.shape:
                raise ValueError("Cotangent shape does not match output shape.")
            f.gradient = array(1.0, outer_shape=f.shape) if _cotangent is None else _cotangent
            for t in reversed(order):
                t._prop_grad()
            return {root: root.gradient for root in roots}
        return f_vjp

    f.gradient = array(1.0, outer_shape=f.shape) if cotangent is None else cotangent
    for t in reversed(order):
        t._prop_grad()
        if not t.is_leaf:
            t.gradient = array(0.0)

    return {root: root.gradient for root in roots}
