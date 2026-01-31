from __future__ import annotations
import copy
import math
import itertools
import operator
from typing import Callable, Any

from .types import Number


def sign(v: Number) -> float:
    return 1.0 if v > 0 else -1.0 if v < 0 else 0.0


def depth(v: list | Number) -> int:
    cnt = 0
    while isinstance(v, list) and v:
        v = v[0]
        cnt += 1
    return cnt


def check_shape_list(v: list) -> tuple[int, ...]:
    if not v:  # v = []
        return (0, )
    if isinstance(v[0], list):
        return (len(v), ) + check_shape_list(v[0])
    return (len(v), )


def broadcast(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    res_shape = ()
    for d1, d2 in zip(reversed(shape1), reversed(shape2)):
        if d1 == d2 or d2 == 1:
            res_shape += (d1,)
        elif d1 == 1:
            res_shape += (d2,)
        else:
            raise TypeError(f'Cannot broadcast between {shape1} and {shape2}, mismatch at {d1} vs {d2}')
    if len(shape1) > len(shape2):
        return shape1[:-len(shape2)] + tuple(reversed(res_shape))
    return shape2[:-len(shape1)] + tuple(reversed(res_shape))


def unary_elementwise_list(v: list, op: Callable[[Number], Number]) -> list:
    if not isinstance(v[0], list):
        return [op(v_i) for v_i in v]
    return [unary_elementwise_list(v_i, op) for v_i in v]


def binary_elementwise_list(v1: list, v2: list, op: Callable[[Number, Number], Number]) -> list:
    if not isinstance(v1[0], list) and not isinstance(v2[0], list):
        return [op(v1[min(i, len(v1)-1)], v2[min(i, len(v2)-1)]) for i in range(max(len(v1), len(v2)))]
    d1, d2 = depth(v1), depth(v2)
    if d1 == d2:
        return [binary_elementwise_list(v1[min(i, len(v1)-1)], v2[min(i, len(v2)-1)], op)
                for i in range(max(len(v1), len(v2)))]
    if d1 > d2:
        return [binary_elementwise_list(v1_i, v2, op) for v1_i in v1]
    return [binary_elementwise_list(v1, v2_i, op) for v2_i in v2]


def flatten_list(v: list, layers: int = -1) -> list[Number]:
    """
    Flatten an nd-list into a 1D list.
    Does not validate homogeneity.
    """
    shape = check_shape_list(v)
    if len(shape) == 1 and (layers == 0 or layers == -1):  # v already is 1D
        return v

    if len(shape) <= abs(layers):
        raise ValueError('depth out of bound; depth must be less then the length of the shape')

    res: list[Number] = []
    for idx in itertools.product(*(range(s) for s in shape[:layers])):
        src = v
        for i in idx:
            src = src[i]
        # src is a 1D row/list at the last axis
        res.extend(src)
    return res


def reshape_list(v: list, shape: tuple[int, ...]) -> list:
    old_shape = check_shape_list(v)
    if math.prod(old_shape) != math.prod(shape):
        raise ValueError(f'shape mismatched; cannot reshape {old_shape} to {shape}')

    same_subshape_len = 0
    for d1, d2 in zip(reversed(old_shape), reversed(shape)):
        if d1 != d2: break
        same_subshape_len += 1

    v = flatten_list(v, max(len(old_shape) - same_subshape_len - 1, 0))  # ex:(2,3,4,5,6)->(3,8,5,6) flatten to (24,5,6)
    if same_subshape_len == len(old_shape):
        v = [v]
        #  v is the vector that stores the building blocks we later collect from
        #  We would first flatten v down to the longest common shape.
        #  For example: (4, 3)->(2, 2, 3), longest common shape is (3, ), and for
        #  (2, 3)->(3, 2) there is no common shape.
        #  After flattening to the right shape, we would look at each dimension s in the reversed
        #  new shape and collect as many elements as needed into s groups.
        #  For example : (3, 2)->(2, 3) we first flatten down to 1D(6, ) and from that every 3 numbers
        #  we put them into [...], and the next layer every 2 [...] we put them into [...]
        #  However, for (3, 2)->(1, 3, 2) the building blocks should be (3, 2) vecs, if we don't
        #  put the (3, 2) in a [...] beforehand we would be collecting from a bunch of (2, )s

    for s in reversed(shape[:len(shape)-same_subshape_len]):
        v = [[v[i+j] for j in range(s)] for i in range(0, len(v), s)]

    return v[0]  # remove the extra layer, we don't need to


def swapaxes_list(v: list, axis1: int, axis2: int) -> list:
    shape = check_shape_list(v)
    dep = depth(v)
    if abs(axis1) >= dep or abs(axis2) >= dep:
        raise ValueError(f'Axis out of bounds: {axis1}, {axis2} for depth {dep}')

    if axis1 == axis2:
        return v

    axis1 = dep + axis1 if axis1 < 0 else axis1
    axis2 = dep + axis2 if axis2 < 0 else axis2

    trimmed_shape = shape[:max(axis1, axis2) + 1]
    new_shape = list(trimmed_shape)
    new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]

    res: Any = []
    for s in reversed(new_shape):
        res = [copy.deepcopy(res) for _ in range(s)]

    for src_idx in itertools.product(*(range(s) for s in trimmed_shape)):
        dst_idx = list(src_idx)
        dst_idx[axis1], dst_idx[axis2] = dst_idx[axis2], dst_idx[axis1]

        dst_p = res
        src_p = v
        for i, j in zip(src_idx[:-1], dst_idx[:-1]):
            src_p = src_p[i]
            dst_p = dst_p[j]
        dst_p[dst_idx[-1]] = copy.copy(src_p[src_idx[-1]])

    return res


def transpose_list(v: list) -> list:
    shape = [len(v)]
    vv = v
    while isinstance(vv[0], list):
        shape.append(len(vv[0]))
        vv = sum(vv, [])

    stripe = len(vv)
    for s in shape:
        stripe //= s
        vv = [[vv[i + j*stripe] for j in range(s)] for i in range(stripe)]
    return vv[0]


def matmul_list(v1: list, v2: list) -> list:
    d1, d2 = depth(v1), depth(v2)
    if d1 <= 2 and d2 <= 2:
        v1m = v1 if isinstance(v1[0], list) else [v1]
        v2m = v2 if isinstance(v2[0], list) else [v2]
        return [[sum(binary_elementwise_list(row, [v2_row[m] for v2_row in v2m], operator.mul))
                 for m in range(len(v2m[0]))]
                for row in v1m]
    if d1 == d2:
        return [matmul_list(v1[min(i, len(v1)-1)], v2[min(i, len(v2)-1)])
                for i in range(max(len(v1), len(v2)))]
    if d1 > d2:
        return [matmul_list(v1_i, v2) for v1_i in v1]
    return [matmul_list(v1, v2_i) for v2_i in v2]
