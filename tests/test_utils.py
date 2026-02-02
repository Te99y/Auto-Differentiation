import pytest
import operator
import math

from tests.utils import nested_map
from autodiff import array
from autodiff.utils import (
    sign,
    depth,
    check_shape_list,
    broadcast,
    unary_elementwise_list,
    binary_elementwise_list,
    reshape_list,
    flatten_list,
    swapaxes_list,
    transpose_list,
    matmul_list
)


@pytest.mark.parametrize("v, res", [(-1e2, -1.0), (-3e-4, -1.0), (0.0, 0.0), (2e-2, 1.0)])
def test_sign(v, res) -> None:
    assert sign(v) == res


@pytest.mark.parametrize("v, res", [(1.2, 0), ([3, 4], 1), ([[5, 6], [7, 8]], 2), ([[[2], [3]]], 3)])
def test_depth(v, res) -> None:
    assert depth(v) == res


def test_depth_empty_list() -> None:
    assert depth([]) == 0
    assert depth([[]]) == 1
    assert depth([[[]]]) == 2


def test_check_shape() -> None:
    assert check_shape_list([]) == (0,)
    assert check_shape_list([[]]) == (1, 0)
    assert check_shape_list([1]) == (1,)
    assert check_shape_list([[1]]) == (1, 1)
    assert check_shape_list([1, 2]) == (2,)
    assert check_shape_list([[1, 2]]) == (1, 2,)


@pytest.mark.parametrize("shape", [(1, ), (2, 3), (4, 2, 1, 1, 3), (2, 3, 1, 2)])
def test_check_shape_nested(shape: tuple[int, ...]) -> None:
    assert check_shape_list(array([1, 2], outer_shape=shape).value) == shape + (2, )


@pytest.mark.parametrize(
    "shape1, shape2, res",
    [((2, 3), (2, 3), (2, 3)),
     ((2, 3, 4, 5), (4, 1), (2, 3, 4, 5)),
     ((3, 4, 5), (2, 3, 1, 5), (2, 3, 4, 5))]
)
def test_broadcast_valid(shape1, shape2, res) -> None:
    assert broadcast(shape1, shape2) == res


def test_broadcast_invalid() -> None:
    with pytest.raises(TypeError):
        broadcast((2, 3, 4), (2, 3, 5))


def test_broadcast_commutes() -> None:
    assert broadcast((2, 3, 1), (1, 3, 4)) == broadcast((1, 3, 4), (2, 3, 1))


def test_unary_elementwise() -> None:
    v = array([0.1234, 3.2345], outer_shape=(3, 1, 2)).value
    assert unary_elementwise_list(v, math.log) == nested_map(math.log, v)
    assert unary_elementwise_list(v, math.exp) == nested_map(math.exp, v)
    assert unary_elementwise_list(v, math.sin) == nested_map(math.sin, v)
    assert unary_elementwise_list(v, math.cos) == nested_map(math.cos, v)
    assert unary_elementwise_list(v, operator.abs) == nested_map(operator.abs, v)
    with pytest.raises(ValueError):
        unary_elementwise_list([-0.1, 0.2], math.log)


def test_binary_elementwise() -> None:
    v1 = array([2.1234, 4.3456, -5.4567], outer_shape=(2, 1, 4)).value
    v2 = array([-12.6789, 23.789, 45.9012], outer_shape=(2, 1, 4)).value
    v_zip = list(zip(flatten_list(v1), flatten_list(v2)))
    assert binary_elementwise_list(v1, v2, operator.add) == nested_map(operator.add, v1, v2)
    assert binary_elementwise_list(v1, v2, operator.sub) == nested_map(operator.sub, v1, v2)
    assert binary_elementwise_list(v1, v2, operator.mul) == nested_map(operator.mul, v1, v2)
    assert binary_elementwise_list(v1, v2, operator.truediv) == nested_map(operator.truediv, v1, v2)


@pytest.mark.parametrize(
    "s1, s2",
    [((1, ), (1, )),
     ((6, ), (6, )),
     ((6, ), (1, 6, )),
     ((6, ), (2, 3)),
     ((6, ), (3, 1, 2)),
     ((6, ), (1, 2, 1, 3)),
     ((4, 1, 3), (4, 1, 3)),
     ((4, 1, 3), (4, 1, 1, 3)),
     ((4, 1, 3), (1, 4, 1, 3)),
     ((4, 1, 3), (1, 1, 4, 1, 3)),
     ((4, 1, 3), (1, 1, 4, 1, 1, 3)),
     ((4, 1, 3), (4, 1, 3, 1)),
     ((4, 1, 3), (4, 1, 3, 1, 1)),
     ((4, 1, 3), (1, 4, 1, 3, 1, 1)),
     ((4, 1, 3), (1, 1, 4, 1, 1, 3, 1)),
     ((4, 1, 3), (1, 1, 4, 1, 1, 3, 1, 1)),
     ((4, 1, 3), (2, 3, 2, 1)),
     ((4, 1, 3), (2, 2, 1, 3)),
     ((4, 1, 3), (2, 1, 3, 2)),
     ((4, 1, 3), (1, 2, 3, 2)),
     ((4, 1, 3), (1, 2, 3, 2, 1)),
     ((2, 2, 1, 3), (4, 1, 3)),
     ((2, 2, 1, 3), (1, 3, 4)),
     ((2, 2, 1, 3), (3, 4, 1)),
     ((2, 2, 1, 3), (2, 1, 3, 2)),
     ((2, 2, 1, 3), (12, )),
     ((2, 2, 1, 3), (12, 1)),
     ((2, 2, 1, 3), (1, 12)),
     ((2, 2, 1, 3), (1, 12, 1)),
     ((2, 2, 1, 3), (1, 1, 12, 1)),
     ((2, 2, 1, 3), (1, 1, 12, 1, 1)),
     ]
)
def test_reshape(s1, s2) -> None:
    v1 = array(0.0, outer_shape=s1).value
    v2 = array(0.0, outer_shape=s2).value
    assert reshape_list(v1, s2) == v2


def test_reshape_invalid() -> None:
    with pytest.raises(ValueError) as e:
        reshape_list([[1, 2, 3], [1, 2, 3]], (4, ))
    assert str(e.value) == 'shape mismatched; cannot reshape (2, 3) to (4,)'


def test_flatten() -> None:
    assert flatten_list([]) == []
    assert flatten_list([1]) == [1]
    assert flatten_list([1, 2, 3]) == [1, 2, 3]
    assert flatten_list([[1, 2], [3, 4]], layers=0) == [[1, 2], [3, 4]]
    assert flatten_list([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], layers=0) \
           == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    assert flatten_list([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], layers=1) \
           == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert flatten_list([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], layers=2) \
           == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert flatten_list([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], layers=-1) \
           == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert flatten_list([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], layers=-2) \
           == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]


def test_flatten_invalid() -> None:
    with pytest.raises(ValueError):
        flatten_list([1, 2, 3], layers=1)
    with pytest.raises(ValueError):
        flatten_list([[1, 2, 3], [1, 2, 3]], layers=2)
    with pytest.raises(ValueError):
        flatten_list([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]], layers=3)


@pytest.mark.parametrize(
    "shape",
    [(1, ),
     (6, ),
     (1, 2),
     (2, 1),
     (3, 1, 2),
     (1, 1, 2, 3),
     (1, 1, 2, 3, 1),
     (1, 1, 2, 3, 1, 1)]
)
def test_flatten_reshape_roundtrip(shape: tuple[int, ...]):
    v = array(0, outer_shape=shape).value
    assert reshape_list(flatten_list(v), shape) == v


def test_swapaxes():
    assert swapaxes_list([[1]], 0, 0) == [[1]]
    assert swapaxes_list([[1]], 0, 1) == [[1]]
    assert swapaxes_list([[1, 2], [3, 4]], 0, 1) == [[1, 3], [2, 4]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0, 1) == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, 0) == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0, 2) == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 0) == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, 2) == [[[1, 3], [2, 4]], [[5, 7], [6, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 1) == [[[1, 3], [2, 4]], [[5, 7], [6, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0, -1) == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0, -2) == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert swapaxes_list([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], -1, -2) == [[[1, 3], [2, 4]], [[5, 7], [6, 8]]]


def test_swapaxes_invalid() -> None:
    with pytest.raises(ValueError):
        swapaxes_list([], 0, 0)
    with pytest.raises(ValueError):
        swapaxes_list([], 0, 1)
    with pytest.raises(ValueError):
        swapaxes_list([1, 2, 3], 0, 1)
    with pytest.raises(ValueError):
        swapaxes_list([[1, 2, 3], [4, 5, 6]], 1, 2)


def test_transpose() -> None:
    assert transpose_list([1]) == [1]
    assert transpose_list([[1]]) == [[1]]
    assert transpose_list([1, 2]) == [1, 2]
    assert transpose_list([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert transpose_list([[[1, 2], [3, 4]]]) == [[[1], [3]], [[2], [4]]]
    assert transpose_list([[[[1, 2], [3, 4]]]]) == [[[[1]], [[3]]], [[[2]], [[4]]]]


def test_transpose_invalid() -> None:
    with pytest.raises(IndexError):
        transpose_list([])


def test_matmul():
    assert matmul_list([[1, 2]], [[3], [4]]) == [[11]]  # (1, 2)@(2, 1)
    assert matmul_list([[3], [4]], [[1, 2]]) == [[3, 6], [4, 8]]  # (2, 1)@(1, 2)
    v1 = array([[[1.123, -2.234, 3.345], [-4.456, 5.567, -6.678]]], outer_shape=(2, 1)).value  # (2, 1, 1, 2, 3)
    v2 = swapaxes_list(v1, -1, -2)
    res = matmul_list(v1, v2)  # (2, 1, 1, 2, 3)@(2, 1, 1, 3, 2)
    expected = [[[[[17.44091, -39.778676], [-39.778676, 95.44310899999999]]]],
                [[[[17.44091, -39.778676], [-39.778676, 95.44310899999999]]]]]
    assert flatten_list(res) == pytest.approx(flatten_list(expected), rel=1e-12, abs=1e-12)
    assert check_shape_list(res) == (2, 1, 1, 2, 2)
