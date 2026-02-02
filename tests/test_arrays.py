from __future__ import annotations

import numpy as np
import pytest

from autodiff import array


def _to_numpy(a: array) -> np.ndarray:
    return np.array(a.value, dtype=float)


def _assert_array_value_close(a: array, b: np.ndarray, *, atol: float = 1e-12, rtol: float = 1e-12) -> None:
    an = _to_numpy(a)
    assert an.shape == b.shape
    assert np.allclose(an, b, atol=atol, rtol=rtol)


def test_array_init_scalar() -> None:
    a = array(3.0)
    assert a.shape == (1,)
    assert a.value == [3.0]


def test_array_init_list_shape() -> None:
    a = array([[1.0, 2.0], [3.0, 4.0]])
    assert a.shape == (2, 2)
    assert a.value == [[1.0, 2.0], [3.0, 4.0]]


def test_array_init_outer_shape_repeats() -> None:
    a = array([1.0, 2.0], outer_shape=(2, 1))
    assert a.shape == (2, 1, 2)
    assert a.value == [[[1.0, 2.0]], [[1.0, 2.0]]]


@pytest.mark.parametrize("lhs, rhs", [
    (array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0])),
    (array([[1.0, 2.0], [3.0, 4.0]]), array([[10.0, 20.0], [30.0, 40.0]])),
])
def test_array_binary_ops_match_numpy(lhs: array, rhs: array) -> None:
    ln = _to_numpy(lhs)
    rn = _to_numpy(rhs)

    _assert_array_value_close(lhs + rhs, ln + rn)
    _assert_array_value_close(lhs - rhs, ln - rn)
    _assert_array_value_close(lhs * rhs, ln * rn)
    _assert_array_value_close(lhs / rhs, ln / rn)


def test_array_broadcast_add_matches_numpy() -> None:
    a = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # (2,3)
    b = array([10.0, 20.0, 30.0])                    # (3,)
    _assert_array_value_close(a + b, _to_numpy(a) + _to_numpy(b))


def test_array_unary_ops_match_numpy() -> None:
    a = array([[0.1, 0.2], [0.3, 0.4]])
    an = _to_numpy(a)
    _assert_array_value_close(-a, -an)
    _assert_array_value_close(abs(a), np.abs(an))
    _assert_array_value_close(a.exp(), np.exp(an))
    _assert_array_value_close(a.sin(), np.sin(an))
    _assert_array_value_close(a.cos(), np.cos(an))


def test_array_log_domain_error() -> None:
    a = array([-1.0, 0.1])
    with pytest.raises(ValueError):
        _ = a.log()


def test_array_flatten_matches_numpy_ravel() -> None:
    a = array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2,2,2)
    f = a.flatten()
    assert f.shape == (8,)
    assert f.value == list(_to_numpy(a).ravel())


def test_array_reshape_matches_numpy() -> None:
    a = array([[1.0, 2.0], [3.0, 4.0]])
    r = a.reshape((4,))
    assert r.shape == (4,)
    assert r.value == [1.0, 2.0, 3.0, 4.0]

    r2 = a.reshape((1, 2, 2))
    assert r2.shape == (1, 2, 2)
    _assert_array_value_close(r2, _to_numpy(a).reshape((1, 2, 2)))


def test_array_transpose_matches_numpy() -> None:
    a = array([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2)
    t = a.transpose()
    _assert_array_value_close(t, _to_numpy(a).transpose())


def test_array_swapaxes_matches_numpy() -> None:
    a = array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2,2,2)
    s = a.swapaxes(0, 2)
    _assert_array_value_close(s, np.swapaxes(_to_numpy(a), 0, 2))


def test_array_matmul_matches_numpy() -> None:
    a = array([[1.0, 2.0], [3.0, 4.0]])
    b = array([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    _assert_array_value_close(c, _to_numpy(a) @ _to_numpy(b))


def test_array_matmul_batch_matches_numpy() -> None:
    # shape (2, 1, 2, 3) @ (2, 1, 3, 4) -> (2, 1, 2, 4)
    a = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], outer_shape=(2, 1))
    b = array([[7.0, 8.0, 9.0, 10.0],
               [11.0, 12.0, 13.0, 14.0],
               [15.0, 16.0, 17.0, 18.0]], outer_shape=(2, 1))
    c = a @ b
    _assert_array_value_close(c, _to_numpy(a) @ _to_numpy(b))


def test_array_matmul_shape_error() -> None:
    a = array([[1.0, 2.0]])  # (1,2)
    b = array([[1.0, 2.0]])  # (1,2)
    with pytest.raises(ValueError):
        _ = a @ b
