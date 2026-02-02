from __future__ import annotations

import numpy as np

from autodiff import array, tensor, flatten, reshape, transpose, swapaxes, matmul, vjp
from tests.utils import assert_close


def _np(a):
    return np.array(a, dtype=float)


def test_linalg_wrappers_return_tensor() -> None:
    a = array([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2)
    out = flatten(a)
    assert isinstance(out, tensor)
    assert out.arr.shape == (4,)
    assert out.arr.value == [1.0, 2.0, 3.0, 4.0]


def test_linalg_wrappers_on_tensor_forward_match_numpy() -> None:
    x = tensor([[1.0, 2.0], [3.0, 4.0]])

    assert_close(flatten(x), tensor(_np(x.arr.value).ravel().tolist()))
    assert_close(transpose(x), tensor(_np(x.arr.value).transpose().tolist()))
    assert_close(swapaxes(x, 0, 1), tensor(np.swapaxes(_np(x.arr.value), 0, 1).tolist()))

    r = reshape(x, (4,))
    assert_close(r, tensor(_np(x.arr.value).reshape((4,)).tolist()))


def test_matmul_wrapper_matches_numpy() -> None:
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0], [7.0, 8.0]])
    out = matmul(a, b)
    expected = (_np(a.arr.value) @ _np(b.arr.value)).tolist()
    assert_close(out, tensor(expected))


def test_matmul_tensor_vjp_shapes() -> None:
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    y = tensor([[5.0, 6.0], [7.0, 8.0]])
    z = x @ y  # (2,2)

    grads = vjp(z, inputs={x: x.arr, y: y.arr}, cotangent=None, pull_back=False)
    assert grads[x].shape == x.shape
    assert grads[y].shape == y.shape


def test_transpose_then_matmul_matches_numpy() -> None:
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0], [7.0, 8.0]])
    z = a.transpose() @ b
    expected = (_np(a.arr.value).transpose() @ _np(b.arr.value)).tolist()
    assert_close(z, tensor(expected))
