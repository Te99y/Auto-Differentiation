from __future__ import annotations

from .types import TensorLike
from .tensor import tensor


def _to_tensor(x: TensorLike) -> tensor:
    """
    Converts a non-tensor to a leaf tensor that doesn't require grad.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to convert to a tensor, if not already.

    Returns
    -------
    tensor
            The converted tensor if x was not a tensor.

    Raises
    ------
    None
    """
    return x if isinstance(x, tensor) else tensor.const(x)


def flatten(x: TensorLike) -> tensor:
    """
    Flatten a tensor and add it to the computation graph. The propagation function for value,
    tangents and gradients are also added to the parent x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to flatten.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.flatten(), op_name="flat")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.flatten()
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    # Forward-mode: flatten tangent
    def _prop_tan():
        out.tangent = x_t.tangent.flatten()
    out._prop_tan = _prop_tan

    # Reverse-mode: reshape grad back
    def _prop_grad():
        x_t.gradient += out.gradient.reshape(x_t.shape)
    out._prop_grad = _prop_grad

    return out


def reshape(x: TensorLike, shape: tuple[int, ...]) -> tensor:
    """
    Reshape a tensor and add it to the computation graph. The propagation function for value,
    tangents and gradients are also added to the parent x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to reshape.
    shape : tuple[int, ...]
            The shape to transform to.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.flatten(), op_name="flat")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.reshape(shape)
        out.shape = shape
    out._prop_val = _prop_val

    # Forward-mode: reshape tangent
    def _prop_tan():
        out.tangent = x_t.tangent.reshape(shape)
    out._prop_tan = _prop_tan

    # Reverse-mode: reshape grad back
    def _prop_grad():
        x_t.gradient += out.gradient.reshape(x_t.shape)
    out._prop_grad = _prop_grad

    return out


def transpose(x: TensorLike) -> tensor:
    """
    Transpose a tensor and add it to the computation graph. The propagation function for value,
    tangents and gradients are also added to the parent x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to transpose.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.transpose(), op_name="tra")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.transpose()
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = x_t.tangent.transpose()
    out._prop_tan = _prop_tan

    def _prop_grad():
        # d/dX transpose: gradient transposed back
        x_t.gradient += out.gradient.transpose()
    out._prop_grad = _prop_grad

    return out


def swapaxes(x: TensorLike, axis1: int, axis2: int) -> tensor:
    """
    Swaps two axes of a tensor and add it to the computation graph. The propagation function
    for value, tangents and gradients are also added to the parent x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to flatten.
    axis1 : int
            The first axis
    axis2 : int
            The second axis to swap with the first

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.swapaxes(axis1, axis2), op_name="swp")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.swapaxes(axis1, axis2)
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = x_t.tangent.swapaxes(axis1, axis2)
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += out.gradient.swapaxes(axis1, axis2)
    out._prop_grad = _prop_grad

    return out


def matmul(a: TensorLike, b: TensorLike) -> tensor:
    """
    Returns a tensor a@b after adding it to the computation graph. The propagation function
    for value, tangents and gradients are added to parents a and b and the output tensor.

    Parameters
    ----------
    a : int | float | list | array | tensor
            The first value to perform matmul.
    a : int | float | list | array | tensor
            The second value to perform matmul.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    a_t, b_t = _to_tensor(a), _to_tensor(b)
    out = tensor.intermediate(a_t.arr @ b_t.arr, op_name="mat")
    out.add_parent(a_t, b_t)
    a_t.add_child(out); b_t.add_child(out)

    def _prop_val():
        out.arr = a_t.arr @ b_t.arr
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    # Forward-mode JVP:
    # d(A@B)=dA@B + A@dB
    def _prop_tan():
        out.tangent = (a_t.tangent @ b_t.arr) + (a_t.arr @ b_t.tangent)
    out._prop_tan = _prop_tan

    # Reverse-mode VJP:
    # dA += G @ B^T
    # dB += A^T @ G
    def _prop_grad():
        a_t.gradient += a_t.fix_broadcast_grad(out.gradient @ b_t.arr.swapaxes(-1, -2))
        b_t.gradient += b_t.fix_broadcast_grad(a_t.arr.swapaxes(-1, -2) @ out.gradient)
    out._prop_grad = _prop_grad

    return out
