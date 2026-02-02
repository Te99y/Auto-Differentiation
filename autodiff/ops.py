from __future__ import annotations

from .tensor import tensor
from .types import TensorLike
from .utils import sign


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


def add(a: TensorLike, b: TensorLike) -> tensor:
    """
    Returns a tensor a+b after adding it to the computation graph. The propagation function
    for value, tangents and gradients are added to parents a and b and the output tensor.

    Parameters
    ----------
    a : int | float | list | array | tensor
            The first value to be added.
    b : int | float | list | array | tensor
            The second value to add.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    a_t, b_t = _to_tensor(a), _to_tensor(b)
    out = tensor.intermediate(a_t.arr + b_t.arr, op_name="add")
    out.add_parent(a_t, b_t)
    a_t.add_child(out); b_t.add_child(out)

    def _prop_val():
        out.arr = a_t.arr + b_t.arr
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = a_t.tangent + b_t.tangent
    out._prop_tan = _prop_tan

    def _prop_grad():
        a_t.gradient += a_t.fix_broadcast_grad(out.gradient * 1.0)  # convert potential int to float
        b_t.gradient += b_t.fix_broadcast_grad(out.gradient * 1.0)
    out._prop_grad = _prop_grad
    return out


def sub(a: TensorLike, b: TensorLike) -> tensor:
    """
    Returns a tensor a-b after adding it to the computation graph. The propagation function
    for value, tangents and gradients are added to parents a and b and the output tensor.

    Parameters
    ----------
    a : int | float | list | array | tensor
            The first value to be subtracted.
    b : int | float | list | array | tensor
            The second value to subtract.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    a_t, b_t = _to_tensor(a), _to_tensor(b)
    out = tensor.intermediate(a_t.arr - b_t.arr, op_name="sub")
    out.add_parent(a_t, b_t)
    a_t.add_child(out); b_t.add_child(out)

    def _prop_val():
        out.arr = a_t.arr - b_t.arr
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = a_t.tangent - b_t.tangent
    out._prop_tan = _prop_tan

    def _prop_grad():
        a_t.gradient += a_t.fix_broadcast_grad(out.gradient * 1.0)
        b_t.gradient += b_t.fix_broadcast_grad(out.gradient * -1.0)
    out._prop_grad = _prop_grad

    return out


def mul(a: TensorLike, b: TensorLike) -> tensor:
    """
    Returns a tensor a*b after adding it to the computation graph. The propagation function
    for value, tangents and gradients are added to parents a and b and the output tensor.

    Please note that this is elementwise multiplication instead of matmul.

    Parameters
    ----------
    a : int | float | list | array | tensor
            The first value to perform elementwise multiplication.
    b : int | float | list | array | tensor
            The second value to perform elementwise multiplication.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    a_t, b_t = _to_tensor(a), _to_tensor(b)
    out = tensor.intermediate(a_t.arr * b_t.arr, op_name="mul")
    out.add_parent(a_t, b_t)
    a_t.add_child(out); b_t.add_child(out)

    def _prop_val():
        out.arr = a_t.arr * b_t.arr
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = b_t.arr * a_t.tangent + a_t.arr * b_t.tangent
    out._prop_tan = _prop_tan

    def _prop_grad():
        a_t.gradient += a_t.fix_broadcast_grad(out.gradient * b_t.arr)
        b_t.gradient += b_t.fix_broadcast_grad(out.gradient * a_t.arr)
    out._prop_grad = _prop_grad

    return out


def div(a: TensorLike, b: TensorLike) -> tensor:
    """
    Returns a tensor a/b after adding it to the computation graph. The propagation function
    for value, tangents and gradients are added to parents a and b and the output tensor.

    Parameters
    ----------
    a : int | float | list | array | tensor
            The first value to perform elementwise division.
    b : int | float | list | array | tensor
            The second value to perform elementwise division.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    a_t, b_t = _to_tensor(a), _to_tensor(b)
    out = tensor.intermediate(a_t.arr / b_t.arr, op_name="div")
    out.add_parent(a_t, b_t)
    a_t.add_child(out); b_t.add_child(out)

    def _prop_val():
        out.arr = a_t.arr / b_t.arr
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        one_over_b = 1.0 / b_t.arr
        out.tangent = one_over_b * (a_t.tangent - a_t.arr * one_over_b * b_t.tangent)
    out._prop_tan = _prop_tan

    def _prop_grad():
        g_over_b = out.gradient / b_t.arr
        a_t.gradient += a_t.fix_broadcast_grad(g_over_b)
        b_t.gradient += b_t.fix_broadcast_grad(-g_over_b * a_t.arr / b_t.arr)
    out._prop_grad = _prop_grad

    return out


def neg(x: TensorLike) -> tensor:
    """
    Negate a tensor and add it to the computation graph. The propagation function for
    value, tangents and gradients are also added to the parent x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to be negated.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(-x_t.arr, op_name="neg")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = -x_t.arr
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = -x_t.tangent
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += -out.gradient
    out._prop_grad = _prop_grad

    return out


def abs_(x: TensorLike) -> tensor:
    """
    Taking the elementwise absolute value of a tensor and add it to the computation graph.
    The propagation function for value, tangents and gradients are also added to the parent
    x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to take absolute.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(abs(x_t.arr), op_name="abs")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = abs(x_t.arr)
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = x_t.tangent * x_t.arr.elementwise(sign)
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += out.gradient * x_t.arr.elementwise(sign)
    out._prop_grad = _prop_grad

    return out


def exp(x: TensorLike) -> tensor:
    """
    Taking the elementwise exponential value of a tensor and add it to the computation graph.
    The propagation function for value, tangents and gradients are also added to the parent
    x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to take exponential.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.exp(), op_name="exp")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.exp()
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = x_t.tangent * out.arr
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += out.gradient * out.arr
    out._prop_grad = _prop_grad

    return out


def log(x: TensorLike) -> tensor:
    """
    Taking the elementwise log value of a tensor and add it to the computation graph.
    The propagation function for value, tangents and gradients are also added to the parent
    x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to take log.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.log(), op_name="log")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.log()
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = x_t.tangent / x_t.arr
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += out.gradient / x_t.arr
    out._prop_grad = _prop_grad

    return out


def sin(x: TensorLike) -> tensor:
    """
    Taking the elementwise-sine value of a tensor and add it to the computation graph.
    The propagation function for value, tangents and gradients are also added to the parent
    x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to take sine.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.sin(), op_name="sin")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.sin()
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = x_t.arr.cos() * x_t.tangent
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += out.gradient * x_t.arr.cos()
    out._prop_grad = _prop_grad

    return out


def cos(x: TensorLike) -> tensor:
    """
    Taking the elementwise-cosine value of a tensor and add it to the computation graph.
    The propagation function for value, tangents and gradients are also added to the parent
    x and the output tensor.

    Parameters
    ----------
    x : int | float | list | array | tensor
            The value to take cosine.

    Returns
    -------
    tensor
            The resulting tensor.

    Raises
    ------
    None
    """
    x_t = _to_tensor(x)
    out = tensor.intermediate(x_t.arr.cos(), op_name="cos")
    out.add_parent(x_t)
    x_t.add_child(out)

    def _prop_val():
        out.arr = x_t.arr.cos()
        out.shape = out.arr.shape
    out._prop_val = _prop_val

    def _prop_tan():
        out.tangent = -x_t.arr.sin() * x_t.tangent
    out._prop_tan = _prop_tan

    def _prop_grad():
        x_t.gradient += -(out.gradient * x_t.arr.sin())
    out._prop_grad = _prop_grad

    return out
