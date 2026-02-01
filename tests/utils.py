from autodiff import array, tensor
from autodiff.utils import check_shape_list, flatten_list, reshape_list


def nested_map(fn, v1, v2=None) -> list:
    """
    Extension of the map() to nested list
    """
    old_shape = check_shape_list(v1)
    if v2 is None:
        return reshape_list(list(map(fn, flatten_list(v1))), old_shape)
    return reshape_list(list(map(fn, flatten_list(v1), flatten_list(v2))), old_shape)


def assert_close(a: list | array | tensor, b: list | array | tensor, tol: float = 1e-12) -> None:
    """
    Asserts the elements in two containers are close.
    """
    if type(a) != type(b):
        raise ValueError("type mismatched; a and b must be of the same type")

    v1 = a if isinstance(a, list) else a.value if isinstance(a, array) else a.arr.value
    v2 = b if isinstance(b, list) else b.value if isinstance(b, array) else b.arr.value

    assert check_shape_list(v1) == check_shape_list(v2)
    for x, y in zip(flatten_list(v1), flatten_list(v2)):
        assert abs(x - y) <= tol


def scalar(x: float, *, requires_grad: bool = True) -> tensor:
    """
    Convenience: create a scalar tensor.
    """
    return tensor(x, requires_grad=requires_grad, is_leaf=True)
