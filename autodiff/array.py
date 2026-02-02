from __future__ import annotations

import math
import operator
from copy import deepcopy
from typing import Any, Callable

from .types import ArrayLike, TensorLike
from .utils import (
    check_shape_list, broadcast,
    unary_elementwise_list, binary_elementwise_list,
    flatten_list, swapaxes_list, reshape_list, transpose_list,
    matmul_list
)

BinaryOP = Callable[[float, float], float]
UnaryOP = Callable[[float], float]


class array:
    """A minimal n-d array backed by nested Python lists.

    This is intentionally NumPy-free; it supports elementwise ops, basic unary/binary
    functions, shape inference, broadcasting, and a few linalg helpers.
    """
    def __init__(self, init_value: TensorLike, *, outer_shape: tuple[int, ...] = ()) -> None:
        """
        Creates an array with init_value, and repeat the structure with outer_shape if given.

        Parameters
        ----------
        init_value : int | float | list | array | tensor
                The value that will be used to create a new array.
        outer_shape : tuple[int, ...]
                The shape to embed the init_value in. The init value will be repeated many times
                to create the structure. For example, init_value is [2, 3], outer_shape is (3, 1),
                the resulting array will be of shape (3, 1, 2) and value [[[2, 3]], [[2, 3]], [[2, 3]]]

        Returns
        -------
        None

        Raises
        ------
        ValueError
                If the provided init_value is a non-homogeneous (ragged) list.
                Or the init_value type is not one of int | float | list | array | tensor
        TypeError
                If the provided inti_value is a list that contains non-numerical entries.
        """
        outer_shape = tuple(outer_shape)

        if isinstance(init_value, array):
            self.shape = outer_shape + init_value.shape
            self.value = deepcopy(init_value.value)
            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]

        elif isinstance(init_value, list):
            queue = init_value
            inner_shape = (len(queue),)
            while isinstance(queue[0], list):
                if not all(isinstance(q, list) and len(q) == len(queue[0]) for q in queue):
                    raise ValueError('Legal lists must be homogeneous')
                inner_shape += (len(queue[0]),)
                queue = sum(queue, [])
            if not all(isinstance(v, (int, float)) for v in queue):
                raise TypeError('Only numbers or lists are allowed in a legal list')

            # Deep-copy to avoid aliasing external lists
            self.value = deepcopy(init_value)

            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]
            self.shape = outer_shape + inner_shape

        elif isinstance(init_value, (int, float)):
            if outer_shape:
                self.shape = outer_shape
                v: Any = float(init_value)
                for s in reversed(outer_shape):
                    v = [deepcopy(v) for _ in range(s)]
                self.value = v
            else:
                self.shape = (1,)
                self.value = [float(init_value)]
        # unwrap tensor-like without importing tensor
        elif hasattr(init_value, "arr"):  # tensor-like
            init_value = init_value.arr
        else:
            raise ValueError(f'Unsupported init type: {type(init_value)}')

    def update_shape(self) -> tuple[int, ...]:
        """
        Update self.shape by the shape of self.value.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[int, ...]
                Returns self.shape after updating it.

        Raises
        ------
        None

        """
        self.shape = check_shape_list(self.value)
        return self.shape

    def broadcast_with(self, other: ArrayLike) -> tuple[int, ...]:
        """
        Returns the final shape after broadcasting with another object.

        Parameters
        ----------
        other : Number | list | array
                The target to broadcast with.

        Returns
        -------
        tuple[int, ...]
                The final shape after broadcasting.


        Raises
        ------
        TypeError
                If the two shapes cannot broadcast.

        """
        shape1 = self.shape
        shape2 = other.shape if isinstance(other, array) else array(other).shape
        return broadcast(shape1, shape2)

    def elementwise(self, op: BinaryOP | UnaryOP, other: array = None, inplace=False) -> array:
        """
        Perform elementwise operation on self or between self and another array.

        If other is None then unary operation is performed on self, such as sign and log, otherwise,
        binary operation is performed between self and another array, such as math.add.

        Parameters
        ----------
        op : BinaryOP | UnaryOP
                The operation to perform, such as math.add.
        other : array = None
                The array to perform elementwise op with.
        inplace: bool = False
                Whether to save the result in self or not. If True, self would be overwritten with the new value and
                shape, otherwise, a new array would be created.

        Returns
        -------
        array
                The resulting array. Returns self if inplace is True, otherwise a new array.

        Raises
        ------
        ValueError
                If op type mismatches with `other`.

        """
        if other is None:
            try:
                _ = op(1.0)
            except TypeError as e:
                raise ValueError('op is not unary; op must be unary if other is None') from e
            return self._elementwise_unary(op, inplace=inplace)

        try:
            _ = op(1.0, 1.0)
        except TypeError as e:
            raise ValueError('op is not binary; op must be binary if other is None') from e
        result = self if inplace else array(0.0)
        result.value = binary_elementwise_list(self.value, other.value, op)
        result.shape = self.broadcast_with(other)
        return result

    def _elementwise_unary(self, op: UnaryOP, inplace=False) -> array:
        result = self if inplace else array(0.0)
        result.value = unary_elementwise_list(self.value, op)
        result.shape = self.shape
        return result

    def flatten(self, *, layers: int = -1) -> array:
        """
        Transforms an array to 1D.

        Parameters
        ----------
        layers : int (optional)
                How many layers to flatten.
                For example:
                |    shape     |  layers | flatten shape |
                |--------------|---------|---------------|
                | (1, 2, 3, 4) |    0    | (1, 2, 3, 4)  |
                | (1, 2, 3, 4) |    1    | (2, 3, 4)     |
                | (1, 2, 3, 4) |    2    | (6, 4)        |
                | (1, 2, 3, 4) |    3    | (24,)         |
                | (1, 2, 3, 4) |   -1    | (24,)         |
                | (1, 2, 3, 4) |   -2    | (6, 4)        |
                | (1, 2, 3, 4) |   -3    | (2, 3, 4)     |


        Returns
        -------
        array
                A new flattened array.

        Raises
        ------
        ValueError
                If layers >= the length of the shape. For example: shape is (1, 2, 3), layers is 3, 4 or above.

        """
        return array(flatten_list(self.value))

    def reshape(self, shape: tuple[int, ...]) -> array:
        """
        Reverse the shape of an array.

        Parameters
        ---------
        shape : tuple[int, ...]
                The shape to transform to.

        Returns
        -------
        A new reshaped array.

        Raises
        ------
        ValueError
                If the number of scalars in the old shape isn't equivalent to that in the new shape.
                For example, in (2, 3)->(5, ) since there were 6 scalars in (2, 3), the numbers don't match.

        """
        return array(reshape_list(self.value, shape))

    def swapaxes(self, axis1: int, axis2: int) -> array:
        """
        Swap two axes of an array.

        Parameters
        ----------
        axis1 : int
                The first axis
        axis2 : int
                The second axis to swap with the first

        Returns
        -------
        array
                A new array with the two axes swapped.

        Raises
        ------
        ValueError
                If either of the axes exceed the length of the shape. For example, shape is (1, 2, 3),
                either axis1 or axis2 is 3, 4 or above.

        """
        return array(swapaxes_list(self.value, axis1, axis2))

    def transpose(self) -> array:
        """
        Reverse the shape of an array.

        Parameters
        ---------
        None

        Returns
        -------
        A new array with the shape reversed.

        Raises
        ------
        None

        """
        return array(transpose_list(self.value))

    def exp(self) -> array: return self._elementwise_unary(math.exp)
    def log(self) -> array: return self._elementwise_unary(math.log)
    def sin(self) -> array: return self._elementwise_unary(math.sin)
    def cos(self) -> array: return self._elementwise_unary(math.cos)

    def __add__(self, other): return self.elementwise(operator.add, other if isinstance(other, array) else array(other))
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self.elementwise(operator.sub, other if isinstance(other, array) else array(other))
    def __rsub__(self, other): return (other if isinstance(other, array) else array(other)) - self
    def __mul__(self, other): return self.elementwise(operator.mul, other if isinstance(other, array) else array(other))
    def __rmul__(self, other): return (other if isinstance(other, array) else array(other)) * self
    def __truediv__(self, other): return self.elementwise(operator.truediv, other if isinstance(other, array) else array(other))
    def __rtruediv__(self, other): return (other if isinstance(other, array) else array(other)) / self
    def __neg__(self): return self.elementwise(operator.neg)
    def __abs__(self): return self.elementwise(operator.abs)

    def __matmul__(self, other: Any):
        other = other if isinstance(other, array) else array(other)
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f'The last dim of {self.shape} does not equal the second to last dim '
                             f'of the {other.shape}. Refer to the signature (...,n,k),(...,k,m)->(...,n,m).')

        for dim1, dim2 in zip(reversed(self.shape[:-2]), reversed(other.shape[:-2])):
            if dim1 != dim2 and dim1 != 0 and dim2 != 0:
                raise ValueError(f'Cannot broadcast between {self.shape} and {other.shape}.')

        return array(matmul_list(self.value, other.value))

    def __rmatmul__(self, other: Any):
        return (other if isinstance(other, array) else array(other)) @ self

    def __str__(self):
        return f'shape:{self.shape}, value:{self.value}'
