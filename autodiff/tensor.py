from __future__ import annotations

import operator
from typing import Any

from .types import Number
from .array import array
from .utils import binary_elementwise_list


class tensor:
    TENSOR_MAP: list[tensor] = []
    SAVE_TO_MAP = False

    def __init__(self, value: Number | Any, op_name: str = "   ", requires_grad: bool = True, is_leaf: bool = True):
        self.arr = array(value)
        self.shape = self.arr.shape
        self.op_name = op_name
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf

        self.tag = len(tensor.TENSOR_MAP)
        self.child: list[tensor] = []
        self.parent: list[tensor] = []

        self.tangent = array(0.0)
        self.gradient = array(0.0)

        self._prop_tan = lambda: None
        self._prop_val = lambda: None
        self._prop_grad = lambda: None

        if tensor.SAVE_TO_MAP:
            tensor.TENSOR_MAP.append(self)

    @staticmethod
    def const(value: Number | Any) -> tensor:
        return tensor(value, requires_grad=False, is_leaf=True)

    @staticmethod
    def intermediate(value: Number | Any, op_name: str) -> tensor:
        return tensor(value, op_name=op_name, requires_grad=True, is_leaf=False)

    def add_parent(self, *args: tensor) -> None:
        self.parent.extend(args)

    def add_child(self, *args: tensor) -> None:
        self.child.extend(args)

    def topo(self, visited: set[tensor], order: list[tensor], roots: list[tensor]) -> None:
        """
        Traverses the computation graph with DFS by chaining up self.parent and records three things:
        1. The nodes visited - in the visited set
        2. The topological order of the nodes from root nodes all the way to leaf nodes - in the order list
        3. The root nodes that requires grad - in the roots list

        The function does not return anything but appends nodes to the input containers (visited, order and roots)
        with side effects.

        Parameters
        ----------
        visited : set[tensor]
                The set of nodes that have been visited.
        order : list[tensor]
                The ordered-list of topologically sorted nodes in the computation graph.
        roots : list[tensor]
                The list of nodes that are leaves and require grads.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if self.parent:
            for p in self.parent:
                if p not in visited:
                    p.topo(visited, order, roots)
        elif self.is_leaf and self.requires_grad:
            roots.append(self)
        order.append(self)
        visited.add(self)

    def fix_broadcast_grad(self, incoming_grad: array) -> array:
        """
        Transforms the shape of the in_coming gradients array to the shape of this tensor.

        Parameters
        ----------
        incoming_grad : array
                The gradients that flow into this tensor.

        Returns
        -------
        array
                The gradient array with the shape same as this tensor.

        Raises
        ------
        None
        """
        original_shape = self.shape
        broadcast_shape = incoming_grad.shape
        if original_shape == broadcast_shape:
            return incoming_grad

        padded_ori_shape = (0,) * (len(broadcast_shape) - len(original_shape)) + original_shape
        v_grad = incoming_grad.value

        for original_s, broadcast_s in zip(padded_ori_shape, broadcast_shape):
            if original_s != broadcast_s:
                for next_grad in v_grad[1:]:
                    v_grad[0] = binary_elementwise_list(v_grad[0], next_grad, operator.add)
                v_grad = v_grad[0]

        res = array(0.0)
        res.value = v_grad
        res.shape = original_shape
        return res

    def zero_grad(self) -> None:
        """
        Resets the gradients of the root tensors to zeros.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        visited = set()
        order: list[tensor] = []
        roots: list[tensor] = []
        self.topo(visited, order, roots)
        for t in order:
            t.gradient = array(0.0)

    # Dunder ops dispatch to ops/linalg (local import avoids circulars)
    def __add__(self, other):
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        from .ops import add
        return add(other, self)

    def __sub__(self, other):
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)

    def __rmul__(self, other):
        from .ops import mul
        return mul(other, self)

    def __truediv__(self, other):
        from .ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        from .ops import div
        return div(other, self)

    def __neg__(self):
        from .ops import neg
        return neg(self)

    def __abs__(self):
        from .ops import abs_
        return abs_(self)

    def __matmul__(self, other):
        from .linalg import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .linalg import matmul
        return matmul(other, self)

    # Convenience math
    def exp(self):
        from .ops import exp
        return exp(self)

    def log(self):
        from .ops import log
        return log(self)

    def sin(self):
        from .ops import sin
        return sin(self)

    def cos(self):
        from .ops import cos
        return cos(self)

    def flatten(self):
        from .linalg import flatten
        return flatten(self)

    def transpose(self):
        from .linalg import transpose
        return transpose(self)
