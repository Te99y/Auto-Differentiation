from __future__ import annotations
import math
from typing import Union

TENSOR_MAP = []
Number = Union[int | float]
ListAlike = Union['array', 'tensor', list]


class array:
    def __init__(self, init_value: Number | ListAlike, outer_shape: list | tuple = ()):
        if isinstance(init_value, tensor):
            self.shape = init_value.arr.shape
            self.value = init_value.arr.value

        elif isinstance(init_value, array):
            self.shape = tuple(outer_shape) + init_value.shape
            self.value = init_value.value
            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [self.value]*s

        elif isinstance(init_value, list):
            queue = init_value.copy()
            while isinstance(queue[0], list):
                if not all(isinstance(q, list) and len(q) == len(queue[0]) for q in queue):
                    raise ValueError('Legal lists must be homogeneous')
                queue = sum(queue, [])  # cancel out 1 level of [] in it, elements become their child

            if not all(isinstance(v, Number) for v in queue):
                raise TypeError('Only numbers or lists are allowed in a legal list')

            self.value = init_value
            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [self.value]*s
            self.shape = self.check_shape()

        elif isinstance(init_value, Number):
            if outer_shape:
                self.shape = tuple(outer_shape)
                self.value = init_value
                for s in reversed(outer_shape):
                    self.value = [self.value]*s
            else:
                self.shape = (1, )
                self.value = [init_value]

    def flatten(self):
        _flatten_list = []

        def _flatten(lst):
            if isinstance(lst[0], list):
                [_flatten(l) for l in lst]
            else:
                _flatten_list.extend(lst)
        _flatten(self.value)
        return _flatten_list

    def check_shape(self):
        arr = self.value
        shape = ()
        while isinstance(arr, list):
            shape += (len(arr), )
            arr = arr[0]
        self.shape = shape
        return shape

    def broadcast_with(self, other: Number | ListAlike):
        """
        Try to broadcast between 2 arrays. If each dim are equivalent or =1 then they can broadcast
        :param other: The other number or list-like numbers
        :return: If broadcastable : The shape of the simplest broadcast shape
                 Otherwise : None
        """
        shape1 = self.shape
        shape2 = array(other).shape
        if len(shape2) > len(shape1):
            temp = shape1
            shape1 = shape2
            shape2 = temp
        residual = len(shape1)-len(shape2)
        result = () + shape1[:residual]

        for d1, d2 in zip(shape1[residual:], shape2):
            if d1 == d2 or d2 == 1:
                result += (d1, )
            elif d1 == 1:
                result += (d2, )
            else:
                raise TypeError(f'Cannot broadcast between {shape1} and {shape2}, miss matched at {d1} and {d2}')
        return result

    def __add__(self, other: Number | ListAlike):
        other_arr = array(other)
        broadcast_shape = self.broadcast_with(other_arr)


class tensor:
    def __init__(self, value: Number | list | array):
        self.arr = array(value)
        self.op_name = '   '  # primitive
        self.gradient_wrt_parent = 0
        self.parent = []
        self.child = []
        TENSOR_MAP.append(self)

    def gradient(self, root):
        """
        Calculate the gradient of self wrt root_tensor.
        Will not leave gradient on the way.
        :param root: The tensor we differentiate with respect to.
        :return: The gradient as float.
        """
        this_tensor = self
        while t.child:
            pass

    def add_parent(self, *args):
        if not all(isinstance(arg, tensor) for arg in args):
            raise TypeError('parent must be tensor')
        self.parent += [arg for arg in args]

    def add_child(self, *args):
        if not all(isinstance(arg, tensor) for arg in args):
            raise TypeError('child must be tensor')
        self.child += [arg for arg in args]

    def __str__(self):
        parent_description = [(parent.op_name, parent.arr) for parent in self.parent]
        child_description = [(child.op_name, child.arr) for child in self.child]
        return f'{self.op_name} | value:{self.arr} | parent:{parent_description} | child:{child_description}'

    def __add__(self, other):
        other_tensor = other if isinstance(other, tensor) else tensor(other)
        add_tensor = tensor(self.arr + other_tensor.arr)
        add_tensor.op_name = 'add'
        add_tensor.add_parent(self, other_tensor)
        add_tensor.gradient_wrt_parent = [1.0, 1.0]
        self.add_child(add_tensor)
        other_tensor.add_child(add_tensor)
        return add_tensor

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other_tensor = other if isinstance(other, tensor) else tensor(other)
        sub_tensor = tensor(self.arr - other_tensor.arr)
        sub_tensor.op_name = 'sub'
        sub_tensor.gradient_wrt_parent = [1.0, -1.0]
        sub_tensor.add_parent(self, other_tensor)
        self.add_child(sub_tensor)
        other_tensor.add_child(sub_tensor)
        return sub_tensor

    def __rsub__(self, other):
        other_tensor = other if isinstance(other, tensor) else tensor(other)
        return other_tensor - self

    def __neg__(self):
        neg_tensor = tensor(-self.arr)
        neg_tensor.op_name = 'neg'
        neg_tensor.gradient_wrt_parent = [-1.0]
        neg_tensor.add_parent(self)
        self.add_child(neg_tensor)
        return neg_tensor

    def __mul__(self, other):
        other_tensor = other if isinstance(other, tensor) else tensor(other)
        mul_tensor = tensor(self.arr * other_tensor.arr)
        mul_tensor.op_name = 'mul'
        mul_tensor.gradient_wrt_parent = [other_tensor.arr, self.arr]
        mul_tensor.add_parent(self, other_tensor)
        self.add_child(mul_tensor)
        other_tensor.add_child(mul_tensor)
        return mul_tensor

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other_tensor = other if isinstance(other, tensor) else tensor(other)
        div_tensor = tensor(self.arr / other_tensor.arr)
        div_tensor.op_name = 'div'
        div_tensor.gradient_wrt_parent = [1.0 / other_tensor.arr, -other_tensor.arr / (self.arr ** 2)]
        div_tensor.add_parent(self, other_tensor)
        self.add_child(div_tensor)
        other_tensor.add_child(div_tensor)
        return div_tensor

    def __rtruediv__(self, other):
        other_tensor = other if isinstance(other, tensor) else tensor(other)
        return other_tensor / self


def log(v: tensor | Number):
    v_tensor = v if isinstance(v, tensor) else tensor(v)
    log_tensor = tensor(math.log(v_tensor.arr))
    log_tensor.op_name = 'log'
    log_tensor.gradient_wrt_parent = [1.0 / v_tensor.arr]
    log_tensor.add_parent(v_tensor)
    v_tensor.add_child(log_tensor)
    return log_tensor


def exp(v: tensor | Number):
    v_tensor = v if isinstance(v, tensor) else tensor(v)
    exp_tensor = tensor(math.exp(v_tensor.arr))
    exp_tensor.op_name = 'exp'
    exp_tensor.gradient_wrt_parent = [exp_tensor.arr]
    exp_tensor.add_parent(v_tensor)
    v_tensor.add_child(exp_tensor)
    return exp_tensor


# w = tensor(-1.0)
# b = tensor(2.0)
# y = 1/(1 + w*(-(5*w + b)))
# y = w*4


for t in TENSOR_MAP:
    print(t.__str__())

