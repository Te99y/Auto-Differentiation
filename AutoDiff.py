from __future__ import annotations
import math
import operator
from typing import Union
from copy import deepcopy

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
            self.value = deepcopy(init_value.value)
            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]

        elif isinstance(init_value, list):
            queue = init_value
            while isinstance(queue[0], list):
                if not all(isinstance(q, list) and len(q) == len(queue[0]) for q in queue):
                    raise ValueError('Legal lists must be homogeneous')
                queue = sum(queue, [])  # cancel out 1 level of [] in it, elements become their child

            if not all(isinstance(v, Number) for v in queue):
                raise TypeError('Only numbers or lists are allowed in a legal list')

            self.value = init_value
            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]
            self.shape = self.check_shape()

        elif isinstance(init_value, Number):
            if outer_shape:
                self.shape = tuple(outer_shape)
                self.value = init_value
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]
            else:
                self.shape = (1,)
                self.value = [init_value]

    def flatten(self) -> list:
        _flatten_list = []

        def _flatten(lst):
            if isinstance(lst[0], list):
                [_flatten(l) for l in lst]
            else:
                _flatten_list.extend(lst)

        _flatten(self.value)
        return _flatten_list

    def check_shape(self) -> tuple:
        arr = self.value
        shape = ()
        while isinstance(arr, list):
            shape += (len(arr),)
            arr = arr[0]
        self.shape = shape
        return shape

    def broadcast_with(self, other: Number | ListAlike) -> tuple:
        """
        Try to broadcast between 2 arrays. If each dim are equivalent or =1 then they can broadcast
        :param other: The other number or list-like numbers
        :return: If broadcastable : The shape of the simplest broadcast shape
                 Otherwise : None
        """
        shape1 = self.shape
        shape2 = other.shape if isinstance(other, array) else array(other).shape
        if len(shape2) > len(shape1):
            temp = shape1
            shape1 = shape2
            shape2 = temp
        residual = len(shape1) - len(shape2)
        result = () + shape1[:residual]

        for d1, d2 in zip(shape1[residual:], shape2):
            if d1 == d2 or d2 == 1:
                result += (d1,)
            elif d1 == 1:
                result += (d2,)
            else:
                raise TypeError(f'Cannot broadcast between {shape1} and {shape2}, miss matched at {d1} and {d2}')
        return result

    def elementwise(self, op, other: array = None) -> array:
        """
        Performs elementwise operation on 1 or 2 arrays
        :param op: [+, -, *, /, -(neg), abs]
        :param other: Another list
        :return: The result array
        """
        if not other:
            return self._elementwise_unary(op)

        broadcast_shape = self.broadcast_with(other)
        v1 = self.value
        v2 = other.value
        for _ in range(len(broadcast_shape) - len(self.shape)):
            v1 = [v1]
        for _ in range(len(broadcast_shape) - len(other.shape)):
            v2 = [v2]
        padded_shape1 = (1,) * (len(broadcast_shape) - len(self.shape)) + self.shape
        padded_shape2 = (1,) * (len(broadcast_shape) - len(other.shape)) + other.shape

        broadcast_index = [s-1 for s in broadcast_shape][:-1]
        padded_index1 = [s-1 for s in padded_shape1][:-1]
        padded_index2 = [s-1 for s in padded_shape2][:-1]
        indexes = [0] * (len(broadcast_index))
        fuse = 1
        res = []
        for d in reversed(broadcast_shape[:-1]):  # ex: shape=[2, 3, 4, 5], i=[2, 1, 0]
            fuse *= d
            temp = res
            res = [deepcopy(temp) for _ in range(d)]  # prep the shell for most inner layer

        print(res)

        cnt = 0
        while cnt < fuse:
            pointer1 = v1
            pointer2 = v2
            for i, i1, i2 in zip(indexes, padded_index1, padded_index2):
                pointer1 = pointer1[min(i, i1)]
                pointer2 = pointer2[min(i, i2)]
            p = res
            for i in indexes:
                p = p[i]
            p.extend([op(a, pointer2[0]) for a in pointer1] if len(pointer2) == 1
                     else [op(pointer1[0], b) for b in pointer2])

            # If input is number, index[-1] will raise outOfBound
            if cnt+1 >= fuse:
                break
            # Move to next index, check carry of each digit
            indexes[-1] += 1
            for i in reversed(range(len(indexes))):
                if i > 0 and indexes[i] > broadcast_index[i]:
                    indexes[i] = 0
                    indexes[i - 1] += 1
            cnt += 1

        res_arr = array(0)
        res_arr.value = res
        res_arr.shape = broadcast_shape
        return res_arr

    def _elementwise_unary(self, op) -> array:
        res_arr = array(0)
        res_arr.shape = self.shape
        res_arr.value = deepcopy(self.value)
        pointer = [res_arr.value]
        for _ in res_arr.shape[:-1]:
            new_pointer = []
            for sub in pointer:
                new_pointer += sub
            pointer = new_pointer
        for i in range(len(pointer)):
            for j in range(len(pointer[i])):
                pointer[i][j] = op(pointer[i][j])
        return res_arr

    def __add__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return self.elementwise(operator.add, other_arr)

    def __radd__(self, other: Number | ListAlike) -> array:
        return self + other

    def __sub__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return self.elementwise(operator.sub, other_arr)

    def __rsub__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return other_arr - self

    def __mul__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return self.elementwise(operator.mul, other_arr)

    def __rmul__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return other_arr * self

    def __truediv__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return self.elementwise(operator.truediv, other_arr)

    def __rtruediv__(self, other: Number | ListAlike) -> array:
        other_arr = array(other)
        return other_arr / self

    def __neg__(self) -> array:
        return self.elementwise(operator.neg)

    def __abs__(self) -> array:
        return self.elementwise(operator.abs)

    def exp(self) -> array:
        return self.elementwise(math.exp)

    def log(self) -> array:
        return self.elementwise(math.log)


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
    log_tensor = tensor(v_tensor.arr.log())
    log_tensor.op_name = 'log'
    log_tensor.gradient_wrt_parent = [1.0 / v_tensor.arr]
    log_tensor.add_parent(v_tensor)
    v_tensor.add_child(log_tensor)
    return log_tensor


def exp(v: tensor | Number):
    v_tensor = v if isinstance(v, tensor) else tensor(v)
    exp_tensor = tensor(v_tensor.arr.exp())
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
