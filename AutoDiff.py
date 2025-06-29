from __future__ import annotations
import math
import operator
from typing import Union
from copy import deepcopy

Number = Union[int | float]
ListLike = Union['array', 'tensor', list]


class array:
    """
    An array stores a nd-list of numbers as its value, and its shape is the shape of the list.
    Natively float64 is used, so it holds
    The class that does actual numerical operations.
    Currently, if an inplace operation is called, a new list is assigned to self.value
    """
    def __init__(self, init_value: Number | ListLike, outer_shape: list | tuple = ()) -> None:
        self.shape: tuple
        self.value: list
        outer_shape = tuple(outer_shape)
        if isinstance(init_value, tensor):
            self.shape = deepcopy(init_value.arr.shape)
            self.value = deepcopy(init_value.arr.value)

        elif isinstance(init_value, array):
            self.shape = tuple(outer_shape) + init_value.shape
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
                queue = sum(queue, [])  # cancel out 1 level of [] in it, elements become their child

            if not all(isinstance(v, Number) for v in queue):
                raise TypeError('Only numbers or lists are allowed in a legal list')

            self.value = init_value
            if outer_shape:
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]
            self.shape = outer_shape + inner_shape

        elif isinstance(init_value, Number):
            if outer_shape:
                self.shape = tuple(outer_shape)
                self.value = init_value
                for s in reversed(outer_shape):
                    self.value = [deepcopy(self.value) for _ in range(s)]
            else:
                self.shape = (1,)
                self.value = [init_value]
        else:
            print(type(init_value))
            print(init_value)
            raise ValueError('How did you even get here?')

    def flatten(self) -> array:
        return array(_flatten(self.value))

    def transpose(self) -> array:
        return array(transpose(self.value))

    def update_shape(self) -> tuple:
        arr = self.value
        shape = ()
        while isinstance(arr, list):
            shape += (len(arr),)
            arr = arr[0]
        self.shape = shape
        return shape

    def broadcast_with(self, other: Number | ListLike) -> tuple:
        # Refine this function
        """
        Try to broadcast between 2 arrays. If each dim are equivalent or =1 then they can broadcast
        :param other: The other number or list-like numbers
        :return: If broadcastable : The shape of the simplest broadcast shape.
                 Otherwise : None
        """
        shape1 = self.shape
        shape2 = other.shape if isinstance(other, array) else array(other).shape
        if len(shape2) > len(shape1):  # keep shape1 longer then shape2
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

    def elementwise(self, op, other: array = None, inplace=False) -> array:
        if not other: return self._elementwise_unary(op)
        result_array = self if inplace else array(0)
        result_array.value = binary_elementwise(self.value, other.value, op)
        result_array.shape = self.broadcast_with(other)

        return result_array

    # ==============================================================================================================
    # Before 20250619 the elementwise method was this crazy big chunk of code, and there would be error
    # on some edge cases. I'm grateful that I refactored it. It's beautiful now.
    #
    # def elementwise(self, op, other: array = None, inplace=False) -> array:
    #     """
    #     Performs elementwise operation on 1 or 2 arrays
    #
    #     :param op: [+, -, *, /, neg, abs, exp, log, cos, sin]
    #     :param other: Another list
    #     :param inplace: perform the op inplace or not
    #     :return: The result array
    #     """
    #     if not other:
    #         return self._elementwise_unary(op)
    #
    #     broadcast_shape = self.broadcast_with(other)
    #     v1 = self.value
    #     v2 = other.value
    #     for _ in range(len(broadcast_shape) - len(self.shape)):
    #         v1 = [v1]
    #     for _ in range(len(broadcast_shape) - len(other.shape)):
    #         v2 = [v2]
    #     padded_shape1 = (1,) * (len(broadcast_shape) - len(self.shape)) + self.shape
    #     padded_shape2 = (1,) * (len(broadcast_shape) - len(other.shape)) + other.shape
    #
    #     broadcast_index = [s - 1 for s in broadcast_shape][:-1]
    #     padded_index1 = [s - 1 for s in padded_shape1][:-1]
    #     padded_index2 = [s - 1 for s in padded_shape2][:-1]
    #     indexes = [0] * (len(broadcast_index))
    #     fuse = 1
    #     res = []
    #
    #     for d in reversed(broadcast_shape[:-1]):  # ex: shape=[2, 3, 4, 5], i=[2, 1, 0]
    #         fuse *= d
    #         temp = res
    #         res = [deepcopy(temp) for _ in range(d)]  # prep the shell for most inner layer
    #
    #     cnt = 0
    #     # This is definitely not a good approach
    #     while cnt < fuse:
    #         pointer1 = v1
    #         pointer2 = v2
    #         for i, i1, i2 in zip(indexes, padded_index1, padded_index2):
    #             pointer1 = pointer1[min(i, i1)]
    #             pointer2 = pointer2[min(i, i2)]
    #         p = res
    #         for i in indexes:
    #             p = p[i]  # find the placeholder vector
    #         p.extend(  # extend with a whole vector
    #             [op(a, pointer2[0]) for a in pointer1] if len(pointer2) == 1
    #             else [op(pointer1[0], b) for b in pointer2] if len(pointer2) == 1
    #             else [op(p1, p2) for p1, p2 in zip(pointer1, pointer2)])
    #
    #         # If input is number, index[-1] will raise outOfBound
    #         if cnt + 1 >= fuse:
    #             break
    #         # Move to next index, check carry of each digit
    #         indexes[-1] += 1
    #         for i in reversed(range(len(indexes))):
    #             if i > 0 and indexes[i] > broadcast_index[i]:
    #                 indexes[i] = 0
    #                 indexes[i - 1] += 1
    #         cnt += 1
    #
    #     # Currently if inplace is True, we just points self.value to new array
    #     # result_array = self if inplace else array(0)
    #     # result_array.value = res
    #     # result_array.shape = broadcast_shape
    #     # Turns out if we do the above, we won't be able to differentiate due to not creating a new tensor
    #     result_array = self if inplace else array(0)
    #     result_array.value = res
    #     result_array.shape = broadcast_shape
    #     return result_array
    # ==============================================================================================================

    def _elementwise_unary(self, op, inplace=False) -> array:
        result_array = self if inplace else array(0)
        result_array.shape = self.shape

        #  ------------------ Non-recursive ------------------
        # result_array.value = deepcopy(self.value)
        # pointer = [result_array.value]
        # for _ in result_array.shape[:-1]:
        #     new_pointer = []
        #     for sub in pointer:
        #         new_pointer += sub
        #     pointer = new_pointer
        # for i in range(len(pointer)):
        #     for j in range(len(pointer[i])):
        #         pointer[i][j] = op(pointer[i][j])

        #  -------------------- Recursive --------------------
        def _ew(_lst, _op):
            if isinstance(_lst, list):
                return [_ew(s, _op) for s in _lst]
            return _op(_lst)

        result_array.value = _ew(self.value, op)
        return result_array

    def __str__(self):
        return f'shape:{self.shape}, value:{self.value}'

    def __add__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.add, other_arr)

    def __radd__(self, other: Number | ListLike) -> array:
        return self + other

    def __iadd__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.add, other_arr, inplace=True)

    def __sub__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.sub, other_arr)

    def __rsub__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return other_arr - self

    def __isub__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.sub, other_arr, inplace=True)

    def __mul__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.mul, other_arr)

    def __rmul__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return other_arr * self

    def __imul__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.mul, other_arr, inplace=True)

    def __truediv__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.truediv, other_arr)

    def __rtruediv__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return other_arr / self

    def __itruediv__(self, other: Number | ListLike) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        return self.elementwise(operator.truediv, other_arr, inplace=True)

    def __neg__(self) -> array:
        return self.elementwise(operator.neg)

    def __abs__(self) -> array:
        return self.elementwise(operator.abs)

    def __pow__(self, power: Number, modulo=None) -> array:
        power = array(power)
        return self.elementwise(operator.pow, power)

    def __rpow__(self, other):
        raise NotImplementedError('rpow with an array is not implemented')

    def __ipow__(self, power, modulo=None) -> array:
        return self.elementwise(operator.pow, power, inplace=True)

    def __matmul__(self, other) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        if self.shape[-1] != other_arr.shape[-2]:
            raise ValueError(f'The last dim of {self.shape} does not equal the second to last dim '
                             f'of the {other_arr.shape}. Refer to the signature (...,n,k),(...,k,m)->(...,n,m).')

        for dim1, dim2 in zip(reversed(self.shape[:-2]), reversed(other_arr.shape[:-2])):
            if dim1 != dim2 and dim1 != 0 and dim2 != 0:
                raise ValueError(f'Cannot broadcast between {self.shape} and {other_arr.shape}.')

        result_array = array(0)
        result_array.value = _matmul(self.value, other.value)
        result_array.update_shape()
        return result_array
    
    def __rmatmul__(self, other):
        other_arr = other if isinstance(other, array) else array(other)
        return other_arr @ self
    
    def __imatmul__(self, other) -> array:
        other_arr = other if isinstance(other, array) else array(other)
        if self.shape[-1] != other_arr.shape[-2]:
            raise ValueError(f'The last dim of {self.shape} does not equal the second to last dim '
                             f'of the {other_arr.shape}. Refer to the signature (...,n,k),(...,k,m)->(...,n,m).')

        for dim1, dim2 in zip(reversed(self.shape[2:]), reversed(other_arr.shape[2:])):
            if dim1 != dim2 and dim1 != 0 and dim2 != 0:
                raise ValueError(f'Cannot broadcast between {self.shape} and {other_arr.shape}.')

        self.value = _matmul(self.value, other.value)
        self.update_shape()
        return self

    def abs(self) -> array:
        return self.__abs__()

    def neg(self) -> array:
        return self.__neg__()

    def exp(self) -> array:
        return self.elementwise(math.exp)

    def log(self) -> array:
        return self.elementwise(math.log)

    def sin(self) -> array:
        return self.elementwise(math.sin)

    def cos(self) -> array:
        return self.elementwise(math.cos)


class tensor:
    """
    A tensor stores an array in it. Performing math operations on 1 or 2 tensors would create a new
    tensor. The 1 or 2 tensors will be the parents of the new tensor, and the new tensor would be the
    child of the 1 or 2 tensors. This relationship is essentially the computation graph.
    Currently, when an inplace operation is performed, a new tensor is created. This is done because
    passing differential would be easier. Otherwise, a version tracker is required.

    Since we define our computation graph by performing operations on tensors (aka we do a complete
    forward-pass), the graph is dynamic, meaning that after each value pass, just we discard it.
    Pytorch is based on dynamic-graph while TensorFlow and Jax implements static graph. In that
    case they would save the graph and reuse it, which would require the ._prop_val() function.
    """
    TENSOR_MAP: list[tensor] = []

    def __init__(self,
                 value: Number | list | array | tensor,
                 op_name: str = '   ',
                 requires_grad: bool = True,
                 is_leaf: bool = True):
        self.arr = array(value)
        self.shape = self.arr.shape
        self.op_name = op_name  # default leaf node = '   '
        self.requires_grad = requires_grad  # Allow gradients to flow through it
        self.is_leaf = is_leaf  # I like to call it root, but it's leaf as they appear on the tree structure
        self.tag = len(tensor.TENSOR_MAP)
        self.child: list[tensor] = []
        self.parent: list[tensor] = []
        self.tangent = array(0)
        self.gradient = array(0)
        self._prop_tan = lambda: None  # default do nothing
        self._prop_val = lambda: None  # default do nothing
        self._prop_grad = lambda: None  # default do nothing
        tensor.TENSOR_MAP.append(self)

    def topo(self, visited: set, order: list, roots: list) -> None:
        """
        This function is recursive.
        """
        if self.parent:
            for p in self.parent: None if p in visited else p.topo(visited, order, roots)  # noqa: E701
        elif self.is_leaf and self.requires_grad:
            roots.append(self)
        order.append(self)
        visited.add(self)

    def grad_fwd(self, y: tensor) -> None:
        """
        Calculate the gradient of self(usually an output) w.r.t every root with #param of forward sweeps.
        The intermediate tangents are stored in tensor._tangent along the tensors on the way.

        :return: None
        """
        visited = set()
        order: list[tensor] = []
        roots: list[tensor] = []
        self.topo(visited, order, roots)

        seed_dict = {root: array(0, root.shape) for root in roots}
        for r in roots: r.tangent = seed_dict[r]
        for seed in one_hot_perms(self.shape):
            self.tangent = seed
            seed_dict[self] = seed
            self.gradient = jvp(y, None, seed_dict)

    # def grad_bwd(self):

    def add_parent(self, *args: tensor) -> None:
        self.parent.extend([arg for arg in args])

    def add_child(self, *args: tensor) -> None:
        self.child.extend([arg for arg in args])

    @staticmethod
    def const_tensor(value: Number | list | array | tensor) -> tensor:
        return tensor(value, requires_grad=False, is_leaf=True)

    @staticmethod
    def intermediate_tensor(value: Number | list | array | tensor, op_name: str) -> tensor:
        return tensor(value, op_name=op_name, requires_grad=True, is_leaf=False)

    def __str__(self) -> str:
        parent_description = [(parent.op_name, parent.tag) for parent in self.parent]
        child_description = [(child.op_name, child.tag) for child in self.child]
        return f'{self.tag} | op:{self.op_name} | parent:{parent_description} | child:{child_description} | ' \
               f'shape:{self.arr.shape}' \
               f'\n    val:{self.arr}' \
               f'\n    tan:{self.tangent}'

    def __add__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        add_tensor = tensor.intermediate_tensor(self.arr + other_tensor.arr, op_name='add')
        add_tensor.add_parent(self, other_tensor)
        self.add_child(add_tensor)
        other_tensor.add_child(add_tensor)

        def _prop_val():
            add_tensor.arr = self.arr + other_tensor.arr
            add_tensor.shape = add_tensor.arr.shape
        add_tensor._prop_val = _prop_val
        def _prop_tan(): add_tensor.tangent = self.tangent + other_tensor.tangent
        add_tensor._prop_tan = _prop_tan
        def _prop_grad():
            self.gradient += add_tensor.gradient * 1.0
            other_tensor.gradient += add_tensor.gradient * 1.0
        add_tensor._prop_grad = _prop_grad
        return add_tensor

    def __radd__(self, other) -> tensor:
        return self + other

    def __iadd__(self, other) -> tensor:
        # other_tensor = other if isinstance(other, tensor) else tensor(other)
        # self.arr += other_tensor.arr
        # return self + other_tensor
        return self + other

    def __sub__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        sub_tensor = tensor.intermediate_tensor(self.arr - other_tensor.arr, op_name='sub')
        sub_tensor.add_parent(self, other_tensor)
        self.add_child(sub_tensor)
        other_tensor.add_child(sub_tensor)

        def _prop_val():
            sub_tensor.arr = self.arr - other_tensor.arr
            sub_tensor.shape = sub_tensor.arr.shape
        sub_tensor._prop_val = _prop_val
        def _prop_tan(): sub_tensor.tangent = self.tangent - other_tensor.tangent
        sub_tensor._prop_tan = _prop_tan
        def _prop_grad():
            self.gradient += sub_tensor.gradient * 1.0
            other_tensor.gradient += sub_tensor.gradient * -1.0
        sub_tensor._prop_grad = _prop_grad
        return sub_tensor

    def __rsub__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        return other_tensor - self

    def __isub__(self, other) -> tensor:
        # other_tensor = other if isinstance(other, tensor) else tensor(other)
        # self.arr -= other_tensor.arr
        # return self - other_tensor
        return self - other

    def __abs__(self) -> tensor:
        abs_tensor = tensor.intermediate_tensor(self.arr.abs(), op_name='abs')
        abs_tensor.add_parent(self)
        self.add_child(abs_tensor)

        def _prop_val(): abs_tensor.arr = abs(self.arr)
        abs_tensor._prop_val = _prop_val
        def _prop_tan(): abs_tensor.tangent = self.tangent * self.arr.elementwise(sign)
        abs_tensor._prop_tan = _prop_tan
        def _prop_grad(): self.gradient += abs_tensor.gradient * self.arr.elementwise(sign)
        abs_tensor._prop_grad = _prop_grad
        return abs_tensor

    def __neg__(self) -> tensor:
        neg_tensor = tensor.intermediate_tensor(-self.arr, op_name='neg')
        neg_tensor.add_parent(self)
        self.add_child(neg_tensor)

        def _prop_val(): neg_tensor.arr = -self.arr
        neg_tensor._prop_val = _prop_val
        def _prop_tan(): neg_tensor.tangent = -self.tangent
        neg_tensor._prop_tan = _prop_tan
        def _prop_grad(): self.gradient += -neg_tensor.gradient
        neg_tensor._prop_grad = _prop_grad
        return neg_tensor

    def __mul__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        mul_tensor = tensor.intermediate_tensor(self.arr * other_tensor.arr, op_name='mul')
        mul_tensor.add_parent(self, other_tensor)
        self.add_child(mul_tensor)
        other_tensor.add_child(mul_tensor)

        def _prop_val():
            mul_tensor.arr = self.arr * other_tensor.arr
            mul_tensor.shape = mul_tensor.arr.shape
        mul_tensor._prop_val = _prop_val
        def _prop_tan(): mul_tensor.tangent = other_tensor.arr * self.tangent + self.arr * other_tensor.tangent
        mul_tensor._prop_tan = _prop_tan
        def _prop_grad():
            self.gradient += mul_tensor.gradient * other_tensor.arr
            other_tensor.gradient += mul_tensor.gradient * self.arr
        mul_tensor._prop_grad = _prop_grad
        return mul_tensor

    def __rmul__(self, other) -> tensor:
        return self * other

    def __imul__(self, other) -> tensor:
        # other_tensor = other if isinstance(other, tensor) else tensor(other)
        # self.arr *= other_tensor.arr
        # return self * other_tensor
        return self * other

    def __truediv__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        div_tensor = tensor.intermediate_tensor(self.arr / other_tensor.arr, op_name='div')
        div_tensor.add_parent(self, other_tensor)
        self.add_child(div_tensor)
        other_tensor.add_child(div_tensor)

        def _prop_val():
            div_tensor.arr = self.arr / other_tensor.arr
            div_tensor.shape = div_tensor.arr.shape
        div_tensor._prop_val = _prop_val
        def _prop_tan():
            one_over_b = 1.0 / other_tensor.arr
            div_tensor.tangent = one_over_b * (self.tangent - self.arr * one_over_b * other_tensor.tangent)
        div_tensor._prop_tan = _prop_tan
        def _prop_grad():
            self.gradient += div_tensor.gradient / other_tensor.arr
            other_tensor.gradient += div_tensor.gradient * self.arr
        div_tensor._prop_grad = _prop_grad
        return div_tensor

    def __rtruediv__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        return other_tensor / self

    def __itruediv__(self, other) -> tensor:
        # other_tensor = other if isinstance(other, tensor) else tensor(other)
        # self.arr /= other_tensor.arr
        # return self / other_tensor
        return self / other

    def __matmul__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        matmul_tensor = tensor.intermediate_tensor(self.arr @ other_tensor.arr, op_name='mat')
        matmul_tensor.add_parent(self, other_tensor)
        self.add_child(matmul_tensor)
        other_tensor.add_child(matmul_tensor)

        def _prop_val():
            matmul_tensor.arr = self.arr @ other_tensor.arr
            matmul_tensor.shape = matmul_tensor.arr.shape
        matmul_tensor._prop_val = _prop_val
        def _prop_tan(): matmul_tensor.tangent = self.tangent * other_tensor.arr + self.arr * other_tensor.tangent
        matmul_tensor._prop_tan = _prop_tan
        def _prop_grad():
            self.gradient += matmul_tensor.gradient @ transpose(self.arr)
            other_tensor.gradient += transpose(other_tensor.arr) @ matmul_tensor.gradient
        matmul_tensor._prop_grad = _prop_grad
        return matmul_tensor

    def __rmatmul__(self, other) -> tensor:
        other_tensor = other if isinstance(other, tensor) else tensor.const_tensor(other)
        return other_tensor @ self

    def __imatmul__(self, other) -> tensor:
        return self @ other

    def update_shape(self) -> tuple:
        self.shape = self.arr.update_shape()
        return self.shape

    def abs(self) -> tensor:
        return self.__abs__()

    def neg(self) -> tensor:
        return self.__neg__()

    def exp(self) -> tensor:
        exp_tensor = tensor.intermediate_tensor(self.arr.exp(), op_name='exp')
        exp_tensor.add_parent(self)
        self.add_child(exp_tensor)

        def _prop_val(): exp_tensor.arr = self.arr.exp()
        exp_tensor._prop_val = _prop_val
        def _prop_tan(): exp_tensor.tangent = self.tangent * exp_tensor.arr
        exp_tensor._prop_tan = _prop_tan
        return exp_tensor

    def log(self) -> tensor:
        log_tensor = tensor.intermediate_tensor(self.arr.log(), op_name='log')
        log_tensor.add_parent(self)
        self.add_child(log_tensor)

        def _prop_val(): log_tensor.arr = self.arr.log()
        log_tensor._prop_val = _prop_val
        def _prop_tan(): log_tensor.tangent = self.tangent / self.arr
        log_tensor._prop_tan = _prop_tan
        return log_tensor

    def sin(self) -> tensor:
        sin_tensor = tensor.intermediate_tensor(self.arr.sin(), op_name='sin')
        sin_tensor.add_parent(self)
        self.add_child(sin_tensor)

        def _prop_val(): sin_tensor.arr = self.arr.sin()
        sin_tensor._prop_val = _prop_val
        def _prop_tan(): sin_tensor.tangent = self.arr.cos() * self.tangent
        sin_tensor._prop_tan = _prop_tan
        return sin_tensor

    def cos(self) -> tensor:
        cos_tensor = tensor.intermediate_tensor(self.arr.cos(), op_name='cos')
        cos_tensor.add_parent(self)
        self.add_child(cos_tensor)

        def _prop_val(): cos_tensor.arr = self.arr.cos()
        cos_tensor._prop_val = _prop_val
        def _prop_tan(): cos_tensor.tangent = -self.arr.sin() * self.tangent
        cos_tensor._prop_tan = _prop_tan
        return cos_tensor

    def pow(self, p: Number) -> tensor:
        pow_tensor = tensor.intermediate_tensor(pow(self.arr, p), op_name='pow')
        pow_tensor.add_parent(self)
        self.add_child(pow_tensor)

        def _prop_val(): pow_tensor.arr = self.arr.__pow__(p)
        pow_tensor._prop_val = _prop_val
        def _prop_tan():
            # pow_tensor._tangent = self.arr.elementwise(lambda x: 0 if x == 0 else pow(x, p-1)) * self._tangent
            pow_tensor.tangent = pow_tensor.arr / self.arr * self.tangent  # When x == 0 this will cause problem
        pow_tensor._prop_tan = _prop_tan
        return pow_tensor

    def flatten(self) -> tensor:
        flat_tensor = tensor.intermediate_tensor(flatten(self.arr), op_name='tra')
        flat_tensor.add_parent(self)
        self.add_child(flat_tensor)

        def _prop_val(): flat_tensor.arr = flatten(self.arr)
        flat_tensor._prop_val = _prop_val
        return flat_tensor

    def transpose(self) -> tensor:
        trans_tensor = tensor.intermediate_tensor(transpose(self.arr), op_name='tra')
        trans_tensor.add_parent(self)
        self.add_child(trans_tensor)

        def _prop_val():
            trans_tensor.arr = transpose(self.arr)
            trans_tensor.update_shape()
        trans_tensor._prop_val = _prop_val
        return trans_tensor


def all_tensors():
    print('\n'.join([t.__str__() for t in tensor.TENSOR_MAP], ))


def neg(v: tensor | Number) -> float | tensor:
    return v.neg() if isinstance(v, tensor) else -v


def log(v: tensor | Number) -> float | tensor:
    return v.log() if isinstance(v, tensor) else math.log(v)


def exp(v: tensor | Number) -> float | tensor:
    return v.exp() if isinstance(v, tensor) else math.exp(v)


def sin(v: tensor | Number) -> float | tensor:
    return v.sin() if isinstance(v, tensor) else math.sin(v)


def cos(v: tensor | Number) -> float | tensor:
    return v.cos() if isinstance(v, tensor) else math.cos(v)


def add(v1, v2: tensor | Number) -> float | tensor:
    return v1 + v2


def sub(v1, v2: tensor | Number) -> float | tensor:
    return v1 - v2


def mul(v1, v2: tensor | Number) -> float | tensor:
    return v1 * v2


def div(v1, v2: tensor | Number) -> float | tensor:
    return v1 / v2


def identity(shape: int) -> float | array:
    if shape < 1:
        raise ValueError('Identity matrix must be init with shape >= 1')
    return 1.0 if shape == 1 else array([[float(j == i) for j in range(shape)] for i in range(shape)])


def flatten(v: ListLike) -> ListLike:
    """
    This is a wrapper function
    """
    if isinstance(v, list):
        return _flatten(v)
    if isinstance(v, array) or isinstance(v, tensor):
        return v.flatten()
    else:
        raise ValueError(f'Input must be list|array|tensor, get {type(v)}')


def _flatten(v: list) -> list:
    """
    This function does not validate whether the input is homogeneous.
    Incompatible shape may produce unexpected output.
    """
    while isinstance(v[0], list):
        temp = []
        for v_i in v:
            temp.extend(v_i)
        v = temp
    return v


def transpose(v: ListLike) -> ListLike:
    """
    This is a wrapper function
    """
    if isinstance(v, list):
        return _transpose(v)
    if isinstance(v, array) or isinstance(v, tensor):
        return v.transpose()
    else:
        raise ValueError(f'Input must be list|array|tensor, get {type(v)}')


def _transpose(v: list) -> list:
    """
    This function does not validate whether the input is homogeneous.
    Incompatible shape may produce unexpected output.
    """
    shape = [len(v)]
    while isinstance(v[0], list):
        shape.append(len(v[0]))
        temp = []
        for v_i in v: temp.extend(v_i)
        v = temp

    stripe = len(v)
    for s in shape:
        stripe //= s
        v = [[v[i+j*stripe] for j in range(s)] for i in range(stripe)]
    return v[0]


def kronecker_product(v1: list | array, v2: list | array) -> ListLike:
    """
    This is a wrapper function
    """
    if isinstance(v1, list) and isinstance(v2, list):
        return _kronecker_product(v1, v2)
    if isinstance(v1, array) and isinstance(v2, array):
        return _kronecker_product(v1.value, v2.value)
    raise ValueError(f'Input types do not match. Got {type(v1)}, {type(v2)}')


def _kronecker_product(v1: list, v2: list) -> list:
    """
    This function does not validate whether the input is homogeneous.
    Incompatible shape may produce unexpected output.
    """
    return unary_elementwise(v1, lambda v1_i: binary_elementwise([v1_i], v2, operator.mul))
    # [v1_i] because bin_ew takes list as param


def sign(v: Number) -> float:
    # return 2.0*(v > 0) - (v != 0)
    return 1.0 if v > 0 else -1.0 if v < 0 else 0.0


def one_hot_perms(shape: tuple):
    """
    This is a generator
    """
    seed = array(0.0, shape)
    p = [[seed.value]]
    while isinstance(p[0][0], list):
        new_p = []
        for pi in p:
            new_p.extend(pi)
        p = new_p
    for pi in p:
        for j in range(len(pi)):
            pi[j] = 1.0
            yield seed
            pi[j] = 0.0


def check_shape(v: ListLike | Number) -> tuple:
    """
    This is a wrapper function
    """
    if isinstance(v, Number): return ()
    if isinstance(v, array | tensor): return v.update_shape()
    if isinstance(v, list): return _check_shape(v)
    else: raise ValueError(f'Input type {type(v)} is not allowed.')


def _check_shape(v: list) -> tuple:
    """
    This function does not validate whether the input is homogeneous.
    If the input contains an empty list, an IndexError might be thrown.
    """
    res = (len(v), )
    while isinstance(v[0], list):
        res += (len(v[0]), )
        v = v[0]
    return res


def broadcast(shape1: tuple, shape2: tuple) -> tuple:
    res_shape = ()
    for d1, d2 in zip(reversed(shape1), reversed(shape2)):
        if d1 == d2 or d2 == 1:
            res_shape += (d1,)
        elif d1 == 1:
            res_shape += (d2,)
        else:
            raise TypeError(f'Cannot broadcast between {shape1} and {shape2}, miss matched at {d1} and {d2}')

    return shape1[:-len(shape2)] + tuple(reversed(res_shape)) if len(shape1) > len(shape2) else shape2[:-len(shape1)] + tuple(reversed(res_shape))


def depth(v: list | Number) -> int:
    """
    This function does not validate whether the input is homogeneous.
    If the input contains an empty list, an IndexError might be thrown.
    """
    cnt = 0
    while isinstance(v, list):
        v = v[0]
        cnt += 1
    return cnt


def unary_elementwise(x: list, op) -> list:
    if isinstance(x[0], Number): return [op(x_i) for x_i in x]
    return [unary_elementwise(x_i, op) for x_i in x]


def binary_elementwise(v1: list, v2: list, op) -> list:
    """
    This function does not validate whether the inputs are homogeneous or broadcastable.
    Incompatible shapes may produce unexpected output.
    """
    if isinstance(v1[0], Number) and isinstance(v2[0], Number):
        return [op(v1[min(i, len(v1)-1)], v2[min(i, len(v2)-1)]) for i in range(max(len(v1), len(v2)))]
    d1, d2 = depth(v1), depth(v2)
    if d1 == d2: return [binary_elementwise(v1[min(i, len(v1)-1)], v2[min(i, len(v2)-1)], op) for i in range(max(len(v1), len(v2)))]
    if d1 > d2: return [binary_elementwise(v1_i, v2, op) for v1_i in v1]
    return [binary_elementwise(v1, v2_i, op) for v2_i in v2]


def matmul(v1: ListLike, v2: ListLike) -> ListLike:
    """
    This is a wrapper function
    """
    if isinstance(v1, list) and isinstance(v2, list):
        return _matmul(v1, v2)
    if isinstance(v1, array) and isinstance(v2, array):
        return v1 @ v2
    if isinstance(v1, tensor) and isinstance(v2, tensor):
        return v1 @ v2
    raise ValueError(f'Input types do not match. Got {type(v1)}, {type(v2)}')


def _matmul(x1: list, x2: list) -> list:
    """
    This function does not validate whether the inputs are homogeneous or broadcastable.
    Incompatible shapes may produce unexpected output.\n

    This is essentially elementwise, if we think of the last 2-dim matrices as elements
    of a list with shape (batch_dim, ).
    """
    d1, d2 = depth(x1), depth(x2)
    if d1 <= 2 and d2 <= 2:
        x1 = x1 if isinstance(x1[0], list) else [x1]
        x2 = x2 if isinstance(x2[0], list) else [x2]
        return [[sum(binary_elementwise(row, [x2_row[m] for x2_row in x2], operator.mul)) for m in range(len(x2[0]))] for row in x1]
    if d1 == d2: return [_matmul(x1[min(i, len(x1)-1)], x2[min(i, len(x2)-1)]) for i in range(max(len(x1), len(x2)))]
    if d1 > d2: return [_matmul(x1_i, x2) for x1_i in x1]
    return [_matmul(x1, x2_i) for x2_i in x2]


def jvp(f: tensor, inputs: None | dict[tensor, array], directions: dict[tensor, array]):
    """
    Calculate the JVP where J is the |inputs|x|output| Jacobian, V is the direction vector.
    The intermediate tangents are stored in tensor._tangent along the tensors on the way.\n
    From JAX: jax.jvp(f, primals=(x,), tangents=(v,)), so the x = inputs, v = directions

    What it does:
    Given a tensor, build topology order and find the root tensors.
    From the root tensors, start to propagate value and tangent forward.
    It will eventually reach f, and the propagation stops.
    Returns the ._tangent recorded at f.

    :param f: The function to differentiate
    :param inputs: The value of the roots
    :param directions: A vector of n elements to perform JVP with.
                      The linear combination of Jacobian columns you're interested in.
    :return: None
   """
    visited = set()
    order: list[tensor] = []
    roots: list[tensor] = []
    f.topo(visited, order, roots)
    if len(roots) > len(directions) or (inputs is not None and len(roots) > len(inputs)):
        raise ValueError('Number of roots does not match with length of inputs/directions')

    if inputs is not None:
        for root in roots: root.arr = array(inputs[root])
        for t in order: t._prop_val()
    for root in roots: root.tangent = array(directions[root])
    for t in order: t._prop_tan()

    return f.tangent

