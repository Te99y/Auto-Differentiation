import operator

import AutoDiff as ad
import numpy as np

if __name__ == '__main__':

    # a = ad.array([[3, 2, 1], [10, 20, 30]], outer_shape=(3, 1,))
    # print(f'a.value : {a.value}')
    # print(f'a.shape : {a.shape}')
    # print(f'a.check shape : {a.check_shape()}')
    # print()
    #
    # b = ad.array([3], outer_shape=[2, 3, 3, 1])
    # print(f'b.value : {b.value}')
    # print(f'b.shape : {b.shape}')
    # print(f'b.check shape : {b.check_shape()}')
    # print()

    a = ad.array([[1, 2, 3], [11, 12, 13]])
    b = ad.array([[[-9, -1, -9], [-9, -1, -9], [-9, -1, -9]]])

    t1 = ad.tensor(a)
    t2 = ad.tensor(b)

    print(t1.abs())
    print(t2.abs())
    # print(b @ a)

    c = ad.array([1, 2, 3], outer_shape=(3, 2, 1))
    # print(ad.depth(0))
    print(ad.matmul([[1], [4]], [[1, 1]]))

    # y = ((t1 - t2).exp() - t3.abs().log()) + 2*t3
    # y = ((t1 - t2).log()).neg().abs()
    #
    # print(f't1: {t1}')
    # print(f't2: {t2}')
    # print(f't3: {t3}')
    # print(f' y: {y}')
    # # y._propagate_tan()
    # print()

    # order, roots = y.grad_forward_mode()
    # print('\n'.join(t.__str__() for t in order))
    # print()
    # print('\n'.join(t.__str__() for t in roots))
    # print()
    #
    # t1._tangent = ad.array([1, 1, 1])
    # t2._tangent = ad.array([0, 0, 0])
    # t3._tangent = ad.array([10, 10, 10])
    # for t in order:
    #     t._prop_tan()
    # print(y._tangent)

    #
    # ad.TENSOR_MAP = []
    # x1 = ad.tensor(a)
    # x2 = ad.tensor(b)
    # y = 1.0 / (1.0 + (-(x1.log() + x1*x2 - x2.sin())).exp())
    #
    # print(f'x1: {x1}')
    # print(f'x2: {x2}')
    # print(f' y: {y}')
    # print()
    #
    # order, roots = y.grad_forward_mode()
    # print('\n'.join(t.__str__() for t in order))
    # print()
    # print('\n'.join(t.__str__() for t in roots))

    # c = -b + a
    # print(f'testing : ')
    # print(c.value)
    # c = abs(c)
    # # print(f'testing : ')
    # print(c.value)
    #
    # c = 0.5 + c
    # print(f'c : {c.value}')
    # d = c.log()
    # print(f'd = c.log()')
    # print(f'd : {d.value}')
    # print(f'c : {c.value}')
    # e = d.abs()
    # print(f'e = d.abs()')
    # print(f'e : {e.value}')
    # print(f'd : {d.value}')
    # f = e.log()
    # print(f'f = e.log()')
    # print(f'f : {f.value}')
    # print(f'e : {e.value}')

    # aNP = np.array(a.value)
    # bNP = np.array(b.value)
    # cNP = np.matmul(aNP, np.transpose(aNP))
    # cNP = np.matmul(aNP, np.transpose(aNP))
    # print(list(cNP.flatten()) == c.flatten())



