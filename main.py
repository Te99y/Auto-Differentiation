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

    # a = ad.array([[1, 2, 3], [11, 12, 13]])
    # b = ad.array([[[-9, -1, -9], [-9, -1, -9], [-9, -1, -9]]])
    #
    # t1 = ad.tensor(a)
    # t2 = ad.tensor(b)
    #
    # print(t1.abs())
    # print(t2.abs())
    # print(b @ a)

    np.random.seed(0)
    l = np.random.randint(low=1, high=6)
    size = np.random.randint(low=1, high=5, size=l)
    [n, k, m] = np.random.randint(low=1, high=5, size=3)
    c = ad.array(np.random.rand(*size, n, k).tolist())
    d = ad.array(np.random.rand(*size, k, m).tolist())
    # c = ad.array([[300.38320773, -319.54941813, -463.56006039, -342.67056251, -115.50972527, 99.65008376, -92.66947929,
    #                -258.66999994],
    #               [348.27260333, -238.07690764, -285.38546076, -371.71087571, 427.15175295, -220.25261214,
    #                -498.92294765, -321.72106065],
    #               [339.16841535, 471.93951976, 346.20681176, -188.24184105, 410.12903769, -334.46931805, 98.10095229,
    #                -94.29076193],
    #               [143.92434899, -209.30501581, 238.22961315, 63.27208654, -364.49817926, -443.29309457, 104.03532852,
    #                272.72285255]])
    # d = ad.array([[112.93923888, 167.73989173, 251.91529553, 311.47921457, 360.97651274, 348.05459086, 93.05462098,
    #                483.44348472],
    #               [234.03233164, 63.04080978, 300.22546738, 174.79073029, 244.84752549, 460.18327631, 457.1559201,
    #                405.32965854],
    #               [291.5257906, 418.59462239, 200.18691929, 192.74552789, 48.07161118, 303.49283775, 6.86419619,
    #                174.37188119],
    #               [457.45511254, 187.88592825, 156.5836226, 357.7283956, 161.68338363, 146.59457066, 101.09833252,
    #                93.6738532]])
    # c = ad.array(np.random.randint(low=-1, high=1, size=(4, 4, 4)).tolist())
    # d = ad.array(np.random.randint(low=-1, high=1, size=(4, 8)).tolist())
    np_mat = np.array(c.value) @ np.array(d.value)
    ad_mat = c @ d
    # np_mat = np.array(c.value) + np.array(d.value)
    # ad_mat = c + d
    # ad_mat = ad.array(ad.binary_elementwise(c.value, d.value, operator.add))
    print(f'c: {c}')
    print(f'd: {d}')
    print(f'np_mat.shape: {np.shape(np_mat)}')
    print(f'ad_mat.shape: {ad_mat.check_shape()}')
    print('np mat:')
    # print(np_mat)
    print()
    print('ad mat:')
    # print(ad_mat.value[1])
    print()
    print(f'np=ad: {np.array(ad_mat.value, dtype=np.float64).tolist() == np_mat.tolist()}')
    print(np.allclose(np.array(ad_mat.value), np_mat))
    print()

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
