import AutoDiff as ad
import numpy as np

if __name__ == '__main__':

    a = ad.array([[3, 2, 1], [10, 20, 30]], outer_shape=(3, 1,))
    print(f'a.value : {a.value}')
    print(f'a.shape : {a.shape}')
    print()

    b = ad.array([3], outer_shape=[2, 3, 3, 1])
    print(f'b.value : {b.value}')
    print(f'b.shape : {b.shape}')
    print()

    c = a.broadcast_with(b)
    print(f'broadcast shape : {c}')
    print(f'padded  a.shape : {(1, )*(len(c) - len(a.shape)) + a.shape}')
    print(f'padded  b.shape : {tuple([1]*(len(c) - len(b.shape))) + b.shape}')
    print(f'testing : ')
    print(a+b)

    # a = np.array(a.value)
    # b = np.array(b.value)
    # # print(a+b)
    # c = [0, 1, 2, 3]
    # d = [c]*3
    # d = c
    # print(d)
    # d = [[d]*2]
    # d[0] = 1
    # print(c)
    # print(d)



