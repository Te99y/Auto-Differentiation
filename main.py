import AutoDiff as ad
import numpy as np

if __name__ == '__main__':

    a = ad.array([[3, 2, 1], [10, 20, 30]], outer_shape=(3, 1,))
    a.value[0][0][0][0] = 99
    print(f'a.value : {a.value}')
    print(f'a.shape : {a.shape}')
    print()

    b = ad.array([3], outer_shape=[2, 3, 3, 1])
    print(f'b.value : {b.value}')
    print(f'b.shape : {b.shape}')
    print()

    c = a + b
    print(f'testing : ')
    print(c.value)

    a = np.array(a.value)
    b = np.array(b.value)
    print(c.value == (a+b))
    # c = [0, 1, 2, 3]
    # d = [c]*3
    # d = c
    # print(d)
    # d = [[d]*2]
    # d[0] = 1
    # print(c)
    # print(d)



