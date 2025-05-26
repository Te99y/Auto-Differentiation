import AutoDiff as ad
import numpy as np

if __name__ == '__main__':

    a = ad.array([[3, 2, 1], [10, 20, 30]], outer_shape=(3, 1,))
    a.value[0][0][0][0] = 99
    print(f'a.value : {a.value}')
    print(f'a.shape : {a.shape}')
    print()

    b = ad.array([3], outer_shape=[2, 3, 3, 1])
    b = ad.array(3)
    print(f'b.value : {b.value}')
    print(f'b.shape : {b.shape}')
    print()

    c = -b + a
    print(f'testing : ')
    print(c.value)

    # aNP = np.array(a.value)
    # bNP = np.array(b.value)
    # cNP = aNP * bNP
    # print(list(cNP.flatten()) == c.flatten())



