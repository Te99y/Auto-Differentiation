import AutoDiff as ad
import numpy as np

if __name__ == '__main__':

    a = ad.array([[3, 2], [3, 2]], outer_shape=(2,))
    print(a.value)
    print(a.shape)
    print(a.shape[-1])
    print()

    b = ad.array([2, 1])
    print(b.value)
    print(b.shape)
    print(b.shape[-1])
    print()

    shape1 = list(a.shape)
    shape2 = list(b.shape)
    print(shape1)
    print(shape2)
    print()

    x = np.zeros(shape=(2, 1, 3))
    y = np.ones(shape=(2, 3))
    z = x - y
    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(z))
