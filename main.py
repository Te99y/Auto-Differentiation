import operator
import AutoDiff as ad
import numpy as np

if __name__ == '__main__':
    # a = ad.array([1, 2, 3], outer_shape=(3, ))
    # b = ad.array([-1, 0, 1], outer_shape=(3, ))
    # t1 = ad.tensor(a)
    # t2 = ad.tensor(b)
    #
    # y = (t1 + t2) @ t1
    # seed_dict = {t1: ad.array(0.0, t1.shape), t2: ad.array(0.0, t2.shape)}
    # for seed in ad.one_hot_perms(t1.shape):
    #     seed_dict[t1] = seed
    #     print(ad.jvp(y, None, seed_dict))

    a = np.arange(2*3*4).reshape((1, 2, 3, 4))
    # a = np.array([[[[0], [1]], [[2], [3]], [[4], [5]]], [[[0], [1]], [[2], [3]], [[4], [5]]]])
    # a = np.array([[[[1, 2, 3]]]])
    a_np = a.flatten()
    a_ad = ad.flatten(a.tolist())
    # aT = np.swapaxes(a, -1, -3)
    # a = np.random.randint(low=0, high=5, size=(1, 2, 3, 4))

    print()
    print(a.shape)
    print(a_np.shape)
    print(a_np.tolist())
    # print(a_np.flatten())
    print(a_ad)
    print(a_np.tolist() == a_ad)
