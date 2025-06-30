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

    a = ad.array([1, -2, -3], outer_shape=(4, 3, 2)).value
    # a = [1, 2, 1, 2, 3]
    b = [10, 2, 3]
    print(ad.transpose(a) == np.transpose(a).tolist())
