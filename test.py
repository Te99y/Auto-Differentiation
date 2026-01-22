import math
import operator
import unittest
import numpy as np
import numpy.testing as npt
import AutoDiff as ad
import coverage
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # to match python-native float64
OPS = {
    'unary': [operator.abs, operator.neg, math.exp, math.log, math.sin, math.cos],
    'binary': [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv
    ]
}
np.random.seed(seed=20250614)
ITER = 1

jnp_unary = [
    jnp.abs,
    jnp.negative,
    jnp.exp,
    jnp.log,
    jnp.sin,
    jnp.cos
]
jnp_binary = [
    jnp.add,
    jnp.subtract,
    jnp.multiply,
    jnp.true_divide,
]
jnp_to_tensor_map = {
    jnp.negative:    ad.neg,
    jnp.abs:         abs,
    jnp.exp:         ad.exp,
    jnp.log:         ad.log,
    jnp.sin:         ad.sin,
    jnp.cos:         ad.cos,

    jnp.add:         ad.add,
    jnp.subtract:    ad.sub,
    jnp.multiply:    ad.mul,
    jnp.true_divide: ad.div,
    jnp.matmul:      ad.matmul
}


cov = coverage.Coverage()
cov.start()


class test_array(unittest.TestCase):
    # def test_init(self):
    #     a = ad.array(0.0)
    #     self.assertEqual(a.value, [0.0])
    #     self.assertEqual(a.shape, (1, ))
    #     a = ad.array(0)
    #     self.assertEqual(a.value, [0])
    #     self.assertEqual(a.shape, (1, ))
    #     a = ad.array([0])
    #     self.assertEqual(a.value, [0])
    #     self.assertEqual(a.shape, (1, ))
    #
    #     a = ad.array([0, 1, 2.0, -9])
    #     self.assertEqual(a.value, [0, 1, 2.0, -9])
    #     self.assertEqual(a.shape, (4, ))
    #     a = ad.array([0, 1, 2.0, -9], outer_shape=())
    #     self.assertEqual(a.value, [0, 1, 2.0, -9])
    #     self.assertEqual(a.shape, (4, ))
    #
    #     a = ad.array(2.1, outer_shape=(2, 2, 4))
    #     self.assertEqual(a.value, [
    #         [[2.1, 2.1, 2.1, 2.1], [2.1, 2.1, 2.1, 2.1]],
    #         [[2.1, 2.1, 2.1, 2.1], [2.1, 2.1, 2.1, 2.1]]
    #     ])
    #     self.assertEqual(a.shape, (2, 2, 4))
    #     a = ad.array([2.1], outer_shape=(2, 2, 4))
    #     self.assertEqual(a.value, [
    #         [[[2.1], [2.1], [2.1], [2.1]], [[2.1], [2.1], [2.1], [2.1]]],
    #         [[[2.1], [2.1], [2.1], [2.1]], [[2.1], [2.1], [2.1], [2.1]]]
    #     ])
    #     self.assertEqual(a.shape, (2, 2, 4, 1))
    #
    #     b = ad.array(a)
    #     self.assertEqual(a.value, b.value)
    #     self.assertEqual(a.shape, b.shape)
    #
    # def test_flatten(self):
    #     a = np.random.randn(256, 256)
    #     _a = ad.array(a.tolist())
    #     self.assertEqual(a.flatten().tolist(), _a.flatten().value)
    #
    # def test_check_shape(self):
    #     shape = np.random.randint(low=1, high=5, size=5)
    #     a = np.random.randn(*shape)
    #     _a = ad.array(a.tolist())
    #     self.assertEqual(tuple(shape), _a.update_shape())
    #
    # def test_broadcast(self):
    #     shape = np.random.randint(low=1, high=5, size=5)
    #     a = np.random.randn(*shape)
    #     _a = ad.array(a.tolist())
    #     self.assertEqual(tuple(shape), _a.update_shape())
    #
    # def test_swapaxes(self):
    #     size = np.random.randint(low=1, high=5, size=10)
    #     a = np.random.randint(low=-10, high=10, size=size)
    #     _a = np.swapaxes(a, -1, -2).tolist()
    #     _aT = ad.swapaxes(a.tolist(), -1, -2)
    #     self.assertEqual(_a, _aT)
    #
    # def test_transpose(self):
    #     size = np.random.randint(low=1, high=5, size=10)
    #     a = np.random.randint(low=-10, high=10, size=size)
    #     _a = ad.array(a.tolist())
    #     _aT = ad.transpose(_a).value
    #     self.assertEqual(np.transpose(a).tolist(), _aT)
    #
    # def test_ops(self):
    #     for _ in range(ITER):
    #         l = np.random.randint(low=1, high=6)
    #         size = np.random.randint(low=2, high=20, size=l)
    #         a = np.random.uniform(low=-500, high=500, size=size)
    #         random_drop_size = size[np.random.randint(low=0, high=l):]
    #         ones_mask = np.random.rand(*random_drop_size.shape) < 0.5
    #         random_drop_size[ones_mask] = 1
    #         b = np.random.uniform(low=0.001, high=500, size=random_drop_size)
    #         _a = ad.array(a.tolist())
    #         _b = ad.array(b.tolist())
    #         # Test for unary ops needs class specific functions
    #         self.assertEqual(np.abs(a).tolist(), _a.abs().value)
    #         self.assertEqual(np.negative(a).tolist(), _a.neg().value)
    #         self.assertEqual(np.exp(a).tolist(), _a.exp().value)
    #         self.assertEqual(np.log(b).tolist(), _b.log().value)
    #         self.assertEqual(np.sin(a).tolist(), _a.sin().value)
    #         self.assertEqual(np.cos(a).tolist(), _a.cos().value)
    #         # Binary ops
    #         self.assertTrue(all(op(a, b).tolist() == op(_a, _b).value for op in OPS['binary']))
    #         # Matmul
    #         [n, k, m] = np.random.randint(low=1, high=5, size=3)
    #         mat1 = ad.array(np.random.rand(*size[0:4], n, k).tolist())
    #         mat2 = ad.array(np.random.rand(*size[0:4], k, m).tolist())
    #         np_matmul = np.array(mat1.value, dtype=np.float64) @ np.array(mat2.value, dtype=np.float64)
    #         ad_matmul = mat1 @ mat2
    #         self.assertEqual(np.array(np_matmul, dtype=np.float32).tolist(),
    #                          np.array(ad_matmul.value, dtype=np.float32).tolist())

    def test_reshape(self):
        pass

# class test_tensor(unittest.TestCase):
#     def setUp(self) -> None:
#         ad.tensor.TENSOR_MAP.clear()
#
#     def test_init(self):
#         a = ad.array(0.0)
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [0.0])
#         self.assertEqual(b.shape, (1, ))
#         a = ad.array(0)
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [0])
#         self.assertEqual(b.shape, (1, ))
#         a = ad.array([0])
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [0])
#         self.assertEqual(b.shape, (1, ))
#
#         a = ad.array([0, 1, 2.0, -9])
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [0, 1, 2.0, -9])
#         self.assertEqual(b.shape, (4, ))
#         a = ad.array([0, 1, 2.0, -9], outer_shape=())
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [0, 1, 2.0, -9])
#         self.assertEqual(b.shape, (4, ))
#
#         a = ad.array(2.1, outer_shape=(2, 2, 4))
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [
#             [[2.1, 2.1, 2.1, 2.1], [2.1, 2.1, 2.1, 2.1]],
#             [[2.1, 2.1, 2.1, 2.1], [2.1, 2.1, 2.1, 2.1]]
#         ])
#         self.assertEqual(b.shape, (2, 2, 4))
#         a = ad.array([2.1], outer_shape=(2, 2, 4))
#         b = ad.tensor(a)
#         self.assertEqual(b.arr.value, [
#             [[[2.1], [2.1], [2.1], [2.1]], [[2.1], [2.1], [2.1], [2.1]]],
#             [[[2.1], [2.1], [2.1], [2.1]], [[2.1], [2.1], [2.1], [2.1]]]
#         ])
#         self.assertEqual(b.shape, (2, 2, 4, 1))
#
#         b = ad.array(a)
#         c = ad.tensor(b)
#         self.assertEqual(b.value, c.arr.value)
#         self.assertEqual(b.shape, c.shape)
#
#     def test_op(self):
#         for _ in range(ITER):
#             l = np.random.randint(low=1, high=6)
#             size = np.random.randint(low=1, high=20, size=l)
#             random_drop_size = size[np.random.randint(low=0, high=l):]
#             ones_mask = np.random.rand(*random_drop_size.shape) < 0.5
#             random_drop_size[ones_mask] = 1
#
#             a = np.random.uniform(low=-500, high=500, size=size)
#             b = np.random.uniform(low=0.001, high=500, size=random_drop_size)
#             _a = ad.tensor(ad.array(a.tolist()))
#             _b = ad.tensor(ad.array(b.tolist()))
#             # Test for unary ops needs class specific functions
#             self.assertEqual(np.abs(a).tolist(), _a.abs().arr.value)
#             self.assertEqual(np.negative(a).tolist(), _a.neg().arr.value)
#             self.assertEqual(np.exp(a).tolist(), _a.exp().arr.value)
#             self.assertEqual(np.log(b).tolist(), _b.log().arr.value)
#             self.assertEqual(np.sin(a).tolist(), _a.sin().arr.value)
#             self.assertEqual(np.cos(a).tolist(), _a.cos().arr.value)
#             # Binary ops
#             self.assertTrue(all(op(a, b).tolist() == op(_a, _b).arr.value for op in OPS['binary']))
#             # Matmul
#             [n, k, m] = np.random.randint(low=1, high=5, size=3)
#             mat1 = ad.tensor(np.random.rand(*size[0:4], n, k).tolist())
#             mat2 = ad.tensor(np.random.rand(*size[0:4], k, m).tolist())
#             np_matmul = np.array(mat1.arr.value, dtype=np.float64) @ np.array(mat2.arr.value, dtype=np.float64)
#             ad_matmul = mat1 @ mat2
#             self.assertEqual(np.array(np_matmul, dtype=np.float32).tolist(),
#                              np.array(ad_matmul.arr.value, dtype=np.float32).tolist())
#
#     def test_jvp_unary_binary(self):
#         input_dim = np.random.randint(low=1, high=20)
#         function_depth = np.random.randint(low=1, high=500)
#         x = np.random.uniform(low=-2, high=2, size=(function_depth, input_dim)).astype(np.float64)
#         v = np.random.uniform(low=-2, high=2, size=(function_depth, input_dim))
#
#         # create a random function for jax
#         # and build the tensor graph in ys at the same time
#         func_seq = []
#         ys = [ad.tensor(i.tolist()) for i in x]
#         root_tensors = [t for t in ys]
#
#         def apply_unary(op):
#             rand_index = np.random.choice(function_depth)
#             if op == jnp.log:
#                 ys[rand_index] = ys[rand_index].abs().log()
#                 func_seq.append((rand_index, jnp.abs, (rand_index, )))
#                 func_seq.append((rand_index, jnp.log, (rand_index, )))
#             else:
#                 ys[rand_index] = jnp_to_tensor_map[op](ys[rand_index])
#                 func_seq.append((rand_index, op, (rand_index, )))
#
#         def apply_binary(op):
#             rand_index = np.random.choice(function_depth)
#             rand_root_first = np.random.rand() > 0.5
#             if rand_root_first:
#                 ys[0] = jnp_to_tensor_map[op](ys[rand_index], ys[0])
#                 func_seq.append((0, op, (rand_index, 0)))
#             else:
#                 ys[0] = jnp_to_tensor_map[op](ys[0], ys[rand_index])
#                 func_seq.append((0, op, (0, rand_index)))
#
#         for _ in range(function_depth):
#             func = np.random.choice([apply_unary, apply_binary])
#             apply_unary(np.random.choice(jnp_unary)) if func == apply_unary \
#                 else apply_binary(np.random.choice(jnp_binary))
#
#         def random_function(inputs):
#             outputs = [i for i in inputs]
#             for target_index, op, operand_indexes in func_seq:
#                 args = [outputs[operand] for operand in operand_indexes]
#                 outputs[target_index] = op(*args)
#             return outputs
#
#         f = random_function
#         f(x)
#         (jax_outputs, jax_tangents) = np.array(jax.jvp(f, (x, ), (v, )))
#         tensor_outputs = [y.arr.value for y in ys]
#         # I need to handle float64 - float32 problems, so I just feed everything to np and cast to float32
#         self.assertEqual(np.array(jax_outputs, dtype=np.float32).tolist(), np.array(tensor_outputs, dtype=np.float32).tolist())
#
#         directions = {root: seed.tolist() for root, seed in zip(root_tensors, v)}
#         tensor_tangents = [ad.jvp(y, None, directions).value for y in ys]
#         self.assertEqual(np.array(jax_tangents, dtype=np.float32).tolist(), np.array(tensor_tangents, dtype=np.float32).tolist())
#
#     def test_jvp_matmul(self):
#         for _ in range(ITER):
#             layers = np.random.randint(low=2, high=10)
#             layer_dims = np.exp2(np.random.randint(low=5, high=10, size=layers)).astype(np.int32)
#             x = [np.random.uniform(low=-10, high=10, size=(layer_dims[i], layer_dims[i+1])).astype(np.float64) for i in range(layers-1)]
#             v = [np.random.randint(low=-1, high=1, size=(layer_dims[i], layer_dims[i+1])).astype(np.float64) for i in range(layers-1)]
#
#             def random_function(inputs):
#                 res = [inputs[0]]
#                 for mat in inputs[1:]:
#                     res.append(res[-1] @ mat)
#                 return res
#
#             f = random_function
#             (jax_outputs, jax_tangents) = jax.jvp(f, (x,), (v,))
#
#             # test for f(x)
#             root_tensors = [ad.tensor(x_i.tolist()) for x_i in x]
#             ys = [root_tensors[0]]
#             for y in ys[1:]: ys.append(ys[-1] @ y)
#             tensor_outputs = [y.arr.value for y in ys]
#             for jax_i, tensor_i in zip(jax_outputs, tensor_outputs):
#                 self.assertEqual(np.array(jax_i, dtype=np.float32).tolist(),
#                                  np.array(tensor_i, dtype=np.float32).tolist())
#
#             # test for f'(x)
#             directions = {root: seed.tolist() for root, seed in zip(root_tensors, v)}
#             tensor_tangents = [ad.jvp(y, None, directions).value for y in ys]
#             for jax_i, tensor_i in zip(jax_tangents, tensor_tangents):
#                 self.assertEqual(np.array(jax_i, dtype=np.float32).tolist(),
#                                  np.array(tensor_i, dtype=np.float32).tolist())
#
#     def test_vjp_matmul(self):
#         for _ in range(ITER):
#             size = np.random.randint(low=1, high=5, size=4)
#             m1 = np.random.uniform(low=-1, high=1, size=(*size, 2, 3))
#             m2 = np.random.uniform(low=-1, high=1, size=(*size, 3, 2))
#             m3 = np.random.uniform(low=-1, high=1, size=(1, 2))
#
#             def f(x1, x2, x3):
#                 return (x1 @ x2) * ((jnp.exp(x1) + x1) @ (x3 - jnp.log(jnp.abs(x2)) / x2))
#
#             # jax grad
#             jax_outputs, jax_fvjp = jax.vjp(f, *(m1, m2, m3))
#             cotangents = np.random.uniform(-1, 1, size=jax_outputs.shape)
#             jax_grad = jax_fvjp(cotangents)
#             # ad grad
#             t1 = ad.tensor(m1.tolist())
#             t2 = ad.tensor(m2.tolist())
#             t3 = ad.tensor(m3.tolist())
#             root_tensors = [t1, t2, t3]
#             y = (t1 @ t2) * ((t1.exp() + t1) @ (t3 - t2.abs().log() / t2))
#             ad_grad = ad.vjp(y, None, ad.array(cotangents.tolist()))
#
#             for i, t in enumerate(root_tensors):
#                 self.assertTrue(np.allclose(jax_grad[i], np.array(ad_grad[t].value)))
#
#     def test_vjp_unary_binary(self):
#         for _ in range(ITER):
#             input_dim = np.random.randint(low=1, high=100)
#             function_depth = np.random.randint(low=1, high=20)
#             x = np.random.uniform(low=-16, high=16, size=(function_depth, input_dim)).astype(np.float64)
#
#             # create a random function for jax
#             # and build the tensor graph in ys at the same time
#             func_seq = []
#             ys = [ad.tensor(i.tolist()) for i in x]
#             root_tensors = [t for t in ys]
#
#             def apply_unary(op):
#                 rand_index = np.random.choice(function_depth)
#                 if op == jnp.log:
#                     ys[rand_index] = ys[rand_index].abs().log()
#                     func_seq.append((rand_index, jnp.abs, (rand_index, )))
#                     func_seq.append((rand_index, jnp.log, (rand_index, )))
#                 else:
#                     ys[rand_index] = jnp_to_tensor_map[op](ys[rand_index])
#                     func_seq.append((rand_index, op, (rand_index, )))
#
#             def apply_binary(op):
#                 rand_index = np.random.choice(function_depth)
#                 rand_root_first = np.random.rand() > 0.5
#                 if rand_root_first:
#                     ys[0] = jnp_to_tensor_map[op](ys[rand_index], ys[0])
#                     func_seq.append((0, op, (rand_index, 0)))
#                 else:
#                     ys[0] = jnp_to_tensor_map[op](ys[0], ys[rand_index])
#                     func_seq.append((0, op, (0, rand_index)))
#
#             for _ in range(function_depth):
#                 func = np.random.choice([apply_unary, apply_binary])
#                 apply_unary(np.random.choice(jnp_unary)) if func == apply_unary \
#                     else apply_binary(np.random.choice(jnp_binary))
#
#             def random_function(inputs):
#                 outputs = [i for i in inputs]
#                 for target_index, op, operand_indexes in func_seq:
#                     args = [outputs[operand] for operand in operand_indexes]
#                     outputs[target_index] = op(*args)
#                 return outputs[0]
#
#             # test f(x)
#             f = random_function
#             (jax_outputs, jax_fvjp) = jax.vjp(f, x)
#             tensor_outputs = [y.arr.value for y in ys][0]
#             self.assertTrue(jnp.allclose(jax_outputs, np.array(tensor_outputs)))
#
#             # test f'(x)
#             cotangents = np.random.randint(0, 2, size=jax_outputs.shape).astype(np.float64)
#             jax_grad = jax_fvjp(cotangents)
#             ad.vjp(ys[0], None, ad.array(cotangents.tolist()))
#             ad_grad = [t.gradient.value if t.gradient.value != [0] else ad.array(0, outer_shape=t.shape).value
#                        for t in root_tensors]
#             self.assertTrue(jnp.allclose(jax_grad[0], np.array(ad_grad)))
#
#             # These are useful when debugging
#             # print(f(x))
#             # print(jax_outputs)
#             # print('Root tensors')
#             # for y in root_tensors:
#             #     print(y)
#             # print('Output tensors')
#             # for y in ys:
#             #     print(y)
#             # print(f'cotangents: shape = {cotangents.shape}')
#             # print(f'            val = {cotangents}')
#             # print()
#             # print(f'jax_grad  : shape = {jax_grad[0].shape}')
#             # print(f'            val = {jax_grad[0]}')
#             # print()
#             # print(f'ad_grad   : shape = {len(ad_grad)}, {ad.check_shape(ad_grad[0])}')
#             # print(f'            val = {ad_grad}')
#             # print()
#             # ad.all_tensors()


cov.stop()
cov.save()
if __name__ == '__main__':
    unittest.main()
    cov.html_report()
