import random
import AutoDiff as ad


class MLP:
    def __init__(self, dims: list | tuple, activation):
        if any(u < 1 for u in dims): raise ValueError('Must have >= 1 unit in each layer')
        if not callable(activation): raise ValueError(f'Expects activation to be callable, got {activation}.')

        self.dims = dims
        self.activation = activation
        # A handy tensor constructor  will be good
        self.weights = [ad.tensor(ad.array(random.uniform(-1.0, 1.0), (din, dout))) for din, dout in zip(dims, dims[1:])]
        self.bias = [ad.tensor(ad.array(random.uniform(-1.0, 1.0))) for _ in dims[1:]]

    def gradient(self):
        pass

    def zero_grad(self):
        pass

    def update(self):
        pass

    def predict(self, x: (int, float, list)):
        return


mlp = MLP([2, 3, 3], 'relu')
print(mlp.shape)
print(mlp.weights)
print(mlp.weights[0])
print(mlp.weights[1])
