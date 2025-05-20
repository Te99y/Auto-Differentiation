import random


class MLP:
    def __init__(self, shape, activation):
        if not isinstance(shape, list):
            raise TypeError('Shape must be either list or tuple')
        if len(shape) < 1:
            raise ValueError('Must have at least 1 layer')
        if any(u < 1 for u in shape):
            raise ValueError('Must have at least 1 unit in each layer')
        self.shape = shape
        self.activation = activation

        self.weights = []  # len = layers-1, input layer needs no weight
        self.bias = []
        self.weights_grad = []
        self.bias_grad = []
        for d_in, d_out in zip(shape[:-1], shape[1:]):
            # we use inner product : prev_activation[1 x d_in] * weight_of_a_neuron[d_in x 1]
            # since the weight of a neuron is a column vector[d_in x 1]
            # and we have d_out neurons in a layer
            # so the weight of a layer is a matrix[d_in x d_out]
            self.weights.append([[random.gauss(mu=0.0, sigma=1.0) for _ in range(d_out)] for _ in range(d_in)])
            self.bias.append([0.0]*d_out)

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
