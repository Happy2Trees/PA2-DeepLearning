import numpy as np

class baseLayer():
    def __init__(self):
        pass

    def __call__(self, input):
        output =  self.forward(input)
        return output

    def forward(self, input):
        '''warping part'''
        return input

    def backward(self, input, output_grad):
        pass

class leakyReLU(baseLayer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.has_param = False
    def forward(self, x):
        return np.where(x > 0, x, x * self.alpha)

    def backward(self, input, grad_output):
        grad = np.where(input > 0, 1, self.alpha)
        return grad * grad_output


class ReLU(baseLayer):
    def __init__(self):
        super().__init__()
        self.has_param = False

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, input, grad_output):
        grad = input > 0
        return grad * grad_output


class Conv2d(baseLayer):
    def __init__(self, input_features, output_features, kernel_size):
        super().__init__()
        self.weight = np.zeros(output_features, kernel_size, kernel_size)
        self.bias = np.zeros(kernel_size, kernel_size)

    def forward(self):
        pass




class linear(baseLayer):

    def __init__(self, input, output, learning_rate=0.1):
        super().__init__()
        self.has_param = True
        self.learning_rate = learning_rate

        # xaviar initialize
        self.weight = np.random.normal(loc=0.0,
                                        scale = np.sqrt(2/(input+output)),
                                        size = (input,output))
        self.bias = np.zeros(output)
    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def backward(self, input, grad_output):

        # for the previous layer, gradient update
        grad_input = np.dot(grad_output, self.weight.T)

        # we sum all of batches
        grad_weights = np.dot(input.T, grad_output)
        grad_bias = np.mean(grad_output, axis=0) * input.shape[0]

        # stochastic gradient step
        self.weight = self.weight - self.learning_rate * grad_weights
        self.bias = self.bias - self.learning_rate * grad_bias

        return grad_input