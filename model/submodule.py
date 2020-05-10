import numpy as np
from pathlib import Path
class baseLayer():
    def __init__(self):
        pass

    def __call__(self, input):
        output = self.forward(input)
        return output

    def forward(self, input):
        '''warping part'''
        pass

    def backward(self, input, output_grad):
        pass

class leakyReLU(baseLayer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.has_param = False

    def forward(self, x):
        return np.where(x > 0, x, x * self.alpha)

    # gradient will be 1 or alpha
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

# reference CS231n class of stanford university
# reference : https://cs231n.github.io/convolutional-networks
# reference : https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
class Conv2d(baseLayer):
    def __init__(self, input_features, output_features, kernel_size, padding=1, stride=1, learning_rate=0.011):
        super().__init__()
        self.has_param = True
        self.learning_rate = learning_rate
        self.input_features = input_features
        self.output_features = output_features
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]

        # xaviar uniform Initialize to converge fast and prevent from gradient exploding or vanishing
        # xaviar weight initialization paper : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self.weight = np.random.uniform(low=-6 / np.sqrt(input_features * self.kernel_height * self.kernel_width + output_features * self.kernel_width * self.kernel_height),
                                        high=6 / np.sqrt(input_features * self.kernel_height * self.kernel_width + output_features * self.kernel_width * self.kernel_height),
                                        size=(output_features, input_features, self.kernel_height, self.kernel_width))

        # initialize with uniform distribution
        self.bias = np.random.uniform(low=-1 / np.sqrt(input_features * self.kernel_height * self.kernel_width),
                                        high=1 / np.sqrt(input_features * self.kernel_height * self.kernel_width),
                                        size=(output_features, 1))

    def forward(self, x):
        batch, _, input_height, input_width = x.shape
        # calculate output size because we have to calculate matrix form
        # to use
        self.out_height = (input_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        self.out_width = (input_width + 2 * self.padding - self.kernel_width) // self.stride + 1
        # to calculate matrix form , using image to column function
        # weight_col ==> (output_features, input_features * kernel_height * kernel_width)
        weight_col = self.weight.reshape(self.output_features, -1)
        input_col = self.img_to_col(x)

        # forward
        out = weight_col @ input_col + self.bias

        # reshape the output to original form
        out = out.reshape(self.output_features, self.out_height, self.out_width, batch)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, input, output_grad):
        input_col = self.img_to_col(input)
        # excep output_features summing other axis
        grad_bias = np.sum(output_grad, axis=(0, 2, 3))
        grad_bias = grad_bias.reshape(self.output_features, -1)

        # output_grad reshaped ==> (output_Features,  output_height * output_width * batch)
        output_grad_reshaped = output_grad.transpose(1, 2, 3, 0).reshape(self.output_features, -1)
        grad_weight_reshaped = output_grad_reshaped @ input_col.T
        grad_weight = grad_weight_reshaped.reshape(self.weight.shape)

        # weight_reshaped --> (output_features, input_features * kernel_height * kernel_width)
        weight_reshaped = self.weight.reshape(self.output_features, -1)
        # grad_input_col ==> (input_Features * kernel_height * kernel_width, output_height * output_width * batch)
        grad_input_col = weight_reshaped.T @ output_grad_reshaped
        # multiply two matrix and return to original input matrix form
        grad_input = self.col_to_img(grad_input_col, input.shape)

        # gradien descent
        self.weight = self.weight - self.learning_rate * grad_weight
        self.bias = self.bias - self.learning_rate * grad_bias

        return grad_input

    def img_to_col(self, x):
        zero_x = np.array(x)
        if self.padding > 0 :
            # x shapels consist of 4 channels
            zero_x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
        k, i, j = self.img_to_col_indices(x.shape)

        cols = zero_x[:, k, i, j]
        # change to (kernel_height * kernel_width * channel, output_height * output_width * batch )
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_height * self.kernel_width * x.shape[1], -1)
        return cols

    def img_to_col_indices(self, x_shape):
        batch, input_features, input_height, input_width = x_shape

        # i_column correspond to the kernel index
        i_column = np.repeat(np.arange(self.kernel_height), self.kernel_width)
        i_column = np.tile(i_column, self.input_features)
        # i_row correspond to the output index
        i_row = self.stride * np.repeat(np.arange(self.out_height), self.out_width)

        # j_column correspond to the kernel index
        j_column = np.tile(np.arange(self.kernel_width), self.kernel_height * input_features)
        j_row = self.stride * np.tile(np.arange(self.out_width), self.out_height)

        # j_row correspond to the output index
        i = i_column.reshape(-1, 1) + i_row.reshape(1, -1)
        j = j_column.reshape(-1, 1) + j_row.reshape(1, -1)

        # This takes into account input features. This is because we have to multiply it as much as input_features.
        k = np.repeat(np.arange(self.input_features), self.kernel_height * self.kernel_width).reshape(-1, 1)

        return k, i, j

    def col_to_img(self, column, x_shape):
        batch, input_features, input_height, input_width = x_shape
        input_height_padded, input_width_padded = input_height + 2 * self.padding, input_width + 2 * self.padding
        x_padded = np.zeros((batch, input_features, input_width_padded, input_height_padded))
        k, i, j = self.img_to_col_indices(x_shape)
        column_reshaped = column.reshape(input_features * self.kernel_height * self.kernel_width, -1, batch)
        column_reshaped = column_reshaped.transpose(2, 0, 1)
        
        # padded array with column indicies k, i, j np.add.at add not duplicately
        np.add.at(x_padded, (slice(None), k, i, j), column_reshaped)

        # padded array to original array
        if self.padding is not 0:
            return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_padded

# reference : https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
class MaxPool2d(baseLayer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.has_param = False
        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]
        if stride is not None:
            self.stride = stride
        else:
            self.stride = kernel_size
        self.padding = padding


    def forward(self, x):
        batch, input_features, input_height, input_width = x.shape
        # To prevent the channel from sticking in the column form
        x_reshaped = x.reshape(batch * input_features, 1, input_height, input_width)

        # calculate output size
        self.out_height = int((input_height + 2 * self.padding - self.kernel_height) / self.stride + 1)
        self.out_width = int((input_width + 2 * self.padding - self.kernel_width) / self.stride + 1)


        # output will be (kernel_height * kernel_width, output_height * output_width * batch * input_features )
        x_reshaped_cols = self.img_to_col(x_reshaped)
        # we have to save this form to get backpropagation
        self.shape_x_cols = x_reshaped_cols.shape


        # to find max value of each kernel(column)
        # we have to keep max Index for backpropagation
        self.max_index = np.argmax(x_reshaped_cols, axis=0)

        # Finally, we get all the max value at each column
        out_reshaped = x_reshaped_cols[self.max_index, range(self.max_index.size)]

        # reshaped to each form
        # out will be output_height * output_width * batch * input_features
        out = out_reshaped.reshape(self.out_height, self.out_width, batch, input_features)

        # Transpose to original form
        out = out.transpose(2, 3, 0, 1)
        return out

    def backward(self, input, grad_output):
        batch, input_features, input_height, input_width = input.shape
        # max pooling backward has role which upsamples grad_outputs so

        # (kernel_height * kernel_width, output_height * output_width * batch * input_features )
        # initialize input gradient
        ones_col = np.zeros(self.shape_x_cols)

        # grad output ==> (batch, output_features, output_height, output_width), then flattened
        # grad output_flattened ==> output_height * output_width * batch * input_features
        # caution output features == input features in max pooling
        grad_output_flattened = grad_output.transpose(2, 3, 0, 1).ravel()


        # ones_col ==> (kernel_height * kernel_width, output_height * output_width * batch * input_features )
        ones_col[self.max_index, range(self.max_index.size)] = grad_output_flattened

        # change to original upsampled Image
        reshaped_size = (batch * input_features, 1, input_height, input_width)
        grad_input_reshaped = self.col_to_img(ones_col, reshaped_size)

        # reshape to the input image shape
        grad_input = grad_input_reshaped.reshape(input.shape)

        return grad_input

    def img_to_col(self, x):
        zero_x = x
        if self.padding > 0:
            # x shapels consist of 4 channels
            zero_x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
        k, i, j = self.img_to_col_indices(x.shape)

        cols = zero_x[:, k, i, j]
        # change to (kernel_height * kernel_width * channel, output_height * output_width * batch )
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_height * self.kernel_width * x.shape[1], -1)
        return cols

    def img_to_col_indices(self, x_shape):
        batch, input_features, input_height, input_width = x_shape

        # i_column correspond to the kernel index
        i_column = np.repeat(np.arange(self.kernel_height), self.kernel_width)
        i_column = np.tile(i_column, input_features)
        # i_row correspond to the output index
        i_row = self.stride * np.repeat(np.arange(self.out_height), self.out_width)

        # j_column correspond to the kernel index
        j_column = np.tile(np.arange(self.kernel_width), self.kernel_height * input_features)
        j_row = self.stride * np.tile(np.arange(self.out_width), self.out_height)

        # j_row correspond to the output index numpy magic function
        i = i_column.reshape(-1, 1) + i_row.reshape(1, -1)
        j = j_column.reshape(-1, 1) + j_row.reshape(1, -1)

        # This takes into account input features. This is because we have to multiply it as much as input_features.
        k = np.repeat(np.arange(input_features), self.kernel_height * self.kernel_width).reshape(-1, 1)

        return k, i, j

    def col_to_img(self, column, x_shape):
        batch, input_features, input_height, input_width = x_shape
        input_height_padded, input_width_padded = input_height + 2 * self.padding, input_width + 2 * self.padding
        x_padded = np.zeros((batch, input_features, input_width_padded, input_height_padded))
        k, i, j = self.img_to_col_indices(x_shape)
        column_reshaped = column.reshape(input_features * self.kernel_height * self.kernel_width, -1, batch)
        column_reshaped = column_reshaped.transpose(2, 0, 1)

        # padded array with column indicies k, i, j np.add.at add not duplicately
        np.add.at(x_padded, (slice(None), k, i, j), column_reshaped)

        # padded array to original array
        if self.padding is not 0:
             return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_padded

class linear(baseLayer):

    def __init__(self, input, output, learning_rate=0.012):
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

        # gradient descent step
        self.weight = self.weight - self.learning_rate * grad_weights
        self.bias = self.bias - self.learning_rate * grad_bias

        return grad_input

class flatten(baseLayer):
    def __init__(self):
        super().__init__()
        self.has_param = False

    def forward(self, input):
        batch, input_features, input_height, input_width = input.shape
        # reshape
        output = input.reshape(batch, -1)
        return output

    def backward(self, input, output_grad):
        batch, input_features, input_height, input_width = input.shape
        # output grad shape ==> (batch, input_features * input_height * input_width)
        grad_input = output_grad.reshape(batch, input_features, input_height, input_width)
        return grad_input
