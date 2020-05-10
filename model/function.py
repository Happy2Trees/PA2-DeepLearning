import numpy as np
from model.submodule import baseLayer

# flatten --> reshape
def flatten(input):
    batch, input_features, input_height, input_width = input.shape
    # reshape
    output = input.reshape(batch, -1)
    return output


def softmax(x, keep_dim=True):
    # to preven overflow
    max = np.max(x, axis=-1)
    max = np.expand_dims(max, axis=-1)
    return np.exp(x-max) / np.sum(np.exp(x-max), axis=-1, keepdims=keep_dim)

# ground truth is hot encode, so we can define softmax with -correct + np.log(---)
def crossEntropy(predict, one_hot):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    correct = predict[one_hot.astype(np.bool)]
    error = -correct + np.log(np.sum(np.exp(predict), axis=1))
    return error


# we can calculate cross entropy grad with math, then softmax(f(x) - p) is grad
def grad_crossEntropy(predict, one_hot):

    # Compute crossentropy gradient from predict[batch,n_classes] and ids of correct answers
    batch_size = predict.shape[0]
    return (-one_hot + softmax(predict)) / batch_size

