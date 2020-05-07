import numpy as np

def softmax(x, keep_dim=True):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=keep_dim)


def crossEntropy(predict, one_hot):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    correct = predict[one_hot.astype(np.bool)]
    error = -correct + np.log(np.sum(np.exp(predict), axis=1))
    return error


def grad_crossEntropy(predict, one_hot):

    # Compute crossentropy gradient from predict[batch,n_classes] and ids of correct answers
    batch_size = predict.shape[0]
    # to prevent from underflow of numpy, we change data type to float64
    return (-one_hot + softmax(predict)) / batch_size

