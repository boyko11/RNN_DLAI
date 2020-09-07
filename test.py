import rnn_utils
import numpy as np

softmax_test = np.array([1, 2, 3, 4])

softmax_result = rnn_utils.softmax(softmax_test)

print(softmax_result)


def my_softmax(array):

    array_exp = np.exp(array)
    return array_exp / np.sum(array_exp)


print(my_softmax(softmax_test))