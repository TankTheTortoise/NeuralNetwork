import numpy as np


def relu(num):
    for i in np.nditer(num, op_flags=["readwrite"]):
        if i[...] < 0:
            i[...] = 0

    return num


def relu_d(num):
    return relu(num)


def sigmoid(num):
    return 1 / (1 + np.power(0 - num, 2))


def sigmoid_d(num):
    number = sigmoid(num)
    return number * (1 - number)
