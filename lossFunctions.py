import numpy as np


# Neural networks optimize loss so formula is negative
def mse(real, guess):
    return np.mean(np.power(real - guess, 2))


# Neural networks optimize loss so formula is negative
def mse_d(real, guess):
    return 2 * (guess - real)/real.size

