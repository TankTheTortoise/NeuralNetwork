import numpy as np
from Layers import FCLayer, ActivationLayer
from Networks import Network
import activations as ac
import lossFunctions as ls
# Standard scientific Python imports
from matplotlib import pyplot as plt
from PIL import Image

# Import datasets, classifiers and performance metrics
from keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical


