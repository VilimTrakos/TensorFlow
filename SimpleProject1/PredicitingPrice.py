import tensorflow as tf  # models
import pandas as pd  # reading and processing data
import seaborn as sns  # visualization
import numpy as np  # math computations
import matplotlib.pyplot as plt  # plotting bar chart
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
