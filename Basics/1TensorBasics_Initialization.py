# pip install --upgrade pip
# For GPU users
# pip install tensorflow[and-cuda]
# For CPU users
# pip install tensorflow

import tensorflow as tf
import numpy as np
# 0d tensor
tensor_zero_d = tf.constant(4)
print("0D tensor:", tensor_zero_d)

# 1D bool tensor
tensor_bool = tf.constant([True, True, False])
print(tensor_bool)

# 1D string tensor
tensor_string = tf.constant(["hello world", "hi "])
print(tensor_string)

# 1D float tensor casted to bool tensor
tensor_one_d = tf.constant([2, 0., -3, 8, 90], dtype=tf.float32)
casted_tensor_one_d = tf.cast(tensor_one_d, dtype=tf.bool)
#print("1d: ", tensor_one_d)
#print("1d:  (bool)", casted_tensor_one_d)

# 2D tensor, int
tensor_two_d = tf.constant([
    [1, 2, 0],
    [3, 5, -1],
    [1, 5, 6],
    [2, 3, 8]
])
#print("2d: ", tensor_two_d)

# 3D tensor, int
tensor_three_d = tf.constant([
    [[1, 2, 0],
     [3, 5, -1]],

    [[10, 2, 0],
     [1, 0, 2]],

    [[5, 8, 0],
     [2, 7, 0]],

    [[2, 1, 9],
     [4, -3, 32]],

])
#print("3d: ", tensor_three_d)
# get integer that represents shape of tensor
print("3d.ndim: ", tensor_three_d.ndim)
print("3d.shape: ", tensor_three_d.shape)  # get shape


tensor_four_d = tf.constant([[
    [[1, 2, 0],
     [3, 5, -1]],

    [[10, 2, 0],
     [1, 0, 2]],

    [[5, 8, 0],
     [2, 7, 0]],

    [[2, 1, 9],
     [4, -3, 32]],

],
    [
    [[1, 2, 0],
     [3, 5, -1]],

    [[10, 2, 0],
     [1, 0, 2]],

    [[5, 8, 0],
     [2, 7, 0]],

    [[2, 1, 9],
     [4, -3, 32]],

],
    [
    [[1, 2, 0],
     [3, 5, -1]],

    [[10, 2, 0],
     [1, 0, 2]],

    [[5, 8, 0],
     [2, 7, 0]],

    [[2, 1, 9],
     [4, -3, 32]],

],
    [
    [[1, 2, 0],
     [3, 5, -1]],

    [[10, 2, 0],
     [1, 0, 2]],

    [[5, 8, 0],
     [2, 7, 0]],

    [[2, 1, 9],
     [4, -3, 32]],

]], dtype=tf.float16
)
print("4D.ndim: ", tensor_four_d.ndim)


#############################################

np_array = np.array([1, 2, 5])
print("NP array: ", np_array)

convert_np_to_tensor = tf.convert_to_tensor(np_array)
print("Converted np array:", convert_np_to_tensor)

eye_tensor = tf.eye(
    num_rows=3,
    num_columns=5,
    batch_shape=[1],
    dtype=tf.dtypes.float32,
    name=None
)
print("eye tensor: ", eye_tensor)
# eye tensor:  tf.Tensor(
# [[1. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0.]
# [0. 0. 1. 0. 0.]] x batch_shape dodaje jos toliko koliko je napisano, shape=(3, 5), dtype=float32)

#################################################
# Fill

fill_tensor = tf.fill([2, 3], 8, name="test")
print("Fill tensor: ", fill_tensor)
# Fill tensor:  tf.Tensor(
# [[8 8 8]
# [8 8 8]], shape=(2, 3), dtype=int32)

#################################################
# ones
#ones_tensor = tf.ones([2, 2, 1, 4, 5], dtype=tf.dtypes.float16, name=None)
#print("ones tensor: ", ones_tensor)

################################################
# random
random_tensor = tf.random.normal(  # normal moze biti .uniform  -> podjednako, normal je bell shape picking
    [3, 2],
    mean=10.0,  # oko cega se vrte vrijednosti
    stddev=5.0,  # odstupanje od mean
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print("\nRandom: \n", random_tensor)
###################################################
# range size
range_tensor = tf.range(start=2, limit=10, delta=5)
# krene na dva i za 5 do 10 doda sve vrijednosti u 1d tensor
print("Range: ", range_tensor)
print("Range size: ", tf.size(range_tensor))
