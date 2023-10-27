import tensorflow as tf
import numpy as np

tensor_index_1d = tf.constant([3, 4, 612, 42, 0, 2])

print("1d:", tensor_index_1d[0:4])
# get first 4 elements -> tf.Tensor([  3   4 612  42], shape=(4,), dtype=int32)
print("1d:", tensor_index_1d[2:4 + 1])
# get elements from index 2 to 4 and taht foruth , with no +1 get numbers from index starting untill last index specefied


tensor_index_2d = tf.constant([
    [1, 2, 0],
    [3, 5, -1],
    [1, 5, 6],
    [2, 3, 8]
])
print("2d: ", tensor_index_2d[0:3+1, 0:+1])  # [row , columns]

tensor_index_3d = tf.constant([[
    [1, 2, 0],
    [3, 5, -1],
    [1, 5, 6],
    [2, 3, 8]
], [
    [1, 2, 0],
    [3, 5, -1],
    [1, 5, 6],
    [2, 3, 8]
],
    [
    [1, 2, 0],
    [3, 5, -1],
    [1, 5, 6],
    [2, 3, 8]
], [
    [1, 2, 0],
    [3, 5, -1],
    [1, 5, 6],
    [2, 3, 8]
]])
print("3d: ", tensor_index_3d[0:1, 0:1+1, 0:2+1])  # [skup , row , columns]
