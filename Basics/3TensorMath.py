import tensorflow as tf
import numpy as np


########### ABS ###########
ab_tensor = tf.constant([-1.3, -1, -10, 10])
print("Abs: ", tf.abs(ab_tensor))

################## complex #############
comp_tensor = tf.constant([-2.35 + 3.533j])
print("Complex abs: ", tf.abs(comp_tensor))
# PRINT:Complex abs:  tf.Tensor([4.24318147], shape=(1,), dtype=float64)

################### sqrt ################
square_tensor = tf.constant([3, 1, 6, 2, 5])
squared_values = tf.square(square_tensor)
print("Square tensor values: ", squared_values)


################## low level math operations #########################
tenosr_one = tf.constant([3, 1, 6, 2, 5, 2])
tenosr_two = tf.constant([1, 2, 3, 4, 5, 0])

print(" Addition: ", tenosr_one + tenosr_two)
print(" Subtract: ", tf.subtract(tenosr_one, tenosr_two))
print(" Multiply: ", tf.multiply(tenosr_one, tenosr_two))
print(" Divide: ", tf.divide(tenosr_one, tenosr_two))
print(" No nan Divide: ", tf.math.divide_no_nan(tenosr_one, tenosr_two))
# if we have tensor that are not same shape they can be added, subtracted, multiplyed , but not devided in any way


tenosr_one = tf.constant([3, 1, 6, 5, 2])
tenosr_two_2d = tf.constant([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(" Multiply 2d and 1d: ", tf.multiply(tenosr_one, tenosr_two_2d))
# Multiply 2d and 1d:  tf.Tensor(
# [[ 3  2 18 20 10]
# [ 3  2 18 20 10]], shape=(2, 5), dtype=int32)

########################### ARG MAX , ARG MIN 1d###########################
tensor_arg_max = tf.constant([3, 1, 6, 5, 2])
print("Arg 1d max tensor: ", tf.math.argmax(tensor_arg_max))
print("Arg 1d min tensor: ", tf.math.argmin(tensor_arg_max))
########################### ARG MAX , ARG MIN 2d###########################
tenosr_2d_arg = tf.constant([[1, 2, 10, 4, 5],
                             [1, 20, 3, 4, 5]])
print("Arg 2d max tensor: ", tf.math.argmax(tenosr_2d_arg, 0))
print("Arg 2d min tensor: ", tf.math.argmin(tenosr_2d_arg, 1))

############################# Equal ###################

tenosr_one_1d = tf.constant([1, 2, 10, 4, 5])
tenosr_two_1d = tf.constant([1, 2, 10, 4, 10])
print("Tensor equal: ", tf.equal(tenosr_one_1d, tenosr_two_1d))
# Tensor equal:  tf.Tensor([ True  True  True  True False], shape=(5,), dtype=bool)

##################### Reduce sum, max-> max value output , min, std... ###################

tenosr_2d_reduce_sum = tf.constant([[1, 2, 10, 4, 5],
                                    [1, 20, 3, 4, 5]])
print("reduce sum: ", tf.math.reduce_sum(
    tenosr_2d_reduce_sum, axis=1))  # axes =0 | , axes  = 1 ->

###################### top k ####################
tenosr_topK = tf.constant([1, 2, 10, 4, 5])
print("Top k: ", tf.math.top_k(tenosr_topK))
