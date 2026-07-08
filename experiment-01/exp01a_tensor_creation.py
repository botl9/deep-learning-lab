import tensorflow as tf
import numpy as np

scalar   = tf.constant(42)
vector   = tf.constant([1, 2, 3, 4, 5])
matrix   = tf.constant([[1, 2, 3],
                        [4, 5, 6]])
tensor3d = tf.constant(np.arange(24).reshape(2, 3, 4))

print("Tensor Creation")
print("Scalar  (0-D)  :", scalar.numpy())
print("Vector  (1-D)  :", vector.numpy())
print("Matrix  (2-D)  :\n", matrix.numpy())
print("3-D Tensor shape :", tensor3d.shape)
