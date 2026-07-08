import tensorflow as tf

vector   = tf.constant([1, 2, 3, 4, 5])
matrix   = tf.constant([[1, 2, 3],
                        [4, 5, 6]])

print("Reshape and Transpose")
reshaped   = tf.reshape(vector, [5, 1])
transposed = tf.transpose(matrix)
print("Reshaped (5,1) :\n", reshaped.numpy())
print("Transposed    :\n", transposed.numpy())
