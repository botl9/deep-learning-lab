import tensorflow as tf

a = tf.constant([[1, 2],
                 [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6],
                 [7, 8]], dtype=tf.float32)

print("Matrix Multiplication")
print("a @ b :\n", tf.matmul(a, b).numpy())
