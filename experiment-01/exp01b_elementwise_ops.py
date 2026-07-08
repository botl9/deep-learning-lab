import tensorflow as tf

a = tf.constant([[1, 2],
                 [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6],
                 [7, 8]], dtype=tf.float32)

print("Element-wise Operations")
print("a + b :\n", tf.add(a, b).numpy())
print("a - b :\n", tf.subtract(a, b).numpy())
print("a * b :\n", tf.multiply(a, b).numpy())
print("a / b :\n", tf.divide(a, b).numpy())
