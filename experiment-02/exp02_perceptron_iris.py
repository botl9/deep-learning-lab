import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- step 1: load the iris dataset ---
data = load_iris()
X = data.data       # 150 samples, 4 features each
y = data.target     # 0 = setosa, 1 = versicolor, 2 = virginica

# make it binary: 1 if setosa, 0 otherwise
y = (y == 0).astype(int)

# --- step 2: split into train and test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- step 3: normalize features (zero mean, unit variance) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- step 4: convert to tensorflow tensors ---
X_train = tf.constant(X_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
y_train = tf.reshape(y_train, [-1, 1])
X_test = tf.constant(X_test, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)
y_test = tf.reshape(y_test, [-1, 1])

# --- step 5: initialize one weight per feature + one bias ---
w = tf.Variable(tf.random.normal([4, 1], stddev=0.1))
b = tf.Variable(tf.zeros([1]))

# --- step 6: define the perceptron model ---
def perceptron(x):
    # y = sigmoid(x1*w1 + x2*w2 + x3*w3 + x4*w4 + b)
    z = tf.matmul(x, w) + b
    return tf.sigmoid(z)

# --- step 7: training loop ---
optimizer = tf.optimizers.SGD(learning_rate=0.1)

for epoch in range(50):
    with tf.GradientTape() as tape:
        y_pred = perceptron(X_train)
        # binary cross-entropy loss
        loss = -tf.reduce_mean(
            y_train * tf.math.log(y_pred + 1e-7) +
            (1 - y_train) * tf.math.log(1 - y_pred + 1e-7)
        )
    grads = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(grads, [w, b]))

    if epoch % 10 == 0:
        predicted = tf.cast(y_pred > 0.5, tf.float32)
        acc = tf.reduce_mean(tf.cast(predicted == y_train, tf.float32))
        print(f"epoch {epoch:3d}  loss {loss.numpy():.4f}  acc {acc.numpy():.4f}")

# --- step 8: evaluate on test data ---
y_test_pred = tf.cast(perceptron(X_test) > 0.5, tf.float32)
test_acc = tf.reduce_mean(tf.cast(y_test_pred == y_test, tf.float32))
print(f"\ntest accuracy: {test_acc.numpy():.4f}")

# --- step 9: show sample predictions ---
print("\nsample predictions (first 5 test samples):")
for i in range(5):
    pred = int(y_test_pred[i].numpy())
    actual = int(y_test[i].numpy())
    print(f"  sample {i+1}: predicted {pred}, actual {actual}  {'✓' if pred == actual else '✗'}")
