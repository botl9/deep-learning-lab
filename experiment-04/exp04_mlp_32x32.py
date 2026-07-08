import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# --- step 1: load fashion_mnist dataset ---
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

# --- step 2: combine into single dataset ---
X = np.concatenate([x_train_full, x_test_full], axis=0)
y = np.concatenate([y_train_full, y_test_full], axis=0)

# --- step 3: train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- step 4: normalize and flatten ---
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# --- step 5: one-hot encode labels (10 classes) ---
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# --- step 6: build mlp ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- step 7: train ---
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- step 8: evaluate ---
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\ntest accuracy: {test_acc:.4f}")
