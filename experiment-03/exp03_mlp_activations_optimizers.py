import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- step 1: load dataset ---
data = load_breast_cancer()
X = data.data
y = tf.keras.utils.to_categorical(data.target, num_classes=2)

# --- step 2: train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- step 3: normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- step 4: build mlp ---
def build_model(activation='relu', optimizer='adam'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation=activation,
                              input_shape=(30,)),
        tf.keras.layers.Dense(8, activation=activation),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- step 5: compare activation functions ---
activations = ['relu', 'tanh', 'sigmoid']
print("comparing activation functions (adam):")
print("-" * 35)
for act in activations:
    model = build_model(activation=act, optimizer='adam')
    model.fit(X_train, y_train, epochs=50, verbose=0,
              validation_split=0.1)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  {act:8s}  test accuracy: {acc:.4f}")

# --- step 6: compare optimizers ---
optimizers = ['adam', 'sgd', 'rmsprop']
print("\ncomparing optimizers (relu activation):")
print("-" * 35)
for opt in optimizers:
    model = build_model(activation='relu', optimizer=opt)
    model.fit(X_train, y_train, epochs=50, verbose=0,
              validation_split=0.1)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  {opt:8s}  test accuracy: {acc:.4f}")
