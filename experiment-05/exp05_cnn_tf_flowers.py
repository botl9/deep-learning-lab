import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split

# --- step 1: load tf_flowers dataset ---
IMG_SIZE = 128
NUM_CLASSES = 5
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

dataset = tfds.load('tf_flowers', as_supervised=True)
train_data = dataset['train']

# --- step 2: resize, normalize, convert to numpy ---
X, y = [], []
for image, label in train_data:
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    X.append(image.numpy())
    y.append(label.numpy())
X = np.array(X)
y = np.array(y)

# one-hot encode labels (5 classes)
y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

# --- step 3: train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- step 4: build cnn ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- step 5: train ---
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- step 6: evaluate ---
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\ntest accuracy: {test_acc:.4f}")

# --- step 7: predict on a new image ---
preds = model.predict(X_test[:1])
pred_class = CLASS_NAMES[np.argmax(preds[0])]
actual_class = CLASS_NAMES[np.argmax(y_test[0])]
print(f"predicted: {pred_class:12s}")
print(f"actual:    {actual_class:12s}")
