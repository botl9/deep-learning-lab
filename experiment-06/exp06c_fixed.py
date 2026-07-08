import tensorflow as tf

IMG_SIZE, NUM_CLASSES = 28, 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
x_test  = x_test .reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")
print("=" * 60)

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    data_aug,
    tf.keras.layers.Conv2D(128, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(512, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
], name='Fixed')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\n{'=' * 60}")
print("  Fixed model (BatchNorm + Dropout + augmentation)")
print(f"{'=' * 60}")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), verbose=1)

_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"test accuracy: {acc:.4f}")
