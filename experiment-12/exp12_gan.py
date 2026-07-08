import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
import numpy as np

LATENT_DIM = 100

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)

generator = Sequential([
    Dense(128, activation='relu', input_shape=(LATENT_DIM,)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28, 1))
], name='generator')

discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
], name='discriminator')

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.trainable = False

gan_input = tf.keras.Input(shape=(LATENT_DIM,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
combined = Sequential([generator, discriminator], name='combined')
combined.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss='binary_crossentropy'
)

batch_size = 128
epochs = 20
half_batch = batch_size // 2

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    real_images = x_train[idx]
    real_labels = np.ones((half_batch, 1))

    noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
    fake_images = generator.predict(noise, verbose=0)
    fake_labels = np.zeros((half_batch, 1))

    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
    valid_labels = np.ones((batch_size, 1))
    g_loss = combined.train_on_batch(noise, valid_labels)

    print(f"Epoch {epoch+1:2d}/{epochs}  D loss: {d_loss[0]:.4f}  D acc: {d_loss[1]:.4f}  G loss: {g_loss:.4f}")

noise = np.random.normal(0, 1, (10, LATENT_DIM))
generated = generator.predict(noise, verbose=0)
print("\nGenerated sample pixel means:")
for i in range(10):
    print(f"  Sample {i+1:2d}: mean = {generated[i].mean():.3f}")
