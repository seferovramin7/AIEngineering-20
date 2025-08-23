
-----

## **Creating with AI: An Introduction to Generative Models (GANs and VAEs) with Keras**

This guide provides a step-by-step introduction to two foundational generative models in AI: Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). We will explore the theory behind each and implement them using the Keras API in TensorFlow.

### **1. What are Generative Models?**

In machine learning, models can be broadly categorized as **discriminative** or **generative**.

  * **Discriminative Models:** These models learn to distinguish between different classes of data. Given an input, they predict a label. For example, a model that looks at a picture and outputs "cat" or "dog" is discriminative. It learns the boundary *between* classes.
  * **Generative Models:** These models learn the underlying distribution of the data itself. Instead of just labeling data, they can generate new, synthetic data samples that resemble the original data. For example, a generative model trained on faces could create brand-new, realistic-looking faces of people who do not exist.

We will focus on two popular architectures for this task: **VAEs** and **GANs**.

-----

### **2. Variational Autoencoders (VAEs)**

A VAE is a generative model that learns a compressed representation of the data in a probabilistic manner. It consists of two main parts: an **encoder** and a **decoder**.

  * **Encoder:** This network takes an input (like an image) and compresses it into a low-dimensional representation called the **latent space**. Unlike a standard autoencoder, a VAE's encoder outputs the parameters of a probability distribution (typically the mean, $z\_{\\mu}$, and log variance, $z\_{\\log\\sigma^2}$) for the latent representation.
  * **Decoder:** This network takes a point sampled from the latent space and attempts to reconstruct the original input from it.

By learning a smooth, continuous latent space, we can sample new points from it and use the decoder to generate novel data.

#### **The VAE Loss Function**

The VAE's power comes from its unique loss function, which has two components:

1.  **Reconstruction Loss:** This measures how well the decoder's output matches the original input. It is often calculated using Binary Cross-Entropy or Mean Squared Error.
    $$L_{\text{reconstruction}} = \text{MSE}(\text{input}, \text{output})$$
2.  **Kullback-Leibler (KL) Divergence:** This acts as a regularizer. It forces the latent space distribution learned by the encoder to be close to a standard normal distribution (mean of 0, variance of 1). This ensures the latent space is well-structured and continuous, making it effective for generating new samples.
    $$L_{KL} = -0.5 \sum (1 + z_{\log\sigma^2} - (z_{\mu})^2 - \exp(z_{\log\sigma^2}))$$

The total loss is the sum of these two components.

#### **Step-by-Step Keras Implementation of a VAE**

Let's build a VAE to generate new handwritten digits using the MNIST dataset.

**Step 1: Setup and Data Loading**

First, import the necessary libraries and load the MNIST dataset. We'll flatten the images and normalize them.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess the data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
```

**Step 2: The Encoder**

The encoder maps input images to the parameters of the latent distribution.

```python
latent_dim = 2  # We'll use a 2D latent space for easy visualization

# Define the encoder input
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

# Output layers for mean and log variance
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Instantiate the encoder model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()
```

**Step 3: The Reparameterization Trick**

We need to sample from the latent distribution, but backpropagation cannot flow through a random sampling operation. The **reparameterization trick** solves this by reframing the sampling:
$z = z\_{\\text{mean}} + \\exp(0.5 \\times z\_{\\text{log\_var}}) \\times \\epsilon$, where $\\epsilon$ is a random sample from a standard normal distribution.

```python
class Sampler(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

**Step 4: The Decoder**

The decoder maps points from the latent space back to the original image dimensions.

```python
# Define the decoder input
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

# Instantiate the decoder model
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
```

**Step 5: Define the VAE Model and Loss**

We combine the components and define the custom loss function.

```python
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = Sampler()([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
```

**Step 6: Train the VAE**

Now, we instantiate and train our VAE model.

```python
# Instantiate the VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# Train the model
vae.fit(mnist_digits, epochs=30, batch_size=128)
```

**Step 7: Generate New Digits**

To generate new digits, we sample random points from the latent space and pass them through the decoder.

```python
import matplotlib.pyplot as plt

def plot_latent_space_digits(vae, n=15, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Sample points from the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space_digits(vae)
```

-----

### **3. Generative Adversarial Networks (GANs)**

GANs use a novel, game-theoretic approach to generation. A GAN consists of two neural networks competing against each other: a **Generator** and a **Discriminator**.

  * **Generator ($G$):** Its job is to create fake data. It takes a random noise vector as input and outputs a data sample (e.g., an image) that looks like it came from the real training dataset.
  * **Discriminator ($D$):** Its job is to be a detective. It is a binary classifier that takes a data sample (either real or fake) and tries to determine if it is authentic (`1`) or a forgery (`0`).

#### **The Adversarial Training Process**

1.  The Generator creates a batch of fake images from random noise.
2.  The Discriminator is shown a mix of real images from the training set and the fake images from the Generator.
3.  The Discriminator's weights are updated to get better at telling the real and fake images apart.
4.  The Generator's weights are then updated based on the Discriminator's feedback. The Generator's goal is to produce images that the Discriminator incorrectly classifies as real.

This process is a **zero-sum game**. As the Discriminator gets better, it provides a stronger signal for the Generator to improve. Over time, the Generator learns to produce highly realistic, sharp images.

#### **Step-by-Step Keras Implementation of a GAN**

Let's build a simple GAN, known as a Deep Convolutional GAN (DCGAN), to generate MNIST digits.

**Step 1: Setup and Data Loading**

We use the same MNIST data but will normalize it to the range `[-1, 1]`, which is standard for GANs using the `tanh` activation function.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load and preprocess the data
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
# Normalize the images to [-1, 1]
x_train = (x_train - 127.5) / 127.5
```

**Step 2: The Generator**

The generator takes a random noise vector and upsamples it to create a 28x28 image.

```python
latent_dim = 100

def build_generator():
    model = keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

generator = build_generator()
```

**Step 3: The Discriminator**

The discriminator is a standard convolutional classifier that outputs a single probability.

```python
def build_discriminator():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
    
discriminator = build_discriminator()
```

**Step 4: Define Loss Functions and Optimizers**

We use Binary Cross-Entropy. The key is that the Generator's loss is calculated based on how well it fools the Discriminator.

```python
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)
```

**Step 5: The Custom Training Loop**

GANs require a custom training loop to alternate between training the two networks.

```python
EPOCHS = 50
BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 16

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

# Define the training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f"Epoch {epoch + 1} completed.")

# Start training
train(train_dataset, EPOCHS)
```

**Step 6: Generate and View Results**

After training, you can use the generator to create new digits from random noise.

-----

### **4. VAEs vs. GANs: A Comparison**

| Feature             | Variational Autoencoders (VAEs)                                  | Generative Adversarial Networks (GANs)                                  |
| ------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Architecture** | Encoder-Decoder.                                                 | Generator-Discriminator.                                                |
| **Training** | Stable, straightforward training with a single loss function.    | Unstable, difficult to train. Requires careful hyperparameter tuning. |
| **Output Quality** | Tends to produce slightly blurrier, but more diverse images.     | Can produce very sharp, high-fidelity images.                           |
| **Latent Space** | Learns an explicit, continuous, and well-structured latent space. | Does not explicitly structure the latent space (though variations exist). |
| **Primary Use Case** | Tasks requiring a structured latent space, like interpolation. | Generating highly realistic samples.                                  |
| **Weakness** | Lower sample quality (often blurry).                             | Training instability and mode collapse (lack of diversity).             |