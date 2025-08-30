# Homework: Design Your Digital Fashion Collection with AI

## Background
In this assignment, you will act as a digital fashion designer. Your mission is to train an AI model to generate original clothing designs. Imagine creating a futuristic clothing line where every design is a unique creation from the AI you've trained.

## The Practical Task
Adapt the **VAE** or **GAN** code from the lesson to generate images of fashion items instead of handwritten digits. Use the **Fashion-MNIST** dataset (70,000 grayscale images of 10 clothing categories). Its format is identical to MNIST (28×28 grayscale), so you only need a tiny change.

## Steps
1. **Choose Your Model:** Use either the **VAE** or **GAN** reference in `TOPIC.MD`.
2. **Change the Dataset:** Replace MNIST loading with Fashion-MNIST.

   **Find:**
   ```python
   # VAE
   (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
   # GAN
   (x_train, _), (_, _) = keras.datasets.mnist.load_data()
