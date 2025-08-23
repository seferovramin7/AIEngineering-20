# Homework: Design Your Digital Fashion Collection with AI

### **Background**


In this assignment, you will act as a digital fashion designer. Your mission is to train an AI model to generate original clothing designs. Imagine creating a futuristic clothing line where every design is a unique creation from the AI you've trained.

### **The Practical Task**

Adapt the VAE or GAN code from the lesson to generate images of fashion items instead of handwritten digits. You will use the **Fashion-MNIST** dataset, which contains 70,000 grayscale images of 10 different clothing items.

This dataset has the exact same format as MNIST (28x28 grayscale images), so you will only need to make a very small change to the code.

### **Steps**

1.  **Choose Your Model:** You are free to use either the **VAE** or **GAN** code provided in `TOPIC.MD`.
2.  **Change the Dataset:** In the "Setup and Data Loading" section of your chosen code, replace the line that loads the MNIST dataset with Fashion-MNIST.

    *Find this line of code:*
    ```python
    # For VAE
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    
    # Or for GAN
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    ```

    *And replace it with the following line:*
    ```python
    # Load the Fashion-MNIST dataset
    (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    ```
3.  **Train Your Model:** Run the training process as usual. Your model will now learn the patterns of dresses, shirts, and shoes instead of numbers.
4.  **Generate Your Fashion Collection:** Once the model is trained, use the *generator* (for a GAN) or *decoder* (for a VAE) to create new, unique images of clothing designs.
5.  **Analyze the Results:** Observe the generated images. Do they look like real clothes? Do you see any interesting, strange, or completely new designs?

### **What to Submit**

1.  **One image** showing the grid of your model's creations (e.g., the output from the `plot_latent_space_digits` function in the VAE or a similar grid from the GAN generator).
2.  **A short paragraph** (3-5 sentences) with your reflections:
    *   Which model did you use (VAE or GAN)?
    *   Were the results what you expected?
    *   Which design did you find most interesting and why?

This task is designed to give you a hands-on feel for how generative models can be used in creative fields. Happy creating!