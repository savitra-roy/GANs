# Generative-Adversarial-Network-for-CIFAR-10

A deep learning project that generates realistic 32x32 color images from random noise.

---

## Overview

- Utilizes a **Convolutional Neural Network (CNN)** based architecture for both the Generator and Discriminator.
- Based on the principles of Generative Adversarial Networks to produce novel images.
- Trained on the popular **CIFAR-10 dataset**.
- The Generator learns to create images that are convincing enough to fool the Discriminator.

---

## Model Highlights

- **Generator Architecture:** A deep convolutional network that uses `Conv2DTranspose` layers to upsample a 100-dimensional latent noise vector into a 32x32x3 image. Uses `LeakyReLU` and `tanh` activation functions.
- **Discriminator Architecture:** A standard CNN with `Conv2D` layers designed to classify input images as either "real" (from the dataset) or "fake" (from the generator). It uses `LeakyReLU` activation and a final `sigmoid` output.
- **Regularization:** `Dropout` is used in the Discriminator to prevent overfitting.
- **Dataset:** Trained on the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes.
- **Results:**
  - The model was successfully trained for **100 epochs**.
  - Generated images show increasing coherence and detail as training progresses.

---

## Key Steps

- **Data Loading & Preprocessing:** Loaded the CIFAR-10 dataset and scaled the images to the `[-1, 1]` range to work with the `tanh` activation function of the generator.
- **Model Building:** Constructed the Generator and Discriminator models from scratch using Keras and TensorFlow.
- **Adversarial Training:** Created a combined GAN model where the Discriminator's weights are frozen, allowing the Generator to be trained to fool the Discriminator.
- **Iterative Improvement:** The models were trained in an adversarial loop, where in each step, the Discriminator is trained on real and fake images, and then the Generator is trained to improve its image generation.

---

## Usage

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone [https://github.com/savitra-roy/GANs.git](https://github.com/savitra-roy/GANs.git)
    cd GANs
    pip install tensorflow keras matplotlib numpy
    ```
2.  **Open the notebook:** The entire implementation is contained within a Jupyter Notebook.
    ```bash
    jupyter notebook GANs.ipynb
    ```
3.  **Run the cells:** Execute the cells in the notebook to train the model and generate images.

---

## Findings

- The adversarial training process is highly sensitive to hyperparameters. The balance between the Generator and Discriminator is crucial for stable training.
- The quality of the generated images visibly improves with more training epochs, demonstrating the model's ability to learn the underlying data distribution of the CIFAR-10 dataset.
- Using `LeakyReLU` in the discriminator and `tanh` in the generator's output layer are effective choices for GAN architectures.

---

## Future Directions

- **Implement a more advanced GAN architecture:** Experiment with architectures like Wasserstein GAN (WGAN) or StyleGAN for more stable training and higher-quality image generation.
- **Train on a higher-resolution dataset:** Adapt the model to work with datasets like CelebA or LSUN to generate more detailed images.
- **Conditional GAN:** Modify the model to be a conditional GAN (cGAN), allowing it to generate images of specific classes from the CIFAR-10 dataset.
- **Deploy the model:** Create a web application where users can see newly generated images from the trained generator model.
