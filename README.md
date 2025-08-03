# Generative Adversarial Networks (GANs) from Scratch

This repository contains a Jupyter Notebook implementing a basic **Generative Adversarial Network (GAN)** using TensorFlow and Keras. It is trained on the MNIST dataset to generate synthetic handwritten digits.

---

## 🧠 What is a GAN?

A **GAN** consists of two neural networks:

- **Generator (G):** Learns to map random noise to realistic-looking data (images).
- **Discriminator (D):** Learns to classify data as real (from dataset) or fake (from generator).

These models play a **minimax game** against each other.

---

## 📐 Mathematical Formulation

The GAN objective is expressed as:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

Where:

- \( x \sim p_{\text{data}}(x) \): Real data distribution  
- \( z \sim p_z(z) \): Prior on input noise variables  
- \( G(z) \): Generator output  
- \( D(x) \): Discriminator's estimated probability that \( x \) is real  

The **Generator** wants to minimize this value (fool the discriminator), and the **Discriminator** wants to maximize it (correctly classify real vs fake).

---

## 🏗️ Architecture

### Generator

- Input: Random noise vector \( z \in \mathbb{R}^{100} \)
- Fully connected + LeakyReLU + BatchNorm layers
- Output: \( 28 \times 28 \) image with `tanh` activation

### Discriminator

- Input: Flattened \( 28 \times 28 \) image
- Fully connected + LeakyReLU layers
- Output: Single scalar with `sigmoid` activation (probability real/fake)

---

## 🔧 Dependencies

Install required packages:

```bash
pip install tensorflow numpy matplotlib
