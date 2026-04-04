# Module 4: Unsupervised Learning and Generative Models in Keras - Quiz

## Question 1
**Which of the following is a common application of unsupervised learning?**

- [ ] Spam email classification
- [ ] Stock price prediction
- [ ] Sentiment analysis
- [x] Image segmentation

**Answer:** Image segmentation

**Explanation:** Image segmentation is an unsupervised learning task that involves grouping similar pixels or regions in an image without labeled data. Spam email classification, stock price prediction, and sentiment analysis are typically supervised learning tasks that require labeled training data.

---

## Question 2
**What is the role of the encoder in an autoencoder?**

- [ ] To classify input data
- [x] To compress the input data into a lower dimensional representation
- [ ] To reconstruct the original input
- [ ] To generate new data samples

**Answer:** To compress the input data into a lower dimensional representation

**Explanation:** The encoder in an autoencoder compresses the input data into a lower-dimensional latent space representation (bottleneck). The decoder is responsible for reconstructing the original input from this compressed representation. Autoencoders are not primarily used for classification or generating new data samples (though VAEs can generate data).

---

## Question 3
**Which activation function is commonly used in the output layer of an autoencoder designed for image data?**

- [ ] Tanh
- [ ] ReLU
- [ ] Softmax
- [x] Sigmoid

**Answer:** Sigmoid

**Explanation:** Sigmoid activation is commonly used in the output layer of autoencoders for image data because it outputs values in the range [0, 1], which matches the normalized pixel value range of images. This ensures the reconstructed image has valid pixel values. Tanh outputs [-1, 1], ReLU outputs [0, ∞), and Softmax is used for classification tasks.

---

## Question 4
**What is the main idea behind diffusion models in machine learning?**

- [ ] To reduce the dimensionality of the data
- [x] To generate data by iteratively refining a noisy initial sample
- [ ] To classify data points into different categories
- [ ] To cluster similar data points together

**Answer:** To generate data by iteratively refining a noisy initial sample

**Explanation:** Diffusion models are probabilistic generative models that work by starting with random noise and iteratively refining it through a reverse denoising process to produce coherent data samples. They define a forward process (adding noise) and a reverse process (denoising). Dimensionality reduction, classification, and clustering are different types of machine learning tasks.

---

## Question 5
**Which loss function is commonly used when training diffusion models for image denoising?**

- [ ] Sparse Categorical Crossentropy
- [ ] Hinge Loss
- [x] Mean Squared Error (MSE)
- [ ] Categorical Crossentropy

**Answer:** Mean Squared Error (MSE)

**Explanation:** Mean Squared Error (MSE) is commonly used for diffusion models in image denoising tasks because it measures the pixel-wise difference between the reconstructed (denoised) image and the original clean image. This is a regression task, not classification, so crossentropy losses are not appropriate. Hinge loss is typically used for SVMs.

---

## Quiz Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | Unsupervised Learning Applications | Image segmentation |
| 2 | Autoencoder Components | Encoder compresses data |
| 3 | Autoencoder Output Activation | Sigmoid |
| 4 | Diffusion Models Concept | Generate data by refining noise |
| 5 | Diffusion Model Loss Function | Mean Squared Error (MSE) |

**Total Points:** 5 points

---

## Key Concepts Reviewed

| Concept | Description |
|---------|-------------|
| **Unsupervised Learning** | Finding patterns without labels (clustering, segmentation) |
| **Autoencoder Encoder** | Compresses input to lower-dimensional representation |
| **Autoencoder Decoder** | Reconstructs input from compressed representation |
| **Output Activation** | Sigmoid for image data [0, 1] range |
| **Diffusion Models** | Generate data by iteratively denoising random samples |
| **Loss Functions** | MSE for regression/denoising tasks |

---

*End of Module 4 Quiz*
