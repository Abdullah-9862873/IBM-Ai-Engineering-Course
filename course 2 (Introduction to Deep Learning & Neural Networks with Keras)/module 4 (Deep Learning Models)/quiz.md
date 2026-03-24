# Quiz: Deep Learning Models

## Question 1
Which of the following are applications of restricted Boltzmann machines (RBMs)? Select all that apply.

**Answer:**
- Estimating missing values in different features of a data set ✓
- Automatically extracting features from unstructured data ✓
- Fixing imbalanced data sets ✓

---

## Question 2
Which two of the following statements are correct? Select all that apply.

**Answer:**
- A network with three or more hidden layers and a large number of neurons in each layer is considered a deep neural network. ✓
- The three main factors that contributed to the sudden boom in the deep learning field are advancements in deep learning, data availability, and improvements in computational power. ✓

---

## Question 3
Which layer in a convolutional neural network has the main objective of reducing the spatial dimensions of the data propagating through the network?

**Answer: Pooling layer**

---

## Question 4
Which type of neural networks also take in as input the output from the previous data point that was fed into the network?

**Answer: Recurrent neural networks**

---

## Question 5
Which components and steps are involved in the self-attention mechanism used in transformers?

**Answer:** Query, key, and value vectors; attention scores; and weighted sum

---

## Question 1 (Practice Quiz)
Which layer in a convolutional neural network flattens the output of the last convolutional layer and connects every node of the current layer with every other node of the next layer?

**Answer: Fully connected layer**

---

## Question 2 (Practice Quiz)
Which mechanism enables recurrent neural networks to retain sequential dependencies across temporal steps?

**Answer:** Hidden state propagation that accumulates context from previous time steps

---

## Question 3 (Practice Quiz)
What is the primary advantage of deep neural networks over shallow neural networks in complex machine learning tasks?

**Answer:** Deep networks can learn hierarchical feature representations and capture complex patterns

---

## Question 4 (Practice Quiz)
Which architectural mechanism allows transformers to model long-range dependencies more effectively than sequential processing models?

**Answer:** Self-attention mechanisms that directly connect distant sequence positions

---

## Question 5 (Practice Quiz)
What is the primary purpose of autoencoders in unsupervised learning?

**Answer:** To learn compressed representations of input data and reconstruct the original input

---

## Question 6 (Practice Quiz)
Which neural network architecture is specifically optimized for spatial feature extraction and visual pattern recognition tasks?

**Answer: Convolutional neural networks**

---

## Question 7 (Practice Quiz)
What are the benefits of the pooling layer in a convolutional neural network? Select all that apply.

**Answer:**
- Reduce the spatial dimensions of the data propagating through the neural network ✓
- Provide spatial variance so that the neural network can recognize objects in an image even if the object does not exactly resemble the original object ✓

---

## Question 8 (Practice Quiz)
When implementing a convolutional neural network in Keras for image classification, which layer sequence is most appropriate for the initial feature extraction stage?

**Answer:** Convolution → Activation → Pooling

---

## Question 9 (Practice Quiz)
Which architectural component compensates for the parallel processing nature of transformer models in sequence modeling?

**Answer:** Sinusoidal positional encodings for sequence position awareness

---

## Question 10 (Practice Quiz)
A healthcare startup is working with a dataset of medical images to detect abnormalities. However, the dataset is highly imbalanced, with very few abnormal cases compared to normal ones. The team decides to use an unsupervised learning approach to better understand the data distribution and generate additional samples for the minority class.

**Answer:** Use a Restricted Boltzmann Machine to learn data distribution and generate new samples
