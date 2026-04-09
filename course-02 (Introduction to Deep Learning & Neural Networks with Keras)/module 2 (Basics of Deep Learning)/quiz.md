# Quiz: Basics of Deep Learning

## Question 1
_____________ is an iterative optimization algorithm for finding the minimum of a function.

**Answer: Gradient descent**

---

## Question 2
How does the backpropagation training process start?

**Answer:** By calculating the squared error 'E' between the predicted values and the ground truth labels.

---

## Question 3
Which of the following types of activation functions can cause the vanishing gradient problem? Select three answers.

**Answer:**
- The hyperbolic tangent function ✓
- The sigmoid function ✓

(Note: ReLU and softmax do not cause vanishing gradient)

---

## Question 4
Which of the following activation functions does not activate all neurons simultaneously?

**Answer: The ReLU function**

---

## Question 5
Which of the following activation functions is ideally used in the classifier's output layer?

**Answer: The softmax function**

---

## Question 1 (Practice Quiz)
Which of the following algorithms is used to optimize weights and biases in a neural network?

**Answer: Gradient descent algorithm**

---

## Question 2 (Practice Quiz)
For a cost function, J = Σ(zi - wxi - b)², that we would like to minimize, which of the following expressions represent updating the parameter, w, using gradient descent?

**Answer: w → w - η * ∂J/∂w**

---

## Question 3 (Practice Quiz)
While reviewing a chart of an activation function, you see a function that outputs zero for all negative inputs and returns the input itself for positive values. What type of activation function is this?

**Answer: ReLU function**

---

## Question 4 (Practice Quiz)
What type of activation function outputs values between -1 and 1 and has an S-shaped curve centered at the origin?

**Answer: Hyperbolic tangent function**

---

## Question 5 (Practice Quiz)
In which layer is the softmax activation function most commonly used?

**Answer: Output layer**

---

## Question 6 (Practice Quiz)
What is the correct sequence of steps in the backpropagation training algorithm?

**Answer:** Step c: Calculate the network output using forward propagation → Step a: Calculate the error → Step b: Update weights → Step d: Repeat until threshold reached
- **Correct sequence: c, a, b, d**

---

## Question 7 (Practice Quiz)
Which mathematical phenomenon explains why deeper layers in neural networks may fail to learn effectively during backpropagation?

**Answer: Exponential decay of gradient magnitudes through successive layer computations**

---

## Question 8 (Practice Quiz)
You're manually implementing backpropagation for a neural network project. When calculating how much to update the weight between the input layer and the first hidden layer, which chain rule decomposition should you use?

**Answer:** Apply the chain rule by multiplying the upstream gradient by the local gradient of the weighted sum with respect to the weight

---

## Question 9 (Practice Quiz)
You're experimenting with different activation functions in a 20-layer neural network. After training, you notice that networks using sigmoid activation learn much slower than those using ReLU. What explains this difference?

**Answer:** ReLU functions maintain stronger gradient flow through deep networks compared to sigmoid functions

---

## Question 10 (Practice Quiz)
When implementing a forward pass in a simple neural network, what is the correct order of operations for computing the output of a neuron?

**Answer:** Compute weighted sum of inputs, add bias, then apply activation function
