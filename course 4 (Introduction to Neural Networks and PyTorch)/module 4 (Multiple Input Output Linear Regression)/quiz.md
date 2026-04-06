# Module 4 Quiz: Multiple Input Output Linear Regression

## Question 1
In multiple linear regression, what does each sample in the predictor matrix X represent?

- **A single row of predictor variables** ✓
- The entire set of weights and bias
- A single feature of the model
- The bias term only

---

## Question 2
How is the prediction ŷ calculated in multiple linear regression?

- By applying the sigmoid function to the input features
- By multiplying the bias term with the input vector
- **By performing the dot product of the input vector x and the weight vector w and then adding the bias term** ✓
- By summing all the input features

---

## Question 3
What is true about the dimensions of X and w in multiple linear regression?

- The number of rows in X must be equal to the number of rows in w
- The dimensions of X and w do not matter
- The number of rows in X must be equal to the number of columns in w
- **The number of columns in X must be equal to the number of rows in w** ✓

---

## Question 4
When you train a multiple linear regression model using PyTorch, what is the role of `criterion`?

- To initialize the model parameters
- To update the weights based on the learning rate
- To perform the forward pass through the model
- **To compute the loss function that measures how well the model's predictions match the target values** ✓

---

## Question 5
How does gradient descent update the weights in multiple linear regression?

- By setting the weights to the gradient of the cost function
- **By subtracting the gradient of the cost function from the weights** ✓
- By multiplying the weights by the gradient of the cost function
- By adding the gradient of the cost function to the weights

---

## Question 6
What is the main difference when using linear regression with multiple outputs compared to a single output?

- **The weights are stored in a matrix rather than a vector** ✓
- The number of input features is increased
- The cost function becomes simpler
- The bias term is no longer used

---

## Question 7
In PyTorch, what is the purpose of creating a custom module for linear regression?

- To increase the number of input features
- To manually perform backpropagation
- To create a simpler way to compute the loss function
- **To customize the forward pass and potentially add additional layers or functionality** ✓

---

## Question 8
What does the cost function measure in a multiple output linear regression model?

- The number of samples in the dataset
- The total number of model parameters
- **The sum of squared distances between predictions and targets** ✓
- The average distance between predictions and targets

---

## Question 9
How are the weights and biases updated during training in a multiple output linear regression model?

- By averaging the weights of all outputs
- By multiplying the weights by a fixed factor
- **By using the gradient of the cost function with respect to each weight and bias** ✓
- By directly assigning new random values

---

## Question 10
What is the key difference in the training process for multiple output linear regression compared to single output?

- The optimizer algorithm changes
- The number of training epochs is reduced
- The cost function only evaluates a single output
- **The prediction matrix and the weight matrix dimensions are adjusted for multiple outputs** ✓

---

# Answers Summary

| Question | Answer |
|----------|--------|
| 1 | A single row of predictor variables |
| 2 | By performing the dot product of the input vector x and the weight vector w and then adding the bias term |
| 3 | The number of columns in X must be equal to the number of rows in w |
| 4 | To compute the loss function that measures how well the model's predictions match the target values |
| 5 | By subtracting the gradient of the cost function from the weights |
| 6 | The weights are stored in a matrix rather than a vector |
| 7 | To customize the forward pass and potentially add additional layers or functionality |
| 8 | The sum of squared distances between predictions and targets |
| 9 | By using the gradient of the cost function with respect to each weight and bias |
| 10 | The prediction matrix and the weight matrix dimensions are adjusted for multiple outputs |
