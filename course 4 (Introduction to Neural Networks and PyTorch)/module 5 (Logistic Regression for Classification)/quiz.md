# Module 5 Quiz: Logistic Regression for Classification

## Question 1
In logistic regression, what function is used to convert the linear output to a probability?

- **Sigmoid function** ✓
- ReLU function
- Softmax function
- Tanh function

---

## Question 2
In PyTorch, which module is used to quickly build logistic regression models?

- **nn.Sequential** ✓
- nn.Module
- nn.Linear
- nn.Sigmoid

---

## Question 3
What is the output range of the sigmoid function?

- **0 to 1** ✓
- -1 to 1
- 0 to infinity
- -infinity to infinity

---

## Question 4
When using a custom module for logistic regression, what does the forward method do?

- Initializes model parameters
- **Applies linear function and then sigmoid** ✓
- Calculates loss
- Updates weights

---

## Question 5
In nn.Sequential, how are layers connected?

- Each layer is trained independently
- **Output of one layer becomes input to the next** ✓
- All layers receive the same input
- Layers are connected randomly

---

## Question 6
What type of loss function is typically used for logistic regression in PyTorch?

- **BCELoss (Binary Cross Entropy)** ✓
- MSELoss
- L1Loss
- SmoothL1Loss

---

## Question 7
For a 2D input logistic regression model in PyTorch, what would be the input dimension for nn.Linear?

- 1
- **2** ✓
- 3
- Any number

---

## Question 8
How do you convert sigmoid output to binary class prediction (0 or 1)?

- Round the output
- **Use threshold of 0.5** ✓
- Take absolute value
- Use maximum value

---

## Question 9
Which of the following is true about nn.Sequential in PyTorch?

- It requires manual forward pass implementation
- **It automatically chains layers together** ✓
- It cannot be used for classification
- It only works with linear layers

---

## Question 10
When making predictions with logistic regression on multiple samples, how is the output structured?

- Single scalar value
- **Vector of probabilities (one per sample)** ✓
- Matrix of predictions
- Tensor of same shape as input

---

# Answers Summary

| Question | Answer |
|----------|--------|
| 1 | Sigmoid function |
| 2 | nn.Sequential |
| 3 | 0 to 1 |
| 4 | Applies linear function and then sigmoid |
| 5 | Output of one layer becomes input to the next |
| 6 | BCELoss (Binary Cross Entropy) |
| 7 | 2 |
| 8 | Use threshold of 0.5 |
| 9 | It automatically chains layers together |
| 10 | Vector of probabilities (one per sample) |

---

## Question 1
What does logistic regression predict?

- **The class a sample belongs to** ✓
- The age of a person
- The price of an object
- The weight of an object

---

## Question 2
In the context of logistic regression, what does the class vector "y" represent?

- Different features of the samples
- The bias term in the equation
- Continuous values of the samples
- **Discrete class labels for each sample** ✓

---

## Question 3
If a data set can be separated by a line, what is it called?

- Unclassifiable
- Nonlinear
- **Linearly separable** ✓
- Multiclass

---

## Question 4
In the equation of a line w·x + b, what does "b" represent?

- Feature
- Sample value
- Weight term
- **Bias term** ✓

---

## Question 5
What function is used in logistic regression to obtain the final output?

- Tanh function
- ReLU function
- **Sigmoid function** ✓
- Linear function

---

## Question 6
Which PyTorch package is used for quickly building logistic regression models?

- **torch.nn.Sequential** ✓
- torch.optim
- torch.autograd
- torch.nn.functional

---

## Question 7
In logistic regression, what is the function of the nn.Sigmoid() method?

- **It applies the sigmoid activation function** ✓
- It initializes the model parameters
- It applies a linear transformation
- It creates a linear model

---

## Question 8
What does the parameter θ (theta) represent in a Bernoulli distribution?

- The standard deviation
- **The probability of success** ✓
- The probability of failure
- The variance of the distribution

---

## Question 9
How is the likelihood of a sequence of events calculated in a Bernoulli distribution?

- By dividing the probabilities of individual events
- **By multiplying the probabilities of individual events** ✓
- By adding the probabilities of individual events
- By subtracting the probabilities of individual events

---

## Question 10
What is the purpose of the cross-entropy loss in logistic regression?

- To regularize the model parameters
- **To minimize the number of misclassified samples** ✓
- To maximize the number of misclassified samples
- To increase the learning rate

---

# Answers Summary (Full)

| Question | Answer |
|----------|--------|
| 1 | Sigmoid function |
| 2 | nn.Sequential |
| 3 | 0 to 1 |
| 4 | Applies linear function and then sigmoid |
| 5 | Output of one layer becomes input to the next |
| 6 | BCELoss (Binary Cross Entropy) |
| 7 | 2 |
| 8 | Use threshold of 0.5 |
| 9 | It automatically chains layers together |
| 10 | Vector of probabilities (one per sample) |
| 11 | The class a sample belongs to |
| 12 | Discrete class labels for each sample |
| 13 | Linearly separable |
| 14 | Bias term |
| 15 | Sigmoid function |
| 16 | torch.nn.Sequential |
| 17 | It applies the sigmoid activation function |
| 18 | The probability of success |
| 19 | By multiplying the probabilities of individual events |
| 20 | To minimize the number of misclassified samples |
