# Module 2 Quiz: Linear Regression

## Question 1
Consider the code:

```python
class LR(nn.Module):
    def __init__(self, in_features, out_features):
        super(LR, self).__init__()
        linear = nn.Linear(in_features, out_features)
```

What is wrong with the code?

- "LR" is not required
- **"linear" should be self.linear** ✓
- "super" is not required in the code
- "nn.Module" is not required

---

## Question 2
What does the term "noise" refer to in linear regression?

- The lack of a linear relationship between x and y
- Variations in the model's parameters
- **Random errors added to the data points** ✓
- Errors in the data collection process

---

## Question 3
What is the purpose of the mean squared error (MSE) in linear regression?

- To measure the accuracy of the model predictions
- To compute the standard deviation of the data set
- **To calculate the average squared difference between predicted and actual values** ✓
- To evaluate the model's complexity

---

## Question 4
What is the primary goal of gradient descent in the context of linear regression?

- **To minimize the cost function** ✓
- To compute the gradient of the input features
- To standardize the features of the data set
- To find the maximum value of the cost function

---

## Question 5
What issue can arise if the learning rate in gradient descent is too large?

- The learning process will take longer
- **The algorithm may miss the minimum of the cost function** ✓
- The algorithm may converge to a suboptimal solution
- The algorithm may converge too quickly

---

## Question 6
In PyTorch, why is the "requires_grad" attribute set to "True" for a tensor used in training?

- To visualize the tensor in plots
- To improve the performance of the model
- **To automatically compute the gradients for the tensor** ✓
- To make the tensor immutable during training

---

## Question 7
In gradient descent, what happens to the loss function if the learning rate is set too small?

- **The convergence to the minimum is very slow** ✓
- The loss function may increase rapidly
- The parameter updates become large
- The parameter values oscillate around the minimum

---

## Question 8
What does the term "cost surface" represent in the context of linear regression?

- The gradient of the cost function with respect to parameters
- A matrix representing the data features
- **The plot showing how different parameter values affect the cost** ✓
- A graphical representation of the data points

---

## Question 9
What is the role of the "forward" function in a PyTorch model?

- To compute the loss function
- To initialize model parameters
- To apply transformations to input data
- **To perform the forward pass and compute predictions** ✓

---

## Question 10
What is the significance of contour plots in understanding the cost function?

- **They represent slices of the cost surface at different heights, helping to visualize how cost changes with parameters** ✓
- They show the distribution of data points in the feature space
- They provide a 3D view of how cost changes with different parameter values
- They help visualize the gradient of the cost function

---