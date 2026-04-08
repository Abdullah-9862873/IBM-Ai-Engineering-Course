# Quiz: Cross-Entropy Loss (With Answers)

## Question 1
What is the primary reason for using the cross entropy loss in logistic regression?

- To calculate the number of misclassified samples
- To minimize the variance of the model
- To average the squared errors
- **To maximize the likelihood of the correct class** ✓

## Question 2
What problem is caused using the Mean Squared Error (MSE) in logistic regression?

- **It results in a flat cost surface that can cause parameters to get stuck** ✓
- It minimizes the number of misclassified samples
- It speeds up the training process
- It increases the likelihood of correct classification

## Question 3
What is the advantage of using the sigmoid function over the threshold function in logistic regression?

- The sigmoid function reduces the number of misclassified samples to zero immediately
- **The sigmoid function provides a smooth cost surface** ✓
- The sigmoid function increases the speed of computation
- The sigmoid function is easier to implement

## Question 4
When performing logistic regression, what does the cross-entropy loss aim to minimize?

- The sum of squared residuals
- The maximum likelihood
- **The negative logarithm of the likelihood** ✓
- The difference between predicted and actual classes

## Question 5
How does PyTorch implement the cross entropy loss in logistic regression?

- Using nn.SoftmaxLoss
- Using nn.MSELoss
- Using nn.CrossEntropyLoss
- **Using nn.BCELoss** ✓

## Question 6
Why is Stochastic Gradient Descent (SGD) commonly used as the optimizer in logistic regression?

- It increases the learning rate automatically
- It guarantees finding the global minimum
- It is faster than other optimization methods
- **It minimizes the loss function by using only a portion of the dataset** ✓

## Question 7
What is the role of the sigmoid function in logistic regression?

- It computes the gradient of the loss function
- **It converts linear outputs into probabilities** ✓
- It adjusts the learning rate dynamically
- It performs thresholding on the output

## Question 8
What does the backward function do in the context of PyTorch's logistic regression?

- It updates the model parameters directly
- It resets the model parameters to their initial values
- **It computes the gradients of the loss with respect to the model parameters** ✓
- It calculates the loss function for the next epoch

## Question 9
How are the final model parameters updated in PyTorch's logistic regression?

- **By using the optimizer's step function** ✓
- By increasing the learning rate
- By recalculating the loss function
- By averaging the input data

## Question 10
After training a logistic regression model in PyTorch, what is the typical output range of the model before thresholding?

- **0 to 1** ✓
- 0 to 100
- -1 to 1
- 0 to 10