# Quiz: Shallow Neural Network

## Questions

### Question 1
What is the use of the sigmoid activation function in a neural network?

- To linearly separate data
- To introduce non-linearity by mapping input values to the range [-1, 1] 
- To multiply weights by input values 
- To introduce non-linearity by mapping input values to the range [0, 1]

### Question 2
What does a two-layer neural network consist of?

- One hidden layer and one output layer
- Two hidden layers and one output layer
- Two input layers and one output layer
- One input layer and one output layer

### Question 3
How does adding more neurons to the hidden layer affect the neural network?

- It decreases the model's flexibility
- It increases the model's complexity, leading to underfitting
- It reduces the number of parameters in the model
- It increases the model's flexibility

### Question 4
How is the scaling problem resolved after adding more neurons?

- By adding more layers
- By adjusting the weights
- By shifting the decision boundary
- By applying a different activation function

### Question 5
What is a key cause of overfitting in neural networks?

- Insufficient training data 
- High learning rate
- Too few neurons in the hidden layer
- Too many neurons in the hidden layer

### Question 6
What does underfitting indicate about a neural network?

- The model is too complex for the data
- The model has captured all patterns in the data
- The model has too many layers
- The model cannot capture the complexity of the data

### Question 7
How is a multi-class classification problem implemented in PyTorch?

- By using the sigmoid activation function 
- By using a single neuron in the output layer
- By adding more hidden layers
- By increasing the number of neurons in the output layer to match the number of classes

### Question 8
What is the criterion for loss used in the multi-class classification example in PyTorch? 

- Cross Entropy
- Binary Cross Entropy
- Mean Squared Error
- Hinge Loss

### Question 9
What is the main purpose of backpropagation in neural networks?

- To compute the gradient for updating the weights
- To apply the activation function
- To forward propagate the input data
- To reduce the number of layers in the network

### Question 10
Which activation function partially solves the vanishing gradient problem?

- Relu
- Softmax
- Tanh
- Sigmoid

---

## Answers

1. To introduce non-linearity by mapping input values to the range [0, 1]
2. One hidden layer and one output layer
3. It increases the model's flexibility
4. By adjusting the weights
5. Too many neurons in the hidden layer
6. The model cannot capture the complexity of the data
7. By increasing the number of neurons in the output layer to match the number of classes
8. Cross Entropy
9. To compute the gradient for updating the weights
10. Relu