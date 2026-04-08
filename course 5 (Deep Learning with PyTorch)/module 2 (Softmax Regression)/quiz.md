# Quiz: Softmax Function (With Answers)

## Question 1
What type of classification problems does the Softmax function handle?

- Binary classification
- **Multiclass classification (more than 2 classes)** ✓
- Regression problems
- Clustering problems

## Question 2
What does the argmax function return?

- The maximum value in a sequence
- **The index corresponding to the largest value** ✓
- The sum of all values
- The average value

## Question 3
In Softmax, what are the "logits" (z)?

- Final probabilities after Softmax is applied
- **Raw outputs from the model before applying Softmax** ✓
- The input features
- The gradient values

## Question 4
How do you correctly get the predicted class from Softmax output in PyTorch?

- Using torch.max()
- **Using torch.argmax()** ✓
- Using torch.mean()
- Using torch.sum()

## Question 5
What is the key difference between the custom Softmax module and custom logistic regression module in PyTorch?

- Softmax uses sigmoid activation
- **Softmax has output_size parameter for number of classes** ✓
- Logistic regression has more parameters
- They are identical implementations

## Question 6
In PyTorch, what does torch.softmax(z, dim=1) return?

- Raw logits
- **Probabilities that sum to 1 across classes** ✓
- Class labels
- Gradient values

## Question 7
What does the dim=1 parameter do in torch.argmax(z, dim=1)?

- Finds maximum across rows
- **Finds maximum across columns (classes) for each sample** ✓
- Reduces to single value
- Computes mean

## Question 8
How does Softmax use lines to classify data?

- By using curved decision boundaries
- **By finding which weight vector is closest to the input** ✓
- By clustering similar points together
- By calculating the mean of inputs

## Question 9
For a 3-class classification problem with 2D input, what would be the correct Softmax model initialization?

- Softmax(2, 1)
- **Softmax(2, 3)** ✓
- Softmax(3, 2)
- Softmax(1, 3)

## Question 10
When handling multiple input samples in Softmax, what does the prediction tensor contain?

- Probability values for each class
- **Class index for each sample** ✓
- Gradient values
- Loss values

## Question 11
What is the input dimension for the Softmax classifier when working with MNIST images?

- 10
- 28
- **28 × 28 = 784** ✓
- 784 × 10

## Question 12
What does the view(-1, 784) method do in the training code?

- Creates a 784×784 matrix
- **Flattens each 28×28 image to a 1D vector of 784 elements** ✓
- Resizes images to 784×784
- Converts tensor to float

## Question 13
What type of tensor does the label (y) need to be for CrossEntropyLoss in PyTorch?

- Float tensor
- **Long tensor** ✓
- Double tensor
- Byte tensor

## Question 14
When using PyTorch's CrossEntropyLoss, does it automatically apply Softmax?

- **Yes, it applies Softmax internally** ✓
- No, Softmax must be applied manually
- Only if specified
- Only for binary classification

## Question 15
What do the weight parameters look like after training the Softmax classifier on MNIST?

- Random noise
- **Resemble the digit shapes (0-9)** ✓
- All zeros
- Uniform gray images

## Question 16
In the validation loop, what does torch.max(y_hat, 1) return?

- Only the maximum values
- **Both maximum values and their indices** ✓
- Only the indices
- The mean of all values

## Question 17
How is accuracy calculated in the validation step?

- **(correct predictions) / (total samples)** ✓
- (incorrect predictions) / (total samples)
- (correct predictions) × (total samples)
- (total samples) / (correct predictions)

## Question 18
What is the batch size used for training the MNIST Softmax classifier?

- 10
- **100** ✓
- 5000
- 784

## Question 19
What does the transform=transforms.ToTensor() do when loading MNIST?

- Converts images to PIL format
- **Converts images to PyTorch tensors** ✓
- Normalizes the images
- Resizes images to 28×28

## Question 20
How many output classes are there in MNIST classification?

- 2
- 5
- **10** (digits 0-9) ✓
- 28

## Question 21
What does the Softmax function do in a 1D case?

- **Generalizes logistic regression to handle multiple classes** ✓
- Converts input vectors into integer classes
- Only works with two-dimensional input
- Classifies data into binary classes only

## Question 22
In the context of the Softmax function, what does the argmax function return?

- **The index corresponding to the largest value** ✓
- The average of all values
- The index corresponding to the smallest value
- The sum of all values

## Question 23
When visualizing the Softmax function in 2D, what is the purpose of the weight vectors (w0, w1, w2)?

- **They represent the parameters used to classify input samples** ✓
- They are used to calculate the mean square error
- They define the boundaries of the feature space
- They define the different colors for plotting

## Question 24
How does the Softmax function handle multidimensional inputs in the MNIST dataset?

- It uses only a subset of pixels for classification
- **It flattens the input images to 1D vectors** ✓
- It averages the pixel values
- It adds an additional dimension to the input

## Question 25
What is the purpose of using the Softmax function in PyTorch?

- To optimize the loss function
- **To classify inputs into multiple output classes** ✓
- To classify inputs into two classes only
- To perform regression tasks

## Question 26
In a Softmax classification model, what does the max function applied to "z" return?

- The index of the smallest value in "z"
- The minimum value in "z"
- **The index of the largest value in "z"** ✓
- The average of the values in "z"

## Question 27
What does the "out size" parameter in the custom Softmax module constructor in PyTorch correspond to?

- The size of the training data set
- The learning rate
- **The number of classes in the output** ✓
- The number of input features

## Question 28
When applying the Softmax function to multiple input samples, how are the results obtained for each sample?

- By averaging the samples
- **By calculating the dot product for each sample** ✓
- By finding the maximum value across all samples
- By performing element-wise multiplication

## Question 29
Which loss function is commonly used with the Softmax function in PyTorch for classification tasks?

- **Cross Entropy Loss** ✓
- L2 Regularization
- Mean Squared Error
- Hinge Loss

## Question 30
What happens to the weight parameters of the Softmax model after training on the MNIST dataset?

- They get discarded after each epoch
- They converge to zero
- They remain random
- **They start resembling the output classes** ✓