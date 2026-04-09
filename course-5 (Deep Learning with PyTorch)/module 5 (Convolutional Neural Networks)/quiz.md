# Quiz: Convolutional Neural Networks

## Questions

### Question 1
What is the primary purpose of using convolution in neural networks?

- To increase the speed of training
- To identify patterns in images regardless of position
- To reduce the number of parameters
- To eliminate the need for activation functions

### Question 2
What is the name of the learnable parameters in a convolution operation?

- Bias
- Kernel
- Activation
- Stride

### Question 3
What is the output of a convolution operation called?

- Filter map
- Activation map
- Feature map
- Both B and C

### Question 4
For a 4x4 image with a 2x2 kernel (no padding, stride 1), what is the size of the activation map?

- 2x2
- 3x3
- 4x4
- 5x5

### Question 5
What is the purpose of zero padding in convolution?

- To increase the speed of convolution
- To allow valid convolutions with large strides
- To reduce the activation map size
- To eliminate bias terms

### Question 6
If we have a 4x4 image with a 2x2 kernel and stride = 2, what is the size of the activation map?

- 1x1
- 2x2
- 3x3
- 4x4

### Question 7
What does a larger stride value result in?

- More detailed features detected
- Smaller activation map
- Larger activation map
- No effect on activation map size

### Question 8
In convolution, what operation is performed between the kernel and image?

- Addition
- Subtraction
- Element-wise multiplication and sum
- Division

### Question 9
What is the bias term added to in convolution?

- Only the first element of the activation map
- Every element of the activation map
- Only the diagonal elements
- It is not used in convolution

### Question 10
If we add padding = 1 to a 4x4 image, what is the new image size?

- 4x4
- 5x5
- 6x6
- 8x8

### Question 11
What happens when ReLU activation is applied to negative values in an activation map?

- They become positive
- They become zero
- They remain negative
- They are divided by 2

### Question 12
What is the primary benefit of max pooling in CNNs?

- It increases the number of parameters
- It makes the network invariant to small shifts in the input
- It eliminates the need for convolution
- It speeds up training without any tradeoffs

### Question 13
For max pooling with kernel_size=2 and stride=2 on a 4x4 input, what is the output size?

- 1x1
- 2x2
- 4x4
- 8x8

### Question 14
How does max pooling help with image translation?

- By rotating the image
- By making small shifts result in the same output
- By increasing the resolution
- By applying color filters

### Question 15
What is the operation performed in max pooling?

- Summing all values in a region
- Taking the average of values
- Selecting the maximum value
- Multiplying all values

### Question 16
For multiple output channels, how many kernels are used?

- Equal to the number of output channels
- Equal to the number of input channels
- Equal to input channels × output channels
- Always one

### Question 17
In a RGB image, how many input channels does the image have?

- 1
- 2
- 3
- 4

### Question 18
When convolving multiple input channels, what happens to the results?

- They are averaged
- They are multiplied together
- They are added together
- They are kept separate

### Question 19
If we have 3 input channels and 2 output channels, how many total kernels are there?

- 5
- 6
- 2
- 3

### Question 20
What is the primary advantage of having multiple output channels?

- Faster computation
- Detecting multiple different features
- Reducing overfitting
- Smaller model size

### Question 21
What is the purpose of the flatten step in a CNN?

- To increase the image size
- To convert 2D feature maps to 1D for fully connected layers
- To reduce the number of parameters
- To apply the activation function

### Question 22
In a CNN, where are the convolution kernels obtained from?

- They are preset values
- They are obtained via training
- They are randomly selected each time
- They are calculated from the image size

### Question 23
What is the proper order of operations in a CNN layer?

- Convolution → Pooling → Activation → Flatten
- Convolution → Activation → Pooling → Flatten
- Activation → Convolution → Pooling → Flatten
- Flatten → Convolution → Activation → Pooling

### Question 24
For a 28x28 input image with two convolution layers each followed by 2x2 max pooling, what is the final feature map size?

- 7x7
- 14x14
- 28x28
- 3x3

### Question 25
What does the fully connected layer in a CNN do?

- Extracts features from the image
- Reduces the spatial dimensions
- Makes the final classification
- Applies pooling

---

### Question 26
What is the primary purpose of the convolution operation in a convolutional neural network (CNN)?

- To increase the number of channels in the image
- To apply a non-linear activation function
- To detect local patterns in the input image
- To reduce the size of the image

### Question 27
How does zero padding affect the size of the output activation map in a convolution operation?

- It doubles the size of the activation map
- It decreases the size of the activation map
- It increases the size of the activation map
- It has no effect on the size of the activation map

### Question 28
What is the primary function of max pooling in a convolutional neural network?

- To enhance the contrast of the image
- To reduce the spatial dimensions of the activation map
- To increase the number of channels
- To apply non-linearity to the activation map

### Question 29
Which activation function sets all negative input values to zero?

- Tanh 
- Softmax
- Sigmoid
- ReLU

### Question 30
How are activation functions applied when dealing with multiple channels in a convolutional layer?

- Activation functions are applied individually to each element in every channel
- Activation functions are applied to the sum of all channels
- Activation functions are applied only to the first channel
- Activation functions are applied only to the last channel

### Question 31
In a simple CNN architecture, what is the purpose of flattening the output of the final convolutional layer?

- To increase the spatial dimensions
- To apply a pooling operation 
- To convert the 2D activation map into a 1D tensor
- To reduce the number of output channels

### Question 32
What does the term "output channels" refer to in the context of convolutional layers?

- The number of feature maps
- The number of input images
- The height of the image
- The width of the image

### Question 33
What is the primary advantage of using pre-trained models in PyTorch?

- They eliminate the need for a data set
- They automatically fine-tune the hyperparameters
- They are optimized for speed
- They provide a strong starting point

### Question 34
When using a pre-trained model in PyTorch, why is the requires_grad" attribute often set to "False" for most layers?

- To save memory during training
- To prevent modifying the pre-trained weights
- To automatically adjust learning rates
- To speed up the forward pass

### Question 35
What is the primary purpose of using a GPU in training convolutional neural networks?

- To improve the visualization of the training process
- To reduce the size of the model
- To accelerate the computation of matrix operations
- To simplify the code implementation

---

## Answers

1. To identify patterns in images regardless of position
2. Kernel
3. Both B and C (Activation map or Feature map)
4. 3x3
5. To allow valid convolutions with large strides
6. 2x2
7. Smaller activation map
8. Element-wise multiplication and sum
9. Every element of the activation map
10. 6x6
11. They become zero
12. It makes the network invariant to small shifts in the input
13. 2x2
14. By making small shifts result in the same output
15. Selecting the maximum value
16. Equal to the number of output channels
17. 3
18. They are added together
19. 6
20. Detecting multiple different features
21. To convert 2D feature maps to 1D for fully connected layers
22. They are obtained via training
23. Convolution → Activation → Pooling → Flatten
24. 7x7
25. Makes the final classification
26. To detect local patterns in the input image
27. It increases the size of the activation map
28. To reduce the spatial dimensions of the activation map
29. ReLU
30. Activation functions are applied individually to each element in every channel
31. To convert the 2D activation map into a 1D tensor
32. The number of feature maps
33. They provide a strong starting point
34. To prevent modifying the pre-trained weights
35. To accelerate the computation of matrix operations