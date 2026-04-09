# Module 3 Quiz: CNN Vision Transformer Integration

Answer the following questions by selecting the correct option.

---

### Question 1

At InnovateAI, a team of AI engineers led by Sarah is tasked with developing an image classification model using Keras. They decide to leverage vision transformers and transfer learning. What should be their initial step in implementing this model?

- [ ] Design a custom vision transformer architecture from scratch
- [ ] Implement a convolutional neural network as a baseline before using vision transformers
- [x] Load a pre-trained vision transformer model and prepare the dataset for transfer learning
- [ ] Collect a large dataset to train the vision transformer from scratch

---

### Question 2

Imagine that at AI Solutions Corp., the team led by Jamie is tasked with summarizing the use of vision transformers in Keras. What aspect should they emphasize to demonstrate the application of transfer learning in computer vision tasks?

- [ ] The exclusive use of convolutional layers in vision transformers
- [ ] The necessity of large datasets to train vision transformers effectively
- [ ] The requirement for extensive hyperparameter tuning in every application
- [x] The ability to adapt pre-trained models for specific tasks with minimal data

---

### Question 3

At TechVision Inc., Alex is implementing a vision transformer model using PyTorch for a domain-specific image dataset. Which key step should Alex focus on to effectively apply transfer learning?

- [ ] Use the model without any modifications on the new dataset
- [ ] Train the model from scratch with the new dataset
- [ ] Implement additional layers to the existing model
- [x] Fine-tune the pre-trained model on the new dataset

---

### Question 4

What is the function of the self.features module inside the ConvNet class?

- [x] It applies the pre-trained model architecture for CNN-based feature extraction.
- [ ] It implements the transformer encoder for the hybrid architecture.
- [ ] It computes the loss for the model during training.
- [ ] It loads the dataset into the model.

---

### Question 5

What is the primary operation performed by the PatchEmbed class's proj layer in the Vision Transformer implementation?

- [ ] Divides the image into overlapping patches using a 3×3 convolution
- [ ] Applies a pooling operation to flatten patches
- [x] Projects the input feature map to a lower-dimensional embedding using a 1×1 convolution
- [ ] Flattens the original image directly into a sequence

---

### Question 6

When creating the positional encoding for image patches, which of the following operations is typically performed?

- [ ] Reshuffling the patches randomly each epoch
- [x] Adding a learned or fixed vector to each patch embedding
- [ ] Subtracting the patch mean from each patch
- [ ] Concatenating zeros to patch vectors

---

### Question 7

During Vision Transformer training, what is the typical purpose of including a "Classification Head" at the end of the model?

- [ ] To encode positional information
- [x] To project the final encoder output to the number of target classes
- [ ] To resize the image to its original shape
- [ ] To increase model regularization

---

## Answer Key

| Question | Answer | Explanation |
|----------|--------|--------------|
| Question 1 | Load a pre-trained vision transformer model | Transfer learning starts with loading a pre-trained model and adapting it to the target dataset. |
| Question 2 | Ability to adapt pre-trained models | Transfer learning allows adapting pre-trained models for specific tasks with minimal data. |
| Question 3 | Fine-tune the pre-trained model | Fine-tuning adapts pre-trained weights to the new domain-specific dataset. |
| Question 4 | CNN-based feature extraction | The `self.features` module applies the pre-trained CNN for extracting features. |
| Question 5 | Project to embedding using 1×1 convolution | PatchEmbed projects patches to lower-dimensional embeddings. |
| Question 6 | Adding a learned or fixed vector | Positional encoding is added to patch embeddings to maintain position info. |
| Question 7 | Project to number of target classes | Classification head maps encoder output to class predictions. |

---

**Total Points: 7 points (1 point each)**

Complete all tasks in the lab to answer questions correctly.