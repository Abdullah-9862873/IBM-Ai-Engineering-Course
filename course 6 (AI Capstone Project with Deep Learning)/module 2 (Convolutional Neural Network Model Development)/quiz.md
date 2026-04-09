# Module 2 Quiz: CNN Model Development

Answer the following questions by selecting the correct option.

---

### Question 1

Jane, an AI Engineer at GreenField Innovations, is implementing a CNN model using PyTorch to classify images of agricultural fields. She needs to evaluate the model's performance. Which metric should she use to assess the model's ability to distinguish between classes?

- [ ] Mean Squared Error
- [ ] Model accuracy
- [ ] Cross-entropy loss
- [x] AU-ROC (Area Under Receiver Operating Characteristic curve)

---

### Question 2

At FarmTech Inc., Liam is constructing a CNN model using PyTorch to classify images of different crop types. He needs to ensure the model's architecture is suitable for the task. What is a key component he should include in the CNN model?

- [ ] Dropout layers to increase model capacity
- [x] Convolutional layers to extract spatial features
- [ ] Recurrent layers to capture temporal sequences
- [ ] Batch normalization layers to reduce model size

---

### Question 3

At AgriTech Solutions, the data science team is tasked with classifying satellite images of agricultural land. They plan to build a CNN classifier using Keras. Which of the following steps should they take first in this process?

- [ ] Initialize the CNN model architecture with layers.
- [ ] Train the CNN model using the preprocessed dataset.
- [x] Preprocess the dataset to ensure all images are of the same size.
- [ ] Compile the CNN model with an appropriate loss function and optimizer.

---

### Question 4

Why is HeUniform() passed as kernel_initializer to many convolutional and dense layers?

- [ ] It regularizes the model with L2 norms.
- [ ] It schedules the learning rate automatically.
- [ ] It freezes pretrained ImageNet weights.
- [x] It initializes weights suited to ReLU-based activations.

---

### Question 5

Inside the evaluation (validation) loop, why is torch.no_grad() used?

- [ ] To randomize batch order
- [ ] To reset model weights
- [ ] To enable GPU acceleration
- [x] To disable gradient calculation, saving memory and computation

---

### Question 6

Why might F1 Score be preferred over accuracy on highly imbalanced datasets?

- [ ] It is threshold-independent like ROC-AUC.
- [ ] It considers only true positives and true negatives.
- [ ] It measures the area under the precision-recall curve.
- [x] It balances precision and recall, focusing on minority-class performance.

---

### Question 7

How can you generate a detailed text summary containing per-class precision, recall, and F1?

- [ ] confusion_matrix
- [ ] roc_curve
- [x] classification_report
- [ ] accuracy_score

---

## Answer Key

| Question | Answer | Explanation |
|----------|--------|--------------|
| Question 1 | AU-ROC | AU-ROC measures the model's ability to distinguish between classes, making it suitable for evaluating classification performance. |
| Question 2 | Convolutional layers | Convolutional layers are the key component of CNNs that extract spatial features from images. |
| Question 3 | Preprocess the dataset | Data preprocessing (ensuring uniform image size) is the first step before building and training the model. |
| Question 4 | It initializes weights suited to ReLU-based activations | HeUniform initializer is designed for ReLU/ReLU-like activations to prevent vanishing gradients. |
| Question 5 | To disable gradient calculation | `torch.no_grad()` disables gradient computation during evaluation, saving memory and computation. |
| Question 6 | It balances precision and recall | F1 Score is the harmonic mean of precision and recall, making it better for imbalanced datasets. |
| Question 7 | classification_report | Scikit-learn's `classification_report` provides per-class precision, recall, and F1 scores. |

---

**Total Points: 7 points (1 point each)**

Complete all tasks in the lab to answer questions correctly.