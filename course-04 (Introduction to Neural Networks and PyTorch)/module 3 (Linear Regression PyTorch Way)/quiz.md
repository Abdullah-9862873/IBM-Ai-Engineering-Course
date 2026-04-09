# Module 3 Quiz: Linear Regression PyTorch Way

## Question 1
Which of the following statements is true in Stochastic Gradient Descent?

- Minimizing the error with respect to the second sample minimizes the error with respect to the third sample.
- Minimizing the error with respect to the first sample increases the error with respect to the second sample.
- The value of the approximate cost will fluctuate rapidly with each epoch.
- **The value of the approximate cost will fluctuate rapidly with each iteration.** ✓

---

## Question 2
What is the term used each time you go through the data in Stochastic Gradient Descent?

- Sample
- **Epoch** ✓
- Iteration
- Data space

---

## Question 3
Which of the following equations can you use to obtain the number of iterations in a mini-batch gradient descent?

- **iterations = training size/batch size** ✓
- iterations = training size
- iterations = batch size
- iterations = batch size/training size

---

## Question 4
What do you mean by the convergence rate in mini-batch gradient descent?

- The plot that shows the training sizes with different batch sizes.
- **The plot that shows the cost or average loss with different batch sizes.** ✓
- The plot that shows the iterations with different batch sizes.
- The plot that shows the cost or average loss with different training sizes.

---

## Question 5
Consider the code and answer the following question:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Which element in the equation will hold the current state and will update the parameters based on the computed gradients?

- optim
- parameters
- **SGD** ✓
- lr

---

## Question 6
Which function allows you to display and update the learnable parameters in your model?

- **state_dict()** ✓
- lr()
- parameters()
- model()

---

## Question 7
In the optimizer process, which function will update the parameters?

- optimizer.zero_grad()
- **optimizer.step()** ✓
- model.parameters()
- loss.backward()

---

## Question 8
Which parameters in your model are hyperparameters that can change?

- **learning rate** ✓
- cost
- bias
- slope

---

## Question 9
How can you train your model to minimize the cost of validation errors?

- By splitting the data
- **By using training data and validation data** ✓
- By using training data
- By using gradient descent

---

## Question 10
Where can you store the calculated loss while making a prediction using the training data?

- validation_error
- Models
- learning_rate
- **test_error** ✓

---

# Answers Summary

| Question | Answer |
|----------|--------|
| 1 | The value of the approximate cost will fluctuate rapidly with each iteration. |
| 2 | Epoch |
| 3 | iterations = training size / batch size |
| 4 | The plot that shows the cost or average loss with different batch sizes. |
| 5 | SGD |
| 6 | state_dict() |
| 7 | optimizer.step() |
| 8 | learning rate |
| 9 | By using training data and validation data |
| 10 | test_error |
