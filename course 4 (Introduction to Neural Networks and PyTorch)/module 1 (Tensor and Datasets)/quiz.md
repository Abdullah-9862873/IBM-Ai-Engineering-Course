# Module 1 Quiz: Tensors and Datasets

## Question 1
What is the result of the following:
```python
a = torch.tensor([1, 2, 3])
a.dtype
```

- torch.float32
- Float
- **torch.int64** ✓
- LongTensor

---

## Question 2
Which type of tensor type is used when dealing with unsigned integers that are used in 8-bit images?

- Float
- **Byte** ✓
- Integer
- Double

---

## Question 3
What is a two-dimensional tensor in PyTorch, and how can it be visualized?

- **A 2D tensor is a matrix, with each row representing a sample and each column representing a feature** ✓
- A 2D tensor is a 1D array with multiple dimensions
- A 2D tensor represents a single channel in a color image
- A 2D tensor is a scalar value and can be visualized as a single point

---

## Question 4
Which method would you use to get number of dimensions or the rank of the tensor in PyTorch?

- ndimension()
- size()
- numel()
- **shape()** ✓

---

## Question 5
Which of the following can be used to access the value in the second row and third column of a 2D tensor using indexing in PyTorch?

- tensor[2][1]
- tensor[1][2]
- tensor[2, 1]
- **tensor[1, 2]** ✓

---

## Question 6
Consider the following code:
```python
a = torch.tensor([[0, 1, 1], [1, 0, 1]])
```
What is the output of `a.size()` and `a.ndimension()`?

- **(2, 3), 2** ✓
- (2, 3), 3
- (3, 2), 2
- (3, 2), 3

---

## Question 7
When creating a custom dataset class for images, what method needs to be overridden to access the items in the data set?

- __iter__
- **__getitem__** ✓
- __len__
- __init__

---

## Question 8
How do you apply a transform to a data set object in PyTorch?

- **By passing the transform object to the data set class constructor** ✓
- By using the apply_transform() method on the data set object
- By calling the transform() method directly on the data set object
- By manually transforming each sample in the data set

---

## Question 9
Which technique is most effective for applying several image transformations in a sequence for a dataset?

- By converting image data into tensors
- By splitting the data set into training and testing sets
- By applying a single transform to a data set
- **By using Compose to chain multiple transforms together** ✓

---

## Question 10
In the context of building a data set for images using PyTorch, what is the purpose of using the Image.open() function?

- To open the image file and load its contents into a tensor
- To obtain the path of the image file
- **To read the image file and convert it into a format that PyTorch can process** ✓
- To display the image directly on the screen

---

# Answers Summary

| Question | Answer |
|----------|--------|
| 1 | torch.int64 |
| 2 | Byte |
| 3 | A 2D tensor is a matrix, with each row representing a sample and each column representing a feature |
| 4 | shape() |
| 5 | tensor[1, 2] |
| 6 | (2, 3), 2 |
| 7 | __getitem__ |
| 8 | By passing the transform object to the data set class constructor |
| 9 | By using Compose to chain multiple transforms together |
| 10 | To read the image file and convert it into a format that PyTorch can process |
