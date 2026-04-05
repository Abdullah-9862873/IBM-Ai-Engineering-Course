# Course Overview: Introduction to Neural Networks with PyTorch

## Course Description
PyTorch is one of the top 10 highest paid skills in IT technology today. The use of PyTorch in neural networks is most common, and thus, professionals with PyTorch skills are highly sought after by all IT organizations. This course is suitable for all aspiring AI engineers who want to gain fundamental knowledge of neural networks using PyTorch.

As an AI developer, you will use PyTorch to design, train, and optimize neural networks to enable computers to perform tasks, such as image recognition, natural language processing and predictive analytics.

## Course Structure (6 Modules)

At the end of this course, you will be able to:
- Perform tensor operations in PyTorch
- Implement and train linear regression models from scratch
- Apply logistic regression to classification problems
- Handle data and training models using gradient descent for optimization

---

### Module 1: Tensor Basics
- Learn the basics of 1D tensors
- Apply various methods to classify the type of data in a tensor and the type of tensor
- Differentiate simple and partial derivatives in PyTorch
- Build a simple dataset class and object
- Build a dataset for images

### Module 2: Linear Regression
- Learn about linear regression and classes
- Build custom modules using nn.Module to make predictions
- Minimize cost and calculate loss using PyTorch
- Understand gradient descent method and apply it on the cost function

### Module 3: Stochastic Gradient Descent
- Implement stochastic gradient descent using PyTorch's data loader
- Compare batch gradient descent and stochastic gradient descent
- Learn about convergence rate and using PyTorch's optimization modules
- Best practices for splitting data for robust model evaluation and training

### Module 4: Multiple Linear Regression
- Use the class Linear to perform linear regression in multiple dimensions
- Learn about model parameters and how to calculate cost and perform gradient descent in PyTorch
- Extend linear regression for multiple outputs

### Module 5: Logistic Regression
- Learn the fundamentals of linear classifiers and logistic regression
- Implement logistic regression for prediction
- Cover statistical concepts: Bernoulli distribution and Maximum Likelihood Estimation

### Module 6: Final Project
- Implement the final project
- Build a logistic regression model aimed at predicting the outcome of League of Legends matches

---

## Key Skills Learned
- Tensor 1D operations
- Linear Regression
- Logistic Regression
- Gradient Descent
- Data Loader
- Optimization
- Training, Validation, and Test Split

## Prerequisites
- Basic knowledge of Python programming
- Familiarity with PyTorch
- Use of Git and GitHub as code repository

## Related Courses in AI Engineering Professional Certificate
1. Machine Learning with Python
2. Introduction to Deep Learning and Neural Networks with Keras
3. Deep Learning with Keras and TensorFlow
4. Deep Learning with PyTorch
5. AI Capstone Project with Deep Learning
6. Generative AI and LLMs Architecture and Data Preparation
7. GenAI Model Foundations for NLP and Language Understanding
8. Generative AI Language Modeling with Transformers
9. AI Engineering with Transformer Based LLMs
10. Project: Generative AI Applications with RAG and LangChain

---

# Module 1: Tensors and Datasets

## Video: Tensors - Building Blocks of Neural Networks

### What is a Tensor?
- A **Pytorch tensor** is a data structure that is a generalization for numbers and dimensional arrays in Python
- It is the fundamental building block of neural networks in PyTorch
- In neural networks:
  - Input `x` is a tensor
  - Output `y` is a tensor
  - Network parameters are also tensors

### Tensor Operations
- Neural networks apply a series of tensor operations on the input
- Many tensor operations are generalizations of familiar mathematical operations like multiplication and addition
- Focus on tensor operations that are vector and matrix operations

### Converting Data to Tensors
- **Database example**: Each row of a database can be treated as a PyTorch tensor
- **Image example**: Images are converted to rectangular tensors (2D or 3D arrays)
- Images in Python are usually represented as 2D arrays and 3D arrays
- Each tensor input is simply a matrix or rectangular array

### Tensor Features
- **Easy conversion**: PyTorch tensors can be easily converted to NumPy arrays and vice-versa
  - This gives PyTorch the ability to work within the Python ecosystem
- **GPU integration**: Can easily integrate PyTorch with GPU
  - This is an important factor for training neural networks

### Parameters and Gradients
- **Parameters** in neural networks are a kind of tensor that allow you to calculate gradients or derivatives
- **Gradients and derivatives** allow you to train the neural network
- Use parameters for neural networks in PyTorch by setting `requires_grad = True`

### Dataset Class
- Using the Dataset class makes it much easier to work with large datasets in PyTorch

### Topics Covered in This Module
1. Tensors in 1-Dimension and 2-Dimensions
2. Derivatives
3. The Dataset Class

---

## Video: 1D Tensors - Basics and Operations

### What is a 1D Tensor?
- **0D tensor**: Just a number
- **1D tensor**: An array of numbers (vector, row in a database, time series)
- A tensor contains elements of a single data type

### Tensor Types
| Data Type | Tensor Type |
|----------|-------------|
| Real numbers (float) | floatTensor, doubleTensor |
| Unsigned integers (8-bit images) | byteTensor |
| Integers | intTensor, longTensor |

### Creating a Tensor
```python
import torch
# Create from Python list
tensor = torch.tensor([7, 4, 3, 2, 6])
```

### Finding Tensor Information
```python
tensor.dtype    # Data type stored in tensor
tensor.type()  # Type of tensor
tensor.size()   # Number of elements
tensor.ndim     # Number of dimensions (rank)
```

### Creating Specific Types
```python
# Explicitly create float tensor
tensor = torch.FloatTensor([1.0, 2.0, 3.0])

# Specify dtype in constructor
tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
```

### Converting Tensor Types
```python
# Convert long tensor to float
tensor = tensor.type(torch.FloatTensor)
```

### Reshaping Tensors (1D to 2D)
```python
# Using view method
tensor_2d = tensor.view(5, 1)  # 5 rows, 1 column
tensor_2d = tensor.view(-1, 1)  # Let PyTorch infer rows
```

### Converting Between NumPy and PyTorch
```python
import numpy as np

# NumPy to PyTorch
torch_tensor = torch.from_numpy(numpy_array)

# PyTorch to NumPy
numpy_array = torch_tensor.numpy()

# Important: Changes to one affect the other (they share memory)
```

### Converting Pandas to Tensor
```python
import pandas as pd
numpy_array = pandas_series.values
torch_tensor = torch.from_numpy(numpy_array)
```

### Converting Tensor to List
```python
python_list = tensor.tolist()
```

### Accessing Individual Values
```python
value = tensor[0].item()  # Returns Python number
```

---

## Video: 1D Tensor Operations

### Indexing and Slicing
```python
tensor = torch.tensor([1, 2, 3, 4, 5])

# Change first element
tensor[0] = 100

# Slice (1 to 3, not including 3)
slice_tensor = tensor[1:3]
```

### Vector Addition
```python
U = torch.tensor([1, 2])
V = torch.tensor([3, 4])
Z = U + V  # Result: [4, 6]
```

### Scalar Multiplication
```python
U = torch.tensor([1, 2])
Z = 2 * U  # Result: [2, 4]
```

### Hadamard Product (Element-wise)
```python
U = torch.tensor([1, 2])
V = torch.tensor([3, 4])
Z = U * V  # Result: [3, 8]
```

### Dot Product
```python
U = torch.tensor([1, 2])
V = torch.tensor([3, 4])
Z = torch.dot(U, V)  # Result: 11 (1*3 + 2*4)
```

### Broadcasting
```python
# Add scalar to tensor
tensor = torch.tensor([1, 2, 3])
result = tensor + 5  # Result: [6, 7, 8]
```

### Universal Functions
```python
# Mean/average
tensor.mean()

# Maximum value
tensor.max()

# Apply function to all elements
torch.sin(tensor)
torch.cos(tensor)
```

### Useful Functions
```python
# Linspace - evenly spaced numbers
torch.linspace(0, 2*np.pi, 100)  # 100 samples from 0 to 2π
```

### Plotting with PyTorch
```python
import matplotlib.pyplot as plt

x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)

plt.plot(x.numpy(), y.numpy())
plt.show()
```

---

## Video: 2D Tensors

### Examples of 2D Tensors
- **Database**: Each row = sample, each column = feature/attribute
- **Grayscale images**: 2D grid of intensity values (0-255)
  - 0 = black, 255 = white

### 3D Tensors
- **Color images**: Combination of 3 color channels (Blue, Green, Red)
- Each color channel is a 2D tensor
- Together they form a 3D tensor

### Creating 2D Tensors
```python
# From nested lists (each nested list = one row)
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
tensor = torch.tensor(A)
```

### 2D Tensor Attributes
```python
tensor.ndim        # Number of dimensions (rank)
tensor.shape       # Number of rows and columns
tensor.size()      # Shape of tensor
tensor.numel()    # Total number of elements
```

- Shape convention: `[rows, columns]`
- Axis 0: vertical axis (rows)
- Axis 1: horizontal axis (columns)

---

## Video: 2D Tensor Indexing and Slicing

### Indexing 2D Tensors
```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access element at row 1, column 2 (0-indexed)
value = tensor[1][2]  # Result: 6

# Or using comma notation
value = tensor[1, 2]   # Result: 6
```

### Slicing 2D Tensors
```python
# First row, first two columns
slice = tensor[0, 0:2]  # Result: [1, 2]

# Last two rows, last column
slice = tensor[-2:, -1]
```

---

## Video: 2D Tensor Operations

### Tensor Addition
```python
X = torch.tensor([[1, 2], [3, 4]])
Y = torch.tensor([[5, 6], [7, 8]])
Z = X + Y  # Element-wise addition
```

### Scalar Multiplication
```python
Y = torch.tensor([[1, 2], [3, 4]])
Z = 2 * Y  # Each element multiplied by 2
```

### Hadamard Product (Element-wise)
```python
X = torch.tensor([[1, 2], [3, 4]])
Y = torch.tensor([[5, 6], [7, 8]])
Z = X * Y  # Element-wise multiplication
```

### Matrix Multiplication
```python
# Important: columns of A must equal rows of B
A = torch.tensor([[1, 2], [3, 4]])  # 2x2
B = torch.tensor([[1, 0], [0, 1]])  # 2x2
C = torch.mm(A, B)  # Matrix multiplication
# Or: C = A @ B
```

**Matrix multiplication rule**: `(m×n) × (n×p) = (m×p)`

---

## Video: Derivatives in PyTorch

### Why Derivatives?
- Derivatives are used for generating parameters in neural networks
- Essential for training neural networks via gradient descent

### Simple Derivatives

**Example: y = x²**

```python
import torch

# Create tensor x with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)

# Define function y = x²
y = x ** 2

# Calculate derivative
y.backward()

# Access the gradient
print(x.grad)  # Result: 4.0
```

**How it works:**
1. Create tensor `x` with `requires_grad=True`
2. Define `y` as a function of `x`
3. Call `backward()` on `y` to calculate derivatives
4. Access gradient via `x.grad`

### Tensor Attributes

| Attribute | Description |
|-----------|-------------|
| `.data` | Holds the data of the tensor |
| `.grad` | Holds the gradient/derivative value |
| `.grad_fn` | Points to node in backwards graph |
| `.is_leaf` | Whether tensor is a leaf in the graph |
| `.requires_grad` | Whether gradient will be calculated |

### Backwards Graph
- PyTorch creates a backwards graph internally
- Tensors and backwards functions are nodes in the graph
- If `is_leaf=True`, PyTorch won't evaluate its derivative
- The `backward()` function calculates derivatives and evaluates at the point

### Example 2: z = x² + 3x

```python
x = torch.tensor(2.0, requires_grad=True)
z = x ** 2 + 3 * x

z.backward()
print(x.grad)  # Result: 7.0 (derivative: 2x + 3 = 4 + 3)
```

---

## Video: Partial Derivatives

### Partial Derivatives with Multiple Variables

**Example: f = u + v² + 2uv**

```python
# Define variables with initial values
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)

# Define function f = u + v² + 2uv
f = u + v**2 + 2*u*v

# Calculate partial derivatives
f.backward()

# Access partial derivatives
print(u.grad)  # df/du = v + 2u = 2 + 2 = 4
print(v.grad)  # df/dv = 2v + 2u = 4 + 2 = 6
```

**Mathematical rules:**
- ∂f/∂u: Treat `v` as constant, differentiate w.r.t `u`
- ∂f/∂v: Treat `u` as constant, differentiate w.r.t `v`

### Summary
- Use `requires_grad=True` to enable gradient calculation
- Call `.backward()` on the result tensor
- Access gradient via `.grad` attribute
- Works for both simple and partial derivatives

---

## Video: Custom Dataset Class

### Why Custom Datasets?
- For organizing and loading data for neural networks
- Makes it easier to work with large datasets

### Creating a Custom Dataset Class

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, length=100):
        self.x = torch.zeros(length, 2)
        self.y = torch.zeros(length, 1)
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

### Key Methods

| Method | Description |
|--------|-------------|
| `__init__` | Initialize data (features x, targets y) |
| `__len__` | Return number of samples |
| `__getitem__` | Return a single sample by index |

### Using the Dataset

```python
# Create dataset object
dataset = ToyDataset(length=100)

# Get length
len(dataset)  # Returns 100

# Access single sample
sample = dataset[0]  # Returns (x_tensor, y_tensor)

# Iterate over dataset
for x, y in dataset:
    print(x, y)
```

---

## Video: Dataset Transforms

### Why Transforms?
- Transform data (normalize, standardize, etc.)
- Apply preprocessing to tensors

### Creating a Custom Transform

```python
class AddMult:
    def __init__(self, add_x=0, mul_y=1):
        self.add_x = add_x
        self.mul_y = mul_y
    
    def __call__(self, sample):
        x, y = sample
        return x + self.add_x, y * self.mul_y
```

### Applying Transforms

**Method 1: Direct application**
```python
transform = AddMult(add_x=1, mul_y=2)
sample = dataset[0]
sample = transform(sample)
```

**Method 2: Automatic via Dataset constructor**
```python
class ToyDataset(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = torch.zeros(length, 2)
        self.y = torch.zeros(length, 1)
        self.length = length
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

---

## Video: Composing Transforms

### Multiple Transforms

```python
class Add:
    def __init__(self, add=0):
        self.add = add
    
    def __call__(self, sample):
        x, y = sample
        return x + self.add, y

class Mult:
    def __init__(self, mul=1):
        self.mul = mul
    
    def __call__(self, sample):
        x, y = sample
        return x * self.mul, y
```

### Using Compose

```python
from torchvision import transforms

# Compose multiple transforms
composed = transforms.Compose([
    Add(add=1),
    Mult(mul=2)
])

# Apply to sample
sample = composed(sample)
```

Or pass to dataset constructor:
```python
dataset = ToyDataset(length=100, transform=composed)
```

Each sample retrieved will have transforms applied in order.

---

## Video: Image Dataset

### Fashion-MNIST Dataset
- 60,000 28x28 grayscale images of clothing
- 10 label classes
- CSV file contains: class label, image filename

### Loading CSV Data
```python
import pandas as pd

# Load CSV with labels
data_df = pd.read_csv('labels.csv')

# View first few rows
data_df.head()

# Get image name and class
image_name = data_df.iloc[index, 1]
label = data_df.iloc[index, 0]
```

### Building Custom Image Dataset

```python
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.data_names.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(img_path)
        
        # Get label
        label = self.data_names.iloc[idx, 0]
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

---

## Video: TorchVision Transforms

### Built-in Transforms
```python
from torchvision import transforms

# Crop image
crop_transform = transforms.CenterCrop(20)

# Convert to tensor
to_tensor = transforms.ToTensor()

# Compose transforms
composed_transform = transforms.Compose([
    transforms.CenterCrop(20),
    transforms.ToTensor()
])
```

### Common Transforms
| Transform | Description |
|-----------|-------------|
| `ToTensor()` | Convert PIL Image to tensor |
| `CenterCrop(size)` | Crop center of image |
| `Normalize(mean, std)` | Normalize tensor |
| `RandomHorizontalFlip()` | Random flip |
| `Resize(size)` | Resize image |

---

## Video: TorchVision Datasets

### Using Pre-built Datasets
```python
from torchvision import datasets

# MNIST dataset
dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
```

### Common Parameters
| Parameter | Description |
|-----------|-------------|
| `root` | Root directory of dataset |
| `train` | True for training, False for test |
| `download` | Download if not present |
| `transform` | Transform to apply |

### Other Built-in Datasets
- `datasets.MNIST`
- `datasets.FashionMNIST`
- `datasets.CIFAR10`
- `datasets.CIFAR100`
- `datasets.ImageFolder`

---

# Module 1 Summary

## Key Concepts Learned
1. **Tensors**: 1D and 2D data structures in PyTorch
2. **Tensor Operations**: Addition, multiplication, dot product, matrix multiplication
3. **Derivatives**: Using `requires_grad`, `backward()`, `.grad`
4. **Custom Datasets**: Creating dataset classes with `__init__`, `__len__`, `__getitem__`
5. **Transforms**: Custom transforms and `transforms.Compose`
6. **Image Datasets**: Loading images, using TorchVision transforms and datasets
