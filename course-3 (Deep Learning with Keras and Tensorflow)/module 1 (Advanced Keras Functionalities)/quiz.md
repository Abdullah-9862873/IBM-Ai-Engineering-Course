# Module 1: Advanced Keras Functionalities - Quiz

## Question 1
**What is the primary advantage of the Keras Functional API over the Sequential API?**

- [ ] It is only used for pre-trained models.
- [ ] It runs faster than the Sequential API.
- [ ] It is easier to use for simple models.
- [x] It provides more flexibility and control for building complex models.

**Answer:** It provides more flexibility and control for building complex models

**Explanation:** The Functional API enables building complex models with multiple inputs/outputs, shared layers, and non-sequential data flows, which the Sequential API cannot handle.

---

## Question 2
**How do you define an input layer in the Keras Functional API?**

- [ ] `input = InputLayer(shape=(784,))`
- [x] `input = Input(shape=(784,))`
- [ ] `input = Dense(784)`
- [ ] `input = Dense(shape=(784,))`

**Answer:** `input = Input(shape=(784,))`

**Explanation:** In the Functional API, `Input()` is used to define the input layer with the specified shape. `InputLayer` is not the correct function name.

---

## Question 3
**Which of the following code snippets correctly adds a dense layer with 64 units and ReLU activation in the Functional API?**

- [ ] `x = Dense(64, activation='relu', input)`
- [ ] `x = Dense(64, input='relu')(input)`
- [ ] `x = Input(Dense(64, activation='relu'))`
- [x] `x = Dense(64, activation='relu')(input)`

**Answer:** `x = Dense(64, activation='relu')(input)`

**Explanation:** In the Functional API, layers are called as functions on the input tensor. The correct syntax is `Dense(units, activation='activation_name')(input_tensor)`.

---

## Question 4
**Which activation function is suitable for binary classification in the output layer?**

- [ ] Softmax
- [ ] Tanh
- [ ] ReLU
- [x] Sigmoid

**Answer:** Sigmoid

**Explanation:** Sigmoid outputs a value between 0 and 1, making it ideal for binary classification. Softmax is used for multi-class classification, while ReLU and Tanh are typically used in hidden layers.

---

## Question 5
**What is the first step in creating a model using the Keras Functional API?**

- [ ] Compiling the model
- [ ] Evaluating the model
- [x] Defining the input layer
- [ ] Adding the output layer

**Answer:** Defining the input layer

**Explanation:** The Functional API starts by defining the input layer using `Input()`, then layers are connected sequentially, and finally the model is created by specifying inputs and outputs.

---

## Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | Functional API Advantage | Flexibility and control for complex models |
| 2 | Input Layer Syntax | `Input(shape=(784,))` |
| 3 | Dense Layer Syntax | `Dense(64, activation='relu')(input)` |
| 4 | Activation Functions | Sigmoid for binary classification |
| 5 | Model Creation Steps | Define input layer first |

**Total Points:** 5 points

---

## Module 1 Quiz - TensorFlow 2.X and Custom Layers

## Question 1
**Which feature of TensorFlow enables fast implementation of operations without building graphs, which simplifies the debugging process and facilitates interactive programming?**

- [x] Eager execution
- [ ] High-level API
- [ ] Rich ecosystem
- [ ] Support for various platforms and devices

**Answer:** Eager execution

**Explanation:** Eager execution allows operations to execute immediately without building static computation graphs, providing immediate feedback for easier debugging and supporting interactive programming.

---

## Question 2
**Which component of the TensorFlow ecosystem is a library for training and deploying machine learning models in JavaScript environments?**

- [ ] TensorFlow Hub
- [x] TensorFlow.js
- [ ] TensorFlow Extended (TFX)
- [ ] TensorFlow Lite

**Answer:** TensorFlow.js

**Explanation:** TensorFlow.js is specifically designed for JavaScript environments, allowing ML models to run in web browsers and Node.js applications.

---

## Question 3
**Which method is used to add weights to a custom layer in Keras?**

- [ ] `Custom_weight`
- [ ] `add_parameter`
- [ ] `add_variable`
- [x] `add_weight`

**Answer:** `add_weight`

**Explanation:** The `add_weight()` method is used within the `build()` method of a custom layer to create trainable weights/parameters.

---

## Question 4
**What is the purpose of the `call` method in a custom layer?**

- [ ] To train the model
- [x] To define the forward pass logic
- [ ] To initialize weights
- [ ] To compile the model

**Answer:** To define the forward pass logic

**Explanation:** The `call()` method defines how the layer processes input data during the forward pass. It's called every time the layer is invoked.

---

## Question 5
**Which class do you need to subclass to create a custom layer in Keras?**

- [ ] `Model`
- [ ] `Dense`
- [x] `Layer`
- [ ] `Sequential`

**Answer:** `Layer`

**Explanation:** To create a custom layer, you subclass `tf.keras.layers.Layer` and implement the `__init__()`, `build()`, and `call()` methods.

---

## Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | TensorFlow Features | Eager execution |
| 2 | TensorFlow Ecosystem | TensorFlow.js |
| 3 | Custom Layers | `add_weight` |
| 4 | Custom Layers | Forward pass logic |
| 5 | Custom Layers | `Layer` class |

**Total Points:** 10 points (5 questions × 2 sections)

---

## Module 1 Graded Quiz - Advanced Keras and TensorFlow 2.X

## Question 1
**Which of the following is a correct statement about the Keras Functional API?**

- [x] It allows the creation of models with multiple inputs and outputs.
- [ ] It only supports sequential models.
- [ ] It cannot be used to create shared layers.
- [ ] It simplifies the code for simple models compared to the Sequential API.

**Answer:** It allows the creation of models with multiple inputs and outputs.

**Explanation:** The Functional API is designed for complex architectures including multiple inputs/outputs and shared layers. For simple models, the Sequential API is more straightforward.

---

## Question 2
**What method must be overridden to define the forward pass in a custom Keras layer?**

- [ ] `__init__`
- [x] `call`
- [ ] `build`
- [ ] `compile`

**Answer:** `call`

**Explanation:** The `call()` method defines the forward pass logic of a custom layer. It's called every time the layer processes input data.

---

## Question 3
**In the Keras Functional API, how do you define an input layer?**

- [ ] `input = Dense(784)`
- [ ] `input = InputLayer(shape=(784,))`
- [ ] `input = Dense(shape=(784,))`
- [x] `input = Input(shape=(784,))`

**Answer:** `input = Input(shape=(784,))`

**Explanation:** `Input()` is the correct function to define an input layer in the Functional API with the specified shape.

---

## Question 4
**Which of the following is true about creating custom layers in Keras?**

- [ ] Custom layers cannot use activation functions.
- [ ] Custom layers must always implement a compile method.
- [x] Custom layers are created by subclassing the Layer class.
- [ ] Custom layers cannot have trainable weights.

**Answer:** Custom layers are created by subclassing the Layer class.

**Explanation:** Custom layers are created by subclassing `tf.keras.layers.Layer` and implementing `__init__()`, `build()`, and `call()` methods. They can have activation functions and trainable weights.

---

## Question 5
**What is the purpose of the `build` method in a custom Keras layer?**

- [ ] To define the forward pass logic
- [x] To create and initialize the layer's weights
- [ ] To initialize the layer's attributes
- [ ] To compile the model

**Answer:** To create and initialize the layer's weights

**Explanation:** The `build()` method is called during the first call to the layer and is used to create trainable weights using `add_weight()`.

---

## Question 6
**How can you define a custom dense layer with ReLU activation to a Keras model?**

- [ ] By using the Dense class and specifying the activation function
- [ ] By defining the layer directly in the Sequential model
- [ ] By subclassing the Model class and overriding the call method
- [x] By subclassing the Layer class and implementing `__init__`, `build`, and `call` methods

**Answer:** By subclassing the Layer class and implementing `__init__`, `build`, and `call` methods

**Explanation:** Custom layers require subclassing `Layer` (not `Model`) and implementing the three key methods for initialization, weight creation, and forward pass.

---

## Question 7
**What is one key benefit of using the Functional API in Keras?**

- [ ] It allows for easy debugging by using eager execution.
- [x] It enables the creation of complex models such as multi-input and multi-output models.
- [ ] It ensures faster training of models.
- [ ] It simplifies creating models with a single input and output.

**Answer:** It enables the creation of complex models such as multi-input and multi-output models.

**Explanation:** The Functional API's main advantage is supporting complex architectures including multiple inputs/outputs, shared layers, and non-sequential data flows.

---

## Question 8
**In the following code snippet of a custom layer, what is the purpose of `add_weight`?**

```python
self.w = self.add_weight(shape=(input_shape[-1], self.units))
```

- [ ] To add a new input to the model
- [ ] To add a new layer to the model
- [ ] To compile the model
- [x] To create and initialize weights for the custom layer

**Answer:** To create and initialize weights for the custom layer

**Explanation:** `add_weight()` creates trainable weight tensors for the custom layer with specified shape and initializer.

---

## Question 9
**Which feature of TensorFlow enables the execution of operations immediately in interactive programming?**

- [ ] Rich ecosystem
- [ ] High-level APIs
- [x] Eager execution
- [ ] Scalability

**Answer:** Eager execution

**Explanation:** Eager execution allows operations to run immediately without building static computation graphs, enabling interactive programming and easier debugging.

---

## Question 10
**What is the role of the `Input` function in the Keras Functional API?**

- [ ] To define a custom layer
- [ ] To initialize the model's weights
- [ ] To compile the model
- [x] To define the input tensor for a model

**Answer:** To define the input tensor for a model

**Explanation:** `Input()` creates a placeholder tensor that defines the shape and type of input data for the model.

---

## Complete Quiz Summary

| Question | Topic | Correct Answer |
|----------|-------|----------------|
| 1 | Functional API | Multiple inputs/outputs support |
| 2 | Custom Layers | `call` method |
| 3 | Functional API | `Input(shape=(784,))` |
| 4 | Custom Layers | Subclass Layer class |
| 5 | Custom Layers | Create/initialize weights |
| 6 | Custom Layers | Subclass Layer with 3 methods |
| 7 | Functional API | Complex model creation |
| 8 | Custom Layers | Create weights |
| 9 | TensorFlow 2.X | Eager execution |
| 10 | Functional API | Define input tensor |

**Total Points:** 20 points (15 questions across all quiz sections)
