# Introduction to Machine Learning

## What is Machine Learning?
-> Machine Learning (ML) is a subset of AI that teaches computers to learn from data, identify patterns, and make decisions without explicit human instructions.

### Key Differences:
- **AI** = Broad field making computers appear intelligent (simulates human cognitive abilities)
- **ML** = Subset of AI that uses algorithms and requires feature engineering
- **Deep Learning** = Uses many-layered neural networks to automatically extract features from complex, unstructured data

---

## How Machine Learning Learns

### Types of Learning:

1. **Supervised Learning** 
   -> Trains on **labeled data** to make predictions on new data
   -> Example: Learning to identify spam emails from labeled examples

2. **Unsupervised Learning**
   -> Works **without labels** by finding patterns in data
   -> Example: Grouping similar customers without predefined categories

3. **Semi-Supervised Learning**
   -> Trains on **small labeled data**, then adds self-generated labels with high confidence
   -> Example: Training a model with 10% labeled images, then labeling the rest itself

4. **Reinforcement Learning**
   -> Agent learns by **interacting with environment** and receiving feedback
   -> Example: AI learning to play chess by winning/losing

---

## Machine Learning Techniques

| Technique | Purpose | Example |
|-----------|---------|---------|
| **Classification** | Predict class/category | Benign or Malignant cell |
| **Regression** | Predict continuous values | House price, CO2 emissions |
| **Clustering** | Group similar cases | Customer segmentation |
| **Association** | Find items that co-occur | Grocery items bought together |
| **Anomaly Detection** | Discover abnormal cases | Credit card fraud detection |
| **Sequence Mining** | Predict next event | Website clickstream analysis |
| **Dimension Reduction** | Reduce data/features | Compress image data |
| **Recommendation Systems** | Recommend items to users | Netflix movie suggestions |

### Classification vs Regression:
- **Classification** -> Predicts category (Benign/Malignant, Yes/No)
- **Regression** -> Predicts continuous number (Price: $250,000)

### Clustering Example:
-> Groups data points into colors (Green, Red, Blue)
-> Black points = noise/uncategorized

---

## Real-World Applications

### 1. Healthcare - Cancer Detection
-> Input: Cell sample with characteristics (clump thickness, cell size, etc.)
-> Output: Benign or Malignant prediction
-> Benefit: Early detection saves lives

### 2. Finance - Loan Approval
-> Banks predict **default probability** for each applicant
-> Decision: Approve or deny loan based on prediction

### 3. Business - Customer Churn
-> Use demographic data to predict if customer will unsubscribe
-> Helps companies retain customers

### 4. E-commerce - Recommendations
-> Amazon recommends products
-> Netflix recommends movies
-> Based on similar user preferences

### 5. Computer Vision - Image Recognition
-> Old way: Manually create rules (ears, tail, legs) - FAILED
-> ML way: Algorithm learns distinguishing features automatically
-> Example: Differentiating cats from dogs in images

### 6. Other Applications:
-> Virtual assistants (chatbots)
-> Face recognition (phone unlock)
-> Playing games (chess AI)

---

## Summary

**Machine Learning = Subset of AI**
-> Uses algorithms + feature engineering

**Learning Types:**
-> Supervised, Unsupervised, Semi-Supervised, Reinforcement

**Techniques:**
-> Classification, Regression, Clustering, Association, Anomaly Detection, Sequence Mining, Dimension Reduction, Recommendation Systems

**Impact:**
-> Healthcare, Finance, Business, Gaming, and everyday technology

---

# Machine Learning Model Lifecycle

## What is ML Model Lifecycle?
-> The complete process from defining a problem to deploying a working model

## 5 Stages of ML Lifecycle:

### 1. Problem Definition
-> Define the problem or state the situation
-> Ask: What problem are we solving?

### 2. Data Collection
-> Gather data from various sources
-> Sources: Databases, APIs, sensors, user inputs

### 3. Data Preparation (ETL Process)
-> **E**xtract: Collect data from sources
-> **T**ransform: Clean and format data
-> **L**oad: Store data in a single place
-> Steps: Cleaning, Transforming, Storing data

### 4. Model Development & Evaluation
-> Build ML model using the prepared data
-> Train and test the model
-> Evaluate performance

### 5. Model Deployment
-> Deploy model to production
-> Make it available for real-world use

## Important Note:
-> The lifecycle is **ITERATIVE** (goes back and forth)
-> If problems found in production -> go back to data collection or problem definition
-> Then repeat the process

## Example:
-> Beauty product shopping app
-> Problem: Recommend products to users
-> Collect user data -> Prepare data -> Build model -> Deploy -> Monitor -> Repeat if needed

---

# A Day in the Life of a ML Engineer

## Project Example: Beauty Product Recommender
-> Goal: Recommend similar products based on customer's purchase history
-> End-user pain point: "I want product recommendations based on my purchase history to improve my skin health"

## Importance of Each Process:

### 1. Problem Definition (IMPORTANT!)
-> Must align ML solution with client's needs
-> Understand the end-user's pain point
-> Question: What problem are we solving?

### 2. Data Collection (Time-Consuming)
-> Determine what data company has
-> Sources of data:
   - **User Data**: Demographics, purchase history, transactions
   - **Product Data**: Inventory, ingredients, ratings, popularity
   - **Behavior Data**: Saved products, liked products, search history, visited products
-> Transform data: Wrangle, aggregate, join, merge, map to central source

### 3. Data Preparation (MOST Time-Consuming!)
-> Data often contains errors, wrong formats, missing values
-> Steps:
   - Clean data (filter irrelevant data)
   - Remove extreme values (outliers)
   - Handle missing values (remove or generate)
   - Format data properly (dates, strings)
-> **Feature Engineering**: Create new features
   - Average duration between transactions
   - Products user buys most
   - Skin issues each product targets
-> **Exploratory Data Analysis (EDA)**:
   - Create plots to find patterns
   - Validate data with subject matter experts
   - Correlation analysis (important features)
-> Split data for training/testing (use most recent transaction as test set)

### 4. Model Development
-> Use existing frameworks (don't build from scratch)
-> Techniques for recommendation:
   - **Content-Based Filtering**: Find similarity between products based on content
      -> Example: User bought moisturizing cleanser → recommend moisturizing moisturizer
   - **Collaborative Filtering**: Find similarity between users based on ratings
      -> Group users by age, region, skin type
      -> Recommend products that similar users rated highly
-> Final model = combination of both techniques

### 5. Model Evaluation
-> Tune model using test data
-> Test with real users (pilot group)
-> Collect feedback:
   - User ratings on recommendations
   - Click-through rate
   - Number of purchases from recommendations

### 6. Model Deployment & Monitoring
-> Deploy to app/website
-> Track performance continuously
-> Retrain model with new data when needed

## Key Points:
-> All steps are important for success
-> **Data Collection & Preparation = MOST Time-Consuming** (60-80% of time!)
-> After deployment: Continuous monitoring and improvement needed
-> Lifecycle is iterative - always improving!

---

# Data Scientist vs AI Engineer

## Why the Split?
-> Generative AI breakthroughs created a new field: **AI Engineering**
-> Data Scientists focus on traditional ML, AI Engineers focus on Generative AI

## 4 Key Differences:

### 1. Use Cases

| Data Scientist | AI Engineer |
|----------------|-------------|
| **Data Storyteller** | **AI System Builder** |
| Uses descriptive analytics (describe the past) | Uses prescriptive analytics (best course of action) |
| **Descriptive**: EDA, clustering (customer segmentation) | **Prescriptive**: Decision optimization, recommendation engines |
| **Predictive**: ML models predict future | **Generative**: Create new content |

-> **Examples**:
   - Data Scientist: Regression (predict temperature), Classification (predict success/failure)
   - AI Engineer: Decision optimization, Marketing campaign suggestions, Chatbots, Coding assistants

### 2. Data

| Data Scientist | AI Engineer |
|----------------|-------------|
| **Structured Data** (tabular) | **Unstructured Data** (text, images, videos, audio) |
| Hundreds to thousands of rows | Billions to trillions of tokens |
| Requires lots of cleaning | Less cleaning needed |
| Example: Customer tables | Example: LLM training on text |

### 3. Models

| Data Scientist | AI Engineer |
|----------------|-------------|
| Hundreds of different models | Mainly **Foundation Models** |
| Narrow scope (specialized) | Wide scope (generalizable) |
| Small size (few parameters) | Billions of parameters |
| Train: seconds to hours | Train: weeks to months |
| Less compute power | Hundreds to thousands of GPUs |

### 4. Process

**Data Science Process:**
-> Use Case → Pick right data → Prepare data → Train/validate model (feature engineering, cross-validation) → Deploy for prediction

**Generative AI Process:**
-> Use Case → Skip to pre-trained model → Prompt Engineering → Build AI system (chaining, PEFT, RAG, agents) → Embed in workflow

## Key Concepts:

### AI Democratization
-> Making AI accessible to everyone
-> Foundation models published to open-source (Hugging Face)
-> AI Engineers use pre-trained models instead of training from scratch

### Prompt Engineering
-> Using natural language instructions to make foundation models do tasks
-> No need to retrain model

### Building Blocks for AI Systems:
- Chaining prompts together
- **PEFT** (Parameter-Efficient Fine-Tuning): Fine-tune on specific data
- **RAG** (Retrieval-Augmented Generation): Ground answers in truth
- Autonomous agents: Reason through complex multi-step problems

## Summary:
-> Both fields overlap but work differently
-> Data Scientist = Storytelling with data
-> AI Engineer = Building generative AI systems
-> Both fields evolving fast with new research every day!

---

# Tools for Machine Learning

## What is Data?
-> Collection of raw facts, figures, or information
-> Used to draw insights, inform decisions, and fuel advanced technologies
-> **Central to every ML algorithm** - source of information for pattern discovery and predictions

## What are ML Tools?
-> Provide functionalities for ML pipelines
-> Include modules for: Data preprocessing, Building models, Evaluating, Optimizing, Implementing
-> Simplify complex tasks like handling big data, statistical analysis, predictions

## Popular ML Programming Languages:

| Language | Use Case |
|----------|----------|
| **Python** | Most popular - extensive libraries for data analysis and ML |
| **R** | Statistical learning, data exploration, ML libraries |
| **Julia** | High-performance, parallel/distributed computing (research) |
| **Scala** | Scalable, big data processing, ML pipelines |
| **Java** | Scalable ML apps in production |
| **JavaScript** | ML in web browsers (client-side) |

---

## Tools by Category:

### 1. Data Processing & Analytics

| Tool | Purpose |
|------|---------|
| **PostgreSQL** | Object-relational database (SQL) for storing/retrieving data |
| **Hadoop** | Open-source, scalable disk-based storage and batch-processing |
| **Spark** | Distributed, in-memory data processing (faster than Hadoop) |
| **Apache Kafka** | Distributed streaming platform for real-time analytics |
| **Pandas** | Data manipulation and analysis (DataFrame = tabular data) |
| **NumPy** | Mathematical functions, linear algebra, GPU computing |

### 2. Data Visualization

| Tool | Purpose |
|------|---------|
| **Matplotlib** | Foundation library for customizable plots and visualizations |
| **Seaborn** | High-level interface for attractive statistical graphics (based on Matplotlib) |
| **ggplot2** | R package for building graphics in layers |
| **Tableau** | Business intelligence tool for interactive dashboards |

### 3. Machine Learning

| Tool | Purpose |
|------|---------|
| **NumPy** | Foundational support for numerical computations |
| **Pandas** | Data analysis, cleaning, preparation |
| **SciPy** | Scientific computing (optimization, integration, regression) |
| **Scikit-learn** | Classical ML models - classification, regression, clustering, dimensionality reduction |

### 4. Deep Learning

| Tool | Purpose |
|------|---------|
| **TensorFlow** | Numerical computing, large-scale ML |
| **Keras** | Easy-to-use library for neural networks |
| **Theano** | Efficient mathematical expressions |
| **PyTorch** | Deep learning, computer vision, NLP, experimentation |

### 5. Computer Vision

| Tool | Purpose |
|------|---------|
| **OpenCV** | Real-time computer vision - object detection, image classification, AR |
| **Scikit-Image** | Image processing algorithms - filters, segmentation, feature extraction |
| **TorchVision** | PyTorch computer vision - datasets, pre-trained models, image transforms |

### 6. NLP (Natural Language Processing)

| Tool | Purpose |
|------|---------|
| **NLTK** | Text processing, tokenization, stemming |
| **TextBlob** | Part-of-speech tagging, noun-phrase extraction, sentiment analysis |
| **Stanza** | Stanford NLP - accurate models for POS tagging, NER, dependency parsing |

### 7. Generative AI

| Tool | Purpose |
|------|---------|
| **Hugging Face Transformers** | Transformer models for text generation, translation, sentiment analysis |
| **ChatGPT** | Language model for text generation, chatbots |
| **DALL-E** | Generate images from text descriptions |
| **PyTorch** | Create generative models (GANs, Transformers) |

---

## Summary:
-> Data is essential for ML - it's the fuel for algorithms
-> Python is the most popular ML language
-> Tools simplify ML pipelines: Processing → Visualization → ML → DL → CV → NLP → GenAI

---

# Scikit-Learn Machine Learning Ecosystem

## What is ML Ecosystem?
-> Interconnected tools, frameworks, libraries, platforms, and processes
-> Supports developing, deploying, and managing ML models

## Use Case: Music Streaming App
-> You created an app where users can play/download music, share files, create playlists
-> Goal: Increase user base by understanding listening habits
-> Data collected: Songs played, listening duration, songs skipped
-> Problem: Data has inconsistencies, missing values, outliers
-> Solution: Use ML tools to clean data, train models, and make predictions

## Python ML Ecosystem Stack:

```
NumPy → Pandas → SciPy → Matplotlib → Scikit-learn
   ↓        ↓        ↓          ↓            ↓
Foundation  Data   Scientific  Visualization  ML Models
```

| Library | Role |
|---------|------|
| **NumPy** | Foundational - efficient numerical computations on multidimensional arrays |
| **Pandas** | Data analysis, visualization, cleaning, preparing (uses DataFrames) |
| **SciPy** | Scientific computing - optimization, integration, linear regression |
| **Matplotlib** | Visualization - extensive, customizable plots |
| **Scikit-learn** | Building classical ML models |

---

## What is Scikit-Learn?

### Overview:
-> Free ML library for Python
-> Wide selection of classification, regression, clustering, dimensionality reduction algorithms
-> Works with NumPy and SciPy
-> Excellent documentation + large community support

### Key Features:
-> Easy to use (few lines of code)
-> Constantly evolving (thousands of contributors)

---

## Tasks in Scikit-Learn Pipeline:

| Step | Description |
|------|-------------|
| **Data Preprocessing** | Cleaning, scaling, feature selection, feature extraction |
| **Train/Test Split** | Split data for training and testing |
| **Model Setup** | Initialize model and parameters |
| **Model Fitting** | Train model on training data |
| **Hyperparameter Tuning** | Optimize with cross-validation |
| **Prediction** | Generate predictions on test data |
| **Evaluation** | Measure accuracy (confusion matrix, etc.) |
| **Export** | Save model to file (pickle) for production |

---

## Basic Workflow Example:

### Use Case Context:
-> Imagine you have user listening data (X) and want to predict if users will upgrade to premium (Y)
-> X = features (songs played, duration, skipped)
-> Y = target (0 = free user, 1 = premium user)

### 1. Prepare Data (Scaling)
```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Explanation:**
-> Different features have different ranges (duration: 0-300 mins, songs: 0-1000)
-> This makes model biased toward larger numbers
-> StandardScaler normalizes all features to same scale (mean=0, std=1)
-> fit_transform: Learns the scale from data, then transforms it

### 2. Split Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
```
**Explanation:**
-> Need to test model on data it hasn't seen before
-> 33% of data goes to testing, 67% for training
-> Randomly splits so both sets have similar patterns

### 3. Create Model
```python
from sklearn.svm import SVC
clf = SVC(gamma=0.01, C=1.0)
```
**Explanation:**
-> SVC = Support Vector Classification (algorithm for classification)
-> clf = classifier (the model we'll train)
-> gamma and C = hyperparameters (settings we can tune)
-> These control how the model learns

### 4. Train Model
```python
clf.fit(X_train, Y_train)
```
**Explanation:**
-> fit() = teach the model
-> Model learns patterns in training data
-> Learns relationship between X_train (features) and Y_train (answers)

### 5. Make Predictions
```python
predictions = clf.predict(X_test)
```
**Explanation:**
-> Use trained model to predict on new data
-> Input: X_test (features from test set)
-> Output: predictions (predicted labels: 0 or 1)

### 6. Evaluate Model
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predictions)
```
**Explanation:**
-> Compare predicted vs actual labels
-> Shows: True Positives, True Negatives, False Positives, False Negatives
-> Helps measure accuracy

### 7. Save Model
```python
import pickle
pickle.dump(clf, open('model.pkl', 'wb'))
```
**Explanation:**
-> Save trained model to file
-> Can load and use later without retraining
-> 'wb' = write binary mode

---

## Summary:
-> ML Ecosystem = interconnected tools for developing/deploying ML models
-> Python stack: NumPy → Pandas → SciPy → Matplotlib → Scikit-learn
-> Scikit-learn = all-in-one library for classical ML tasks
-> Pipeline: Preprocess → Split → Train → Predict → Evaluate → Export
