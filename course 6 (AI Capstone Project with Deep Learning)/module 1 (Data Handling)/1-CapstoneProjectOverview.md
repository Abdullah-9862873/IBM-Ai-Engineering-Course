# Capstone Project Overview

Welcome to the AI Capstone Project with Deep Learning. In this advanced course, you will apply deep learning techniques to solve a real-world geospatial image classification problem. Using Keras and PyTorch, you will develop, train, and evaluate convolutional neural networks (CNNs) and vision transformers, gaining hands-on experience in model optimization, performance analysis, and technical reporting.

## Prerequisites

Before starting this capstone project, learners should have completed the following courses from the [IBM AI Engineering Professional Certificate (PC)](https://www.ibm.com/training/professional-certificate/ai-engineering) or the [IBM Deep Learning with PyTorch, Keras and Tensorflow PC](https://www.ibm.com/training/professional-certificate/deep-learning). These courses build the foundational skills necessary for working with deep learning frameworks, designing neural network architectures, and evaluating model performance—essential for tackling the real-world challenges in this capstone:

- Machine Learning with Python
- Introduction to Deep Learning & Neural Networks with Keras
- Deep Learning with Keras and Tensorflow
- Introduction to Neural Networks and PyTorch
- Deep Learning with PyTorch

Completing these courses will equip you with the necessary skills in CNNs, transfer learning, model evaluation metrics, and deep learning frameworks, critical for success in this capstone.

## Course Objectives

After completing this capstone course, you will have:

- Hands-on experience building deep learning models using Keras and PyTorch to solve real-world image classification problems
- Expertise in designing and implementing a complete deep learning pipeline, including data loading, augmentation, and model validation
- Practical skills in applying CNNs and vision transformers to domain-specific challenges like geospatial land classification
- The ability to communicate project outcomes effectively through model evaluation

## Project Scenario

You are an AI Engineer working at a fertilizer company tasked with building a land classification system for agricultural applications. Using satellite imagery, your goal is to develop models that can accurately classify different types of terrain (e.g., crops, forests, water bodies). This project will involve:

- **Data preprocessing**: Loading and augmenting geospatial image datasets
- **Model development**: Designing CNNs and vision transformers in Keras and PyTorch
- **Transfer learning**: Fine-tuning pre-trained models for improved accuracy
- **Performance evaluation**: Comparing models using metrics like accuracy, F1-score, and AU-ROC
- **Technical reporting**: Documenting your methodology, results, and insights in a final project report

## Capstone Project: Phases and Tasks

This project is structured into four modules, each focusing on a critical aspect of deep learning development.

### Module 1: Data Handling

**Objective**: Learn efficient data loading and augmentation techniques for image datasets

**Tasks**:
- Implement memory-based vs. generator-based data loading
- Apply data augmentation using Keras and PyTorch
- Build a custom geospatial data loader for model training

### Module 2: Convolutional Neural Network (CNN) Development

**Objective**: Design and train CNN models for agricultural land classification

**Tasks**:
- Develop Keras-based and PyTorch-based CNN models
- Evaluate performance using accuracy, precision, and recall
- Compare the strengths of each framework

### Module 3: CNN - Vision Transformer Integration

**Objective**: Implement vision transformers and apply transfer learning

**Tasks**:
- Fine-tune pre-trained transformer models in Keras and PyTorch
- Compare vision transformer performance against CNNs

### Module 4: Final Project and Course Wrap-Up

**Objective**: Consolidate your work into a comprehensive solution

**Tasks**:
- Conduct a comparative analysis of CNN and vision transformer models
- Submit a final project report with performance insights
- Reflect on key learnings and future applications

## Final Outcome

By the end of this capstone, you will have:

- Built and trained CNN and vision transformer models for geospatial classification
- Applied transfer learning to enhance model performance
- Evaluated models using standard metrics (accuracy, F1-score, AU-ROC)
- Produced a professional report summarizing your methodology and results

## Grading and Deliverables

Your work will be assessed based on:

- Functional deep learning models (CNNs and vision transformers)
- Proper data handling (loading, augmentation, preprocessing)
- Model evaluation (quantitative metrics and comparative analysis)

In Module 4, you will submit:

- Jupyter notebooks demonstrating lab outputs and model training, evaluated for clarity, insights, and technical rigor

**Note**: To get the best experience, you might find it faster to download the lab notebooks and run them on your own machine. Training and testing deep learning models can require significant computing resources, and the shared online lab environment in this course has limited capacity, which may lead to slower performance.

---

## Video: Capstone Project Overview

Hello, and welcome to the Capstone Project overview video. In this video, you'll get a complete picture of the project structure, your role as a learner, the tasks you'll undertake, and the deliverables you'll create.

This Capstone Project offers a practical, end-to-end experience in applying deep learning techniques to geospatial image classification using satellite data. This project, **Geospatial Land Coverage Classification for a Fertilizer Company**, is designed to enable you to apply key concepts learned.

You will be performing the project tasks in Jupyter Labs using Keras and PyTorch. You'll then download your completed labs and submit them for evaluation. You will use the learned concepts to solve a domain-specific challenge in satellite image analysis and classification.

The project focuses on:

- Developing robust preprocessing pipelines for large image datasets
- Building and comparing deep learning models for geospatial classification
- Evaluating model performance using rigorous classification metrics
- Applying augmentation and transfer learning to improve model generalization

### Your Role

You will take on the role of an AI engineer at **NutriSphere Agritech**, a fertilizer company, aiming to identify agricultural land in a new geographical region. Your goal is to classify satellite imagery to determine land coverage types accurately. This will help your company grow by expanding into untapped markets and predicting future sales in the new territory. As an AI engineer, your work bridges technical remote sensing and commercial decision-making. This would help the company form the strategy for sales and field teams.

### Project Dataset

You will be provided with a dataset consisting of around 6,000, 64x64 pixel satellite images. You can download the dataset via the links provided in the labs to train and test the models. The dataset consists of two classes of land usage: **agricultural** and **non-agricultural**.

### Project Tasks Overview

You will design and train AI models with two fundamentally different architectures, CNNs and a CNN vision transformer or ViT hybrid model for predicting the agricultural land coverage in the new territory.

**Phase 1: Data Handling and Preprocessing**
- You'll build an image-loading pipeline using either memory-based or generator-based strategies, depending on dataset size and available computational resources
- To increase model generalization, you'll apply data augmentation techniques such as flipping, rotation, scaling, and contrast adjustment
- You'll also compare how Keras and PyTorch handle image data loading and transformation

**Phase 2: CNN Model Development**
- You'll experiment with Keras, adjusting model depth, batch size, and learning rates using its high-level API
- Then you'll implement CNNs and PyTorch, giving you full control over parameters including layers, forward passes, and training loops

**Phase 3: Vision Transformers**
- Using transfer learning, you'll use pre-trained CNN models to fine-tune transformer-based architectures
- You'll implement and train ViTs in both Keras and PyTorch and experiment with hybrid models that combine CNNs and transformer components

**Phase 4: Model Evaluation and Comparison**
- You'll assess model performance using a range of metrics—accuracy, precision, recall, F1 score, and AU-ROC (Area Under the Receiver Operating Characteristic curve)
- This will help you compare models systematically and highlight the strengths and tradeoffs of each approach

### Recommended Timeline

To complete the project successfully within the prescribed four-week schedule, you were advised to follow this timeline:

- **Week 1** – Prepare the dataset by implementing data-loading strategies and applying image augmentation techniques using Keras and PyTorch
- **Week 2** – Develop and train CNN models in both Keras and PyTorch, adjusting model parameters and comparing training behavior
- **Week 3** – Implement vision transformers and hybrid models, leveraging transfer learning and exploring architecture variations. Evaluate all models using metrics such as accuracy, precision, recall, F1 score, and AU-ROC
- **Week 4** – Finalize your Jupyter Lab submission

### Evaluation Criteria

You will be evaluated based on the following criteria:

- Efficiency and precision of the data loading and preprocessing pipeline
- Performance and correctness of CNN, ViT, and hybrid model implementations
- Depth and accuracy of model evaluation using classification metrics
- Quality and clarity of the submitted tasks and questions answered in the submitted Jupyter Labs

### Learning Outcomes

After completing the project, you will achieve:

- Enhanced ability to build and evaluate deep learning models for image classification tasks
- Practical experience in implementing CNNs and vision transformers using both Keras and PyTorch
- Improved skills in preprocessing and augmenting large-scale computer vision datasets
- Application of rigorous metrics to assess model performance on geospatial data
- A portfolio-worthy project demonstrating your capability to solve real-world challenges using advanced deep learning techniques

Now that you understand the scope and expectations, let's begin your capstone journey into geospatial deep learning.