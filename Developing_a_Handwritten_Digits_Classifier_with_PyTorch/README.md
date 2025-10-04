# udacity-ml-fundamentals-project2
# MNIST Handwritten Digits Classifier with PyTorch

This repository contains my second project for the **Udacity Machine Learning Fundamentals Nanodegree program**. The project involves developing, training, and evaluating a neural network to classify handwritten digits from the MNIST dataset using PyTorch. The final model achieved an accuracy of over 98% on the test set.

## Project Overview

The goal of this project was to:
1.  Load and preprocess the MNIST dataset.
2.  Design a neural network architecture suitable for image classification.
3.  Train the model using PyTorch, monitoring loss and accuracy.
4.  Experiment with different optimizers, learning rates, and a learning rate scheduler to improve performance.
5.  Evaluate the trained model on the test set.
6.  Save the best-performing model.

## Key Learnings & Skills Demonstrated
-   **Neural Network Design:** Building a multi-layer perceptron (MLP) with fully connected layers, ReLU activation functions, and Dropout for regularization.
-   **PyTorch Fundamentals:**
    -   Defining a custom `nn.Module`.
    -   Utilizing `torch.utils.data.DataLoader` for efficient data handling.
    -   Implementing training and validation loops.
    -   Working with tensors and moving computations to GPU (if available).
-   **Data Preprocessing:** Applying `transforms.ToTensor()` and `transforms.Normalize()` to the MNIST image data.
-   **Model Training & Optimization:**
    -   Using loss functions like `nn.CrossEntropyLoss`.
    -   Employing optimizers such as `Adam` and `SGD` with momentum.
    -   Implementing learning rate scheduling (`StepLR`).
-   **Model Evaluation:** Calculating accuracy and loss on training, validation, and test sets.
-   **Hyperparameter Tuning:** Experimenting with different optimizers, learning rates, and number of epochs to improve model accuracy.
-   **Model Persistence:** Saving and loading trained model weights using `torch.save()` and `load_state_dict()`.

## Dataset
The project uses the well-known [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which consists of:
-   A training set of 60,000 28x28 grayscale images of handwritten digits (0-9).
-   A test set of 10,000 28x28 grayscale images.

The data was normalized using the standard MNIST mean (0.1307) and standard deviation (0.3081).

## Technologies Used
-   Python 3
-   PyTorch
-   Torchvision
-   NumPy
-   Matplotlib
-   Jupyter Notebook

## Files in this Repository
-   `MNIST_Handwritten_Digits-STARTER.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model definition, training, evaluation, and model saving.
-   `requirements.txt`: A file listing the Python dependencies required to run the notebook.
-   `mnist_classifier_best_model.pth`: The saved state dictionary of the best-performing trained model (achieving >98% accuracy).
-   `README.md`: This file.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```
    *(Replace `YOUR_USERNAME/YOUR_REPOSITORY_NAME` with your actual GitHub username and repository name)*

2.  **(Optional, but recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Notebook

1.  Ensure you have Jupyter Notebook or JupyterLab installed (`pip install notebook` or `pip install jupyterlab`).
2.  Navigate to the project directory in your terminal.
3.  Launch Jupyter Notebook/Lab:
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```
4.  Open the `MNIST_Handwritten_Digits-STARTER.ipynb` file from the Jupyter interface.
5.  You can then run the cells sequentially to see the data loading, preprocessing, model definition, training, evaluation, and saving process.

## Model Architecture & Training

The primary model architecture consists of:
-   A Flatten layer to convert 28x28 images into a 784-feature vector.
-   Two hidden fully connected (Linear) layers with ReLU activations and Dropout for regularization.
    -   `fc1`: 784 input features -> 128 output features
    -   `fc2`: 128 input features -> 64 output features
-   An output fully connected layer with 10 output features (for digits 0-9).
    -   `fc3`: 64 input features -> 10 output features

The model was initially trained using the Adam optimizer. A second version (`model_v2`) was trained using SGD with momentum and a StepLR learning rate scheduler for 20 epochs, which yielded the best performance.

## Results
The best model (`model_v2`) achieved an accuracy of **over 98%** on the 10,000 MNIST test images. Training and validation loss/accuracy were plotted over epochs to monitor performance and ensure the model was learning effectively without significant overfitting.

## Acknowledgements
-   This project is part of the Udacity Machine Learning Fundamentals Nanodegree program.
-   Guidance and support from Lamia Zain.
