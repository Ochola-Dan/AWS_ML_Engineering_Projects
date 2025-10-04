# Landmark Classification & Tagging for Social Media

This project tackles a real-world computer vision problem: automatically identifying landmarks in user-supplied images. The goal is to build a complete pipeline, from data processing to model deployment, that can predict the location where a photo was taken, even when GPS metadata is missing.

The project demonstrates an end-to-end machine learning workflow, including data augmentation, building a custom Convolutional Neural Network (CNN), leveraging transfer learning with pre-trained models, and deploying the final model into an interactive application.

![App Screenshot](https://user-images.githubusercontent.com/10678888/159239859-1b6c06b9-92f7-4a00-9854-3860d5b1069b.gif)
*Final application identifying the Colosseum in Rome from a user-uploaded image.*

---

## Core Features & Project Stages

This project is structured into three main parts, each building upon the last:

1.  **CNN from Scratch (`cnn_from_scratch.ipynb`)**:
    *   A deep dive into building a powerful CNN from the ground up.
    *   The final architecture is a custom ResNet-style network with residual blocks, batch normalization, and dropout for regularization.
    *   The model is trained using a `CosineAnnealingLR` scheduler and the `AdamW` optimizer to achieve high performance.

2.  **Transfer Learning (`transfer_learning.ipynb`)**:
    *   An investigation into the power of transfer learning using a pre-trained `ResNet18` model.
    *   The convolutional base of the pre-trained model is frozen to act as a high-quality feature extractor.
    *   Only the final classifier layer is replaced and trained, leading to much faster training times and typically higher accuracy.

3.  **Interactive Application (`app.ipynb`)**:
    *   The best-performing model (from transfer learning) is exported using TorchScript.
    *   A simple, user-friendly application is built using `ipywidgets` that allows anyone to upload an image and receive the top 5 landmark predictions with confidence scores.

---

## Technical Stack

*   **Framework:** PyTorch
*   **Libraries:** NumPy, Pandas, Matplotlib, Seaborn
*   **Tools:** Jupyter Lab, `ipywidgets` (for the interactive app)
*   **Models:** Custom ResNet-style CNN, Pre-trained ResNet18

---

## Getting Started

Follow these instructions to set up the environment and run the project on your local machine. An NVIDIA GPU is highly recommended for faster training.

### Prerequisites

*   Git
*   [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1.  **Clone the Repository**
    Open a terminal and navigate to your desired directory.
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and Activate Conda Environment**
    This command creates an isolated environment with all the necessary base packages.
    ```bash
    # Create the environment
    conda create --name landmark-project -y python=3.7.6
    
    # Activate the environment
    conda activate landmark-project
    ```
    *You must run `conda activate landmark-project` every time you open a new terminal to work on this project.*

3.  **Install All Requirements**
    Install PyTorch and all other required libraries from the `requirements.txt` file.
    ```bash
    # Install PyTorch first (recommended)
    conda install -y pytorch torchvision -c pytorch

    # Install the rest of the packages
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Lab**
    Once all packages are installed, you can start the Jupyter environment.
    ```bash
    jupyter lab
    ```
    This will open a new tab in your web browser. You can now navigate the project files and open the notebooks.

---

## Usage

To experience the full project, please run the notebooks in the following order:

1.  `cnn_from_scratch.ipynb` - To see how a custom CNN is built and trained.
2.  `transfer_learning.ipynb` - To train the transfer learning model and compare its performance.
3.  `app.ipynb` - To use the final, deployed application with your own images.

Execute the cells in each notebook from top to bottom to ensure the correct workflow. The final notebook, `app.ipynb`, also contains the script to generate a submission-ready archive of the project.
