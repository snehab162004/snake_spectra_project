# Snake Detection Project

This project is a machine learning-based snake detection system. It uses image data to classify snakes as venomous or non-venomous based on patterns and color.

## Features

- Train, validate, and test a convolutional neural network (CNN) model on snake images.
- Predict the class of a snake (venomous or non-venomous) from an input image.
- Display snake details and provide first aid instructions based on the classification.

## Project Structure

snake_spectra_project/
│
├── datasets_hackathon.csv # Original dataset
├── datasets_hackathon_updated.csv # Updated dataset with corrected paths
├── train_data.csv # Training dataset
├── val_data.csv # Validation dataset
├── test_data.csv # Test dataset
├── snake_spectra.py # Main script for training and prediction
├── snake_detection_model.keras # Saved model
├── snakes_images/ # Directory containing snake images
├── .gitignore # Git ignore file
├── README.md # This README file
└── venv/ # Virtual environment directory (ignored in .gitignore)


## Setup

### Prerequisites

- Python 3.6 or later
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/snake_detection_project.git
   cd snake_detection_project

2. Create a virtual environment and activate it:

bash
python -m venv venv
source venv/bin/activate

On Windows: venv\Scripts\activate

.gitignore
The .gitignore file ensures that unnecessary files are excluded from the repository.
It includes entries for virtual environment directories, Python cache files, Jupyter Notebook checkpoints, large model files, logs, and temporary files.
