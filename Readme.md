# Simple Hand Digit Recognition with TensorFlow

This repository contains a simple Python script that demonstrates hand digit recognition using the MNIST dataset and TensorFlow.

## Overview

The provided script trains a neural network model to recognize hand-drawn digits from the MNIST dataset. It utilizes the TensorFlow library for building and training the model. The trained model is then evaluated on a test set to measure its accuracy.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using pip:

```
pip install tensorflow numpy matplotlib
```

## Usage

1. Clone the repository:

```
git clone https://github.com/your_username/simple-hand-digit-recognition.git
```

2. Navigate to the project directory:

```
cd simple-hand-digit-recognition
```

3. Run the script:

```
python hand_digit_recognition.py
```

## Explanation

- **Loading the Dataset**: The MNIST dataset is loaded using TensorFlow's `mnist.load_data()` function. It consists of 60,000 training images and 10,000 test images of handwritten digits.

- **Preprocessing the Data**: The images are normalized to have pixel values in the range [0, 1]. Labels are one-hot encoded using TensorFlow's `to_categorical()` function.

- **Defining the Model**: A simple neural network model is defined using the Sequential API. It consists of a flatten layer followed by two dense layers with ReLU activation functions, and a final dense layer with softmax activation for classification.

- **Compiling the Model**: The model is compiled with the Adam optimizer and categorical crossentropy loss function.

- **Training the Model**: The model is trained on the training data for one epoch with a batch size of 100. A validation split of 0.1 is used for validation during training.

- **Evaluating the Model**: The trained model is evaluated on the test data to measure its accuracy.

- **Example Prediction**: An example image from the test set is randomly selected, and the model predicts the digit. The predicted digit along with the image is displayed using Matplotlib.

## Results

After training the model for one epoch, it achieves an accuracy of approximately 98% on the test set.

![image](https://github.com/VEDAMNT/HandDigit_Recognition/assets/99802920/04885618-639c-43df-b1ea-830eff759529)

