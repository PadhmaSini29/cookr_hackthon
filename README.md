**Overview:**

This implements a Convolutional Neural Network (CNN) for classifying cuisine images into seven different categories. It utilizes TensorFlow and OpenCV libraries for image processing and model development. The CNN architecture consists of multiple convolutional layers followed by batch normalization, max-pooling, dropout, and dense layers. The model is trained on a dataset containing images of various cuisines, and its performance is evaluated using classification metrics.

**Features:**
- Loading images from folders with specified extensions.
- Preprocessing images by resizing and ensuring minimum dimensions.
- Splitting data into training and testing sets.
- Defining a CNN model architecture using TensorFlow's Keras API.
- Compiling the model with appropriate loss function and optimizer.
- Training the model with specified epochs and batch size.
- Evaluating the model's performance on test data using accuracy metrics.
- Saving the trained model to a file and loading it for future use.
- Predicting cuisine class probabilities for a given image.
- Displaying prediction results including predicted cuisine, food name, ingredients, and additional metadata from a CSV file.

**Installation:**
Ensure you have the following libraries installed:
- NumPy
- Matplotlib
- OpenCV (cv2)
- scikit-learn
- TensorFlow

You can install these libraries using pip:

```
pip install numpy matplotlib opencv-python scikit-learn tensorflow
```

**Usage:**
1. Import the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D
from sklearn.metrics import classification_report
import csv
```

2. Define the function `load_images_from_folder` to load images from a folder.

3. Define folder paths for each cuisine category and load images from those folders.

4. Preprocess images and labels.

5. Split data into training and testing sets.

6. Define the CNN model architecture using `Sequential` model.

7. Compile the model with appropriate loss function and optimizer.

8. Train the model using the training data.

9. Evaluate the model's performance on test data using `evaluate` method.

10. Print classification report.

11. Plot training history.

12. Load and preprocess an image for prediction.

13. Predict cuisine class probabilities for the image.

14. Display the predicted cuisine and additional information.

**Dataset:**
The dataset used in this code contains images of various cuisines categorized into seven classes:
1. Chettinadu cuisine
2. Kongu Nadu cuisine
3. Mudaliar cuisine
4. Nanjil cuisine
5. Pandiya nadu cuisine
6. Tamil Sahibu cuisine
7. Thanjavur cuisine

Ensure your dataset structure follows the folder hierarchy mentioned in the code for proper execution.
