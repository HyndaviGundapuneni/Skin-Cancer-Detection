# Melanoma Detection using CNN

## Project Overview

Melanoma is a deadly type of skin cancer that accounts for 75% of skin cancer-related deaths. Early detection is crucial to improve survival rates. This project implements a Convolutional Neural Network (CNN) model to classify melanoma from skin images, providing a potential solution to assist dermatologists in accurate and efficient diagnosis.

---

## Dataset

The dataset used for this project is sourced from the **ISIC (International Skin Imaging Collaboration)** repository. It includes labeled images categorized into different skin conditions. 

- **Train Images:** [Path: `/Train`]
- **Test Images:** [Path: `/Test`]

### Data Distribution
The dataset was found to have class imbalance issues, which were addressed using **data augmentation** with the Augmentor library. Additional synthetic samples were generated for underrepresented classes.

---

## Project Features

1. **Data Augmentation:** Rotations and transformations applied to handle class imbalance.
2. **CNN Architecture:**
   - Rescaling layer for image normalization.
   - Three convolutional layers with max-pooling.
   - Dropout layers for regularization.
   - Fully connected dense layers with softmax activation for classification.
3. **Callbacks:**
   - **ModelCheckpoint**: Save the best-performing model.
   - **EarlyStopping**: Prevent overfitting by halting training early if validation accuracy stops improving.
4. **Visualization:**
   - Class distribution and data augmentation results.
   - Training and validation metrics.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-detection-cnn.git
   cd melanoma-detection-cnn
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download and unzip the dataset to the specified paths:

Copy code
/Train
/Test
Run the project in a Jupyter Notebook or Google Colab.

Key Files and Directories
dataset/: Contains the train and test images.
model/: Contains the trained model file model.h5.
scripts/: Includes the Python scripts for training, validation, and testing.
README.md: Project documentation.
Training the Model
The model can be trained by executing the train_model.py script or directly running the cells in the provided notebook. Key configurations include:

Epochs: 20
Batch Size: 32
Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation and Results
The model achieves high accuracy on both training and validation datasets. Key metrics:

Training Accuracy: ~XX%
Validation Accuracy: ~YY%
The model's performance can be visualized using accuracy and loss plots.

Predicting New Samples
To predict new samples, ensure the test images are placed in the /Test directory. Run the following code snippet to load a sample image and generate predictions:

python
Copy code
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# Example: Predict for a single test image
test_image_path = 'path_to_test_image.jpg'
test_image = load_img(test_image_path, target_size=(180, 180))
predictions = model.predict(np.expand_dims(test_image, axis=0))
print(f"Predicted Class: {class_names[np.argmax(predictions)]}")
Contributing
Contributions are welcome! If you find bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Special thanks to the International Skin Imaging Collaboration (ISIC) for providing the dataset.
