# Face Recognition Application

## Overview
The **Face Recognition Application** is a Python-based deep learning project that uses a Convolutional Neural Network (CNN) to identify individuals from facial images. It preprocesses images, trains a model using the LFW (Labeled Faces in the Wild) dataset, and predicts the identity of a person from a new image.

## Features
- **Image Preprocessing:** Resizes images and normalizes pixel values.
- **Data Augmentation:** Applies transformations like rotation, shift, and flipping to enhance training data.
- **CNN Architecture:** Utilizes convolutional layers with batch normalization and dropout for robust training.
- **Callbacks:** Implements early stopping and model checkpointing for efficient training.
- **Learning Rate Scheduler:** Dynamically adjusts the learning rate during training.
- **Prediction Functionality:** Identifies individuals from unseen images.

## Installation
To run this project, ensure you have the following installed:
- Python 3.x
- TensorFlow 2.x
- Required libraries (see `requirements.txt`)

### Steps:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-recognition-app.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd face-recognition-app
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the dataset:**
   Place the `lfw-funneled.tgz` dataset in the project directory.
5. **Extract the dataset:**
   The script automatically extracts the dataset to `lfw-funneled/lfw_funneled`.

## Usage
1. **Train the Model:**
   Run the script to preprocess images, train the model, and save the best weights.
   ```bash
   python FACERECOGNİTİON_BABYYYYYY.py
   ```
2. **Predict on New Images:**
   Use the `predict_image()` function to identify individuals from new images:
   ```python
   predicted_person = predict_image('path_to_new_image.jpg')
   print(predicted_person)
   ```

## Example Output
```
Processing directory: person_1
Processing directory: person_2
...
Identification success: John Doe
```

## Model Architecture
The model consists of:
- Multiple convolutional layers with ReLU activation and He initialization.
- Max pooling layers to reduce spatial dimensions.
- Batch normalization for faster convergence.
- Dropout layers to prevent overfitting.
- Fully connected layers with softmax activation for classification.

## Dataset
The application uses the **Labeled Faces in the Wild (LFW)** dataset. Ensure the dataset is correctly placed in the project directory before running the script.

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-new-feature`).
3. Commit and push your changes.
4. Open a pull request.

## Contact
For any questions or support, please open an issue on GitHub.

