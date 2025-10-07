# Plant Disease Detection - Group 8

This project focuses on the detection of diseases in plant leaves, specifically targeting tomato plants, using a variety of computer vision and image processing techniques. The goal is to preprocess leaf images to enhance features and build a model for accurate disease classification.

## 🚀 Project Pipeline

The project follows a comprehensive pipeline to process and analyze plant leaf images:

1.  **Background Removal**: Isolates the leaf from the background to focus on the region of interest.
2.  **Noise Reduction**: Cleans the image by removing unwanted noise, which can be caused by camera sensors or environmental factors.
3.  **Edge Detection**: Identifies the edges of the leaves and disease spots, which are crucial features for classification.
4.  **Contrast Enhancement**: Improves the contrast of the image to make the disease symptoms more prominent.
5.  **Data Augmentation**: Increases the size and diversity of the dataset by creating modified versions of the images. This helps in training more robust models.
6.  **Prediction**: The final step involves using the preprocessed images to predict the type of disease.

## 📂 Project Modules

The repository is organized into several notebooks and a Python script, each responsible for a specific part of the pipeline:

### 1. `plant-disease-prediction.ipynb`

This is the main notebook that orchestrates the preprocessing and prediction workflow. It applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of the images, making the disease patterns more visible.

### 2. `background_removal.ipynb`

This notebook provides a robust solution for removing the background from the leaf images. It uses techniques like:
*   HSV color thresholding to create a mask for the leaf.
*   Morphological operations (opening and closing) to refine the mask.
*   Identifying the largest connected component to ensure only the main leaf is captured.

### 3. `noise_reduction.ipynb`

This notebook deals with reducing noise in the images. It implements and compares two common filtering techniques:
*   **Bilateral Filter**: Reduces noise while preserving edges.
*   **Gaussian Blur**: Smooths the image to reduce high-frequency noise.

### 4. `edge_detection.py`

A Python script dedicated to edge detection using the Canny edge detector. This helps in extracting important structural features from the leaf images.

### 5. `data_augmentation.ipynb`

To improve the performance and generalization of the prediction model, this notebook is used to augment the dataset. It applies a variety of geometric transformations, including:
*   Rotation
*   Flipping (horizontal and vertical)
*   Translation
*   Shearing
*   Cropping

## ⚙️ Getting Started

### Prerequisites

To run this project, you will need Python and the following libraries:

*   OpenCV (`opencv-python`)
*   NumPy
*   Matplotlib
*   tqdm

You can install these dependencies using pip:
```bash
pip install opencv-python numpy matplotlib tqdm
```

### Dataset

The project expects a dataset of plant leaf images organized into subdirectories, where each subdirectory represents a different disease class. The notebooks use a dataset named `Plant_leave_diseases_dataset_without_augmentation`.

### Running the Project

1.  **Place your dataset** in the root directory of the project.
2.  **Run the notebooks** in the following order to follow the complete pipeline:
    1.  `background_removal.ipynb` - to remove backgrounds from your dataset.
    2.  `noise_reduction.ipynb` - to apply noise reduction to the output of the previous step.
    3.  `data_augmentation.ipynb` - to augment the cleaned dataset.
    4.  `plant-disease-prediction.ipynb` - to run the final preprocessing and prediction steps.

## 👥 Contributors

This project was developed by **Group - 8**.
