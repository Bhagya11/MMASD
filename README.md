
# Autism Spectrum Disorder Detection using sMRI and Phenotypic Data

This project presents a comprehensive workflow for the detection of Autism Spectrum Disorder (ASD) by leveraging both structural Magnetic Resonance Imaging (sMRI) data and phenotypic information. The pipeline encompasses several stages, including data preprocessing, feature extraction, data fusion, and classification.

## Table of Contents

- [Project Workflow](#project-workflow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modules](#modules)
  - [NifTi to sMRI Conversion and Preprocessing](#nifti-to-smri-conversion-and-preprocessing)
  - [Phenotype Preprocessing](#phenotype-preprocessing)
  - [sMRI Feature Extraction](#smri-feature-extraction)
  - [Phenotype Dimensionality Reduction](#phenotype-dimensionality-reduction)
  - [Data Fusion](#data-fusion)
  - [sMRI-based Classification](#smri-based-classification)
  - [Phenotype-based Classification](#phenotype-based-classification)
  - [Revised Classification](#revised-classification)
  - [Sitewise Cross-Validation](#sitewise-cross-validation)

## Project Workflow

The project is structured as a sequential pipeline:

1.  **NifTi to sMRI Conversion**: Converts NIfTI files into 2D sMRI images.
2.  **Phenotype Preprocessing**: Cleans and prepares the phenotypic data for analysis.
3.  **sMRI Feature Extraction**: Extracts meaningful features from the sMRI images using deep learning models (CNN, ResNet, VGG).
4.  **Phenotype Dimensionality Reduction**: Reduces the dimensionality of the phenotypic data to retain the most important features.
5.  **Data Fusion**: Merges the features from both sMRI and phenotypic data.
6.  **Classification**: Employs various machine learning models to classify subjects into ASD and non-ASD groups based on sMRI, phenotypic, and fused data.
7.  **Sitewise Cross-Validation**: Performs a rigorous Leave-One-Site-Out cross-validation to evaluate the model's generalizability across different data sources.

## Getting Started

### Prerequisites

- Python 3.7+
- The project's data, including the NIfTI files and the `Phenotypic_V1_0b_preprocessed.csv` file, should be placed in the root directory.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Bhagya11/MMASD_Modular_Code
    ```
2.  Navigate to the project directory:
    ```bash
    cd MMASD_Modular_Code
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

-   `main.py`: The entry point for running the entire project pipeline.
-   `README.md`: This file, providing an overview of the project.
-   `requirements.txt`: A list of the Python packages required for this project.
-   `src/`: A directory containing the core source code, organized into modules.
    -   `nifti_to_smri.py`: Handles the conversion of NIfTI files to sMRI images.
    -   `phenotype_preprocessing.py`: Contains functions for cleaning and preprocessing the phenotypic data.
    -   `smri_feature_extraction.py`: Implements feature extraction from sMRI images using deep learning models.
    -   `phenotype_dr.py`: Performs dimensionality reduction on the phenotypic data.
    -   `data_fusion.py`: Fuses the sMRI and phenotypic features.
    -   `smri_classification.py`: Classifies subjects based on sMRI features.
    -   `phenotype_classification.py`: Classifies subjects based on phenotypic features.
    -   `classification_revised.py`: Provides a revised and more detailed classification analysis.
    -   `sitewise_cv.py`: Performs Leave-One-Site-Out cross-validation to assess model generalizability.

## Usage

To run the entire pipeline, execute the `main.py` script from the root directory of the project:

```bash
python main.py
```

You can also run individual modules if needed, but it is recommended to follow the sequence in `main.py`.

## Modules

### NifTi to sMRI Conversion and Preprocessing

-   **File**: `src/nifti_to_smri.py`
-   **Description**: This module is responsible for converting 3D NIfTI files into 2D PNG images. It reads the phenotypic data to label the images as "Autism" or "NonAutism".

### Phenotype Preprocessing

-   **File**: `src/phenotype_preprocessing.py`
-   **Description**: This module cleans the phenotypic data by handling missing values and encoding categorical features, making it suitable for machine learning.

### sMRI Feature Extraction

-   **File**: `src/smri_feature_extraction.py`
-   **Description**: This module uses pre-trained deep learning models (CNN, ResNet, and VGG16) to extract features from the generated sMRI images.

### Phenotype Dimensionality Reduction

-   **File**: `src/phenotype_dr.py`
-   **Description**: Reduces the dimensionality of the preprocessed phenotypic data using Principal Component Analysis (PCA).

### Data Fusion

-   **File**: `src/data_fusion.py`
-   **Description**: This module merges the extracted sMRI features with the processed phenotypic data based on the subject ID.

### sMRI-based Classification

-   **File**: `src/smri_classification.py`
-   **Description**: Performs classification using various machine learning models on the sMRI features and evaluates their performance.

### Phenotype-based Classification

-   **File**: `src/phenotype_classification.py`
-   **Description**: Trains and evaluates classification models using the dimensionality-reduced phenotypic data.

### Revised Classification

-   **File**: `src/classification_revised.py`
-   **Description**: This module provides a more detailed classification analysis, including additional metrics like specificity and generates ROC curves for the different models.

### Sitewise Cross-Validation

-   **File**: `src/sitewise_cv.py`

-   **Description**: This module implements a robust evaluation strategy called Leave-One-Site-Out Cross-Validation (LOSO-CV). It assesses how well the models generalize to new, unseen data acquisition sites by training on all sites except one and testing on the held-out site. This process is repeated for every site to provide a measure of real-world model performance.

