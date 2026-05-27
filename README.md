# Fiber Orientation Tensor Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict fiber orientation tensor components from synthetic fiber composite microstructures.

The main prediction target is:

```text
microstructure → orientation tensor
```

The orientation tensor components are:

- `a_xx` : fiber alignment along the X direction
- `a_yy` : fiber alignment along the Y direction
- `a_zz` : fiber alignment along the Z direction

These components approximately satisfy:

```text
a_xx + a_yy + a_zz = 1
```

The project compares:

1. Physics-informed descriptor-based machine learning
2. Image-based deep learning using CNNs

---

# Dataset

Dataset source:
- Synthetic fiber composite microstructure dataset from Zenodo

Each sample contains:
- `.jpg` microstructure image
- `.csv` file with fiber geometry information
- orientation tensor labels embedded in the filename

Example:

```text
0.0030,0.5303,0.4667.jpg
```

Corresponds to:

```text
a_xx = 0.0030
a_yy = 0.5303
a_zz = 0.4667
```

---

# Notebook Structure

## 01_model1.ipynb

### Descriptor-Based Random Forest Model

This notebook builds a physics-informed machine learning model using engineered descriptors extracted from CSV files.

### Workflow

1. Load CSV files
2. Extract statistical descriptors
3. Train/test split (80/20)
4. Train Random Forest regressor
5. Predict orientation tensor
6. Evaluate using R², MAE, feature importance, and parity plots

### Extracted descriptors

- number of fibers
- packing fraction
- mean fiber area
- area standard deviation
- aspect ratio statistics
- orientation spread

### Main result

```text
R² ≈ 0.996
```

### Interpretation

The descriptor-based model achieved excellent performance because the engineered descriptors directly encode physically meaningful microstructure information.

---

## 02_model2.ipynb

### Initial Image-Based CNN Model

This notebook tests whether a convolutional neural network can predict orientation tensors directly from raw microstructure images.

### Workflow

1. Load JPG images
2. Resize images to 64 × 64 grayscale
3. Convert images into CNN input tensors
4. Train/test split (80/20)
5. Train CNN regression model
6. Predict orientation tensor
7. Evaluate using R², MAE, loss curves, and parity plots

### Main result

```text
R² ≈ 0.01
```

### Why Model 2 failed

- relatively small dataset
- unnormalized tensor labels
- unstable CNN training
- regression task more difficult than classification
- raw images lacked explicit physical descriptors

### Interpretation

The initial CNN model demonstrated that raw deep learning alone was insufficient without proper dataset size and training optimization.

---

## 03_model3.ipynb

### Improved Image-Based CNN Model

This notebook improves the CNN model by modifying the training procedure and dataset.

### Improvements from Model 2

- larger image dataset
- tensor label normalization using `StandardScaler`
- simplified CNN architecture
- early stopping
- improved training stability

### Workflow

1. Load larger image dataset
2. Resize images to 64 × 64 grayscale
3. Convert images into CNN input tensors
4. Normalize tensor labels
5. Train optimized CNN regression model
6. Use early stopping
7. Convert predictions back to original tensor scale
8. Evaluate using R², MAE, loss curves, and parity plots

### Main result

```text
R² ≈ 0.991
MAE ≈ 0.012
```

### Interpretation

The improved CNN successfully learned orientation tensor information directly from microstructure images after optimizing the data representation and training process.

---

## 04_model3_test.ipynb

### Using the Trained CNN Model

This notebook demonstrates how to use the trained Model 3 CNN to predict orientation tensor values from a new microstructure image.

### Required saved files

Saved from `03_model3.ipynb`:

```python
model3.save("model3_cnn_tensor.keras")

import joblib
joblib.dump(y_scaler, "model3_y_scaler.pkl")
```

### Prediction workflow

1. Load trained CNN model
2. Load saved label scaler
3. Input a microstructure image
4. Resize image to 64 × 64 grayscale
5. Convert image into CNN tensor input
6. Predict scaled tensor values
7. Convert prediction back to original tensor scale
8. Output predicted tensor values

### Example

```python
pred_tensor, img = predict_tensor_from_image(
    image_path,
    model3,
    y_scaler
)

print("a_xx =", pred_tensor[0])
print("a_yy =", pred_tensor[1])
print("a_zz =", pred_tensor[2])
```

---

# Final Model Comparison

| Model | Input | Method | Performance |
|---|---|---|---|
| Model 1 | engineered descriptors | Random Forest | R² ≈ 0.996 |
| Model 2 | raw images | initial CNN | R² ≈ 0.01 |
| Model 3 | raw images | improved CNN | R² ≈ 0.991 |

---

# Main Scientific Conclusions

1. Engineered descriptors predict orientation tensors extremely well.
2. CNN performance strongly depends on dataset size and training stability.
3. Label normalization and early stopping significantly improve CNN regression performance.
4. Physics-informed descriptors are more interpretable and data-efficient.
5. CNNs can successfully learn orientation information directly from microstructure images after optimization.

---

# Materials Informatics Concepts Used

- data representation
- feature engineering
- supervised regression
- Random Forest regression
- convolutional neural networks
- target normalization
- early stopping
- parity plots
- model comparison
- interpretability
- materials structure-property relationships

---

# Acknowledgement

This project was developed using machine learning concepts, workflows, and coding techniques learned from MATSCI 459: Artificial Intelligence in Materials Science.
