Fiber Orientation Tensor Prediction Project
Project Overview
This project uses machine learning to predict fiber orientation tensor components from synthetic fiber composite microstructures.
The main prediction target is:
```text
image or microstructure descriptors → a_xx, a_yy, a_zz
```
The orientation tensor describes how fibers are aligned in different directions:
`a_xx`: alignment along the X direction
`a_yy`: alignment along the Y direction
`a_zz`: alignment along the Z direction
The tensor components approximately satisfy:
```text
a_xx + a_yy + a_zz = 1
```
This is important because fiber orientation affects anisotropic material properties such as stiffness, strength, and thermal behavior.
---
Dataset
The project uses a synthetic fiber microstructure dataset from Zenodo.
Each sample contains:
A `.jpg` microstructure image
A `.csv` file containing fiber-level geometry information
Ground-truth tensor labels embedded in the filename
Example filename:
```text
0.0030,0.5303,0.4667.jpg
```
This corresponds to:
```text
a_xx = 0.0030
a_yy = 0.5303
a_zz = 0.4667
```
---
Notebook 01: `01_model1.ipynb`
Model 1: Descriptor-Based Random Forest Model
This notebook builds a physics-informed machine learning model using the CSV files.
Workflow
Load CSV files
Extract microstructure descriptors
Build one feature vector per image
Split data into training and testing sets using an 80/20 split
Train a Random Forest regression model
Predict `a_xx`, `a_yy`, and `a_zz`
Evaluate using R², MAE, feature importance, and parity plots
Extracted descriptors
Examples of descriptors include:
number of fibers
packing fraction
mean fiber area
area standard deviation
aspect ratio statistics
orientation spread
Main result
Model 1 achieved very high prediction accuracy:
```text
R² ≈ 0.996
```
Interpretation
The descriptor-based model worked very well because the engineered features directly describe the microstructure. This model is physics-informed, interpretable, and data-efficient.
---
Notebook 02: `02_model2.ipynb`
Model 2: Initial Image-Based CNN Model
This notebook tests whether a convolutional neural network can predict orientation tensors directly from raw microstructure images.
Workflow
Load JPG images
Resize images to 64 × 64 grayscale
Convert images into CNN input tensors
Split data into training and testing sets using an 80/20 split
Train a CNN regression model
Predict `a_xx`, `a_yy`, and `a_zz`
Evaluate using R², MAE, loss curves, and parity plots
Main result
Model 2 performed poorly:
```text
R² ≈ 0.01
```
Why Model 2 failed
The initial CNN model struggled because:
the image dataset was relatively small
target tensor labels were not normalized
CNN regression is harder than classification
raw images do not explicitly provide physical descriptors
training was unstable
Interpretation
This model shows that raw deep learning does not automatically work well without enough data and proper training setup.
---
Notebook 03: `03_model3.ipynb`
Model 3: Improved Image-Based CNN Model
This notebook improves the CNN model by modifying the data and training procedure.
Improvements from Model 2
Model 3 improved the image-based model by adding:
More training images
Target label normalization using `StandardScaler`
A simpler CNN architecture
Early stopping
More stable training and validation behavior
Workflow
Load a larger JPG image dataset
Resize images to 64 × 64 grayscale
Convert images into CNN input tensors
Normalize tensor labels
Train an improved CNN regression model
Use early stopping to monitor validation loss
Convert scaled predictions back to original tensor values
Evaluate using R², MAE, loss curves, and parity plots
Main result
Model 3 achieved strong performance:
```text
R² ≈ 0.991
MAE ≈ 0.012
```
Interpretation
Model 3 shows that CNNs can successfully learn orientation tensor information from images when the dataset is larger and the training process is improved.
---
Notebook 04: `04_model3_test.ipynb`
Purpose
This notebook shows how to use the trained Model 3 CNN to predict orientation tensor values from a new microstructure image.
This is the practical demonstration notebook.
Required saved files
Before using this notebook, Model 3 must be saved from `03_model3.ipynb`:
```python
model3.save("model3_cnn_tensor.keras")

import joblib
joblib.dump(y_scaler, "model3_y_scaler.pkl")
```
The test notebook then loads:
```text
model3_cnn_tensor.keras
model3_y_scaler.pkl
```
Prediction workflow
Load trained CNN model
Load saved label scaler
Input one microstructure image
Resize image to 64 × 64 grayscale
Convert image into CNN input tensor
Predict scaled tensor values
Convert prediction back to original tensor scale
Output predicted `a_xx`, `a_yy`, and `a_zz`
Example use
```python
image_path = r"path_to_your_image.jpg"

pred_tensor, img = predict_tensor_from_image(
    image_path,
    model3,
    y_scaler
)

print("Predicted orientation tensor:")
print("a_xx =", pred_tensor[0])
print("a_yy =", pred_tensor[1])
print("a_zz =", pred_tensor[2])
print("Sum =", pred_tensor.sum())
```
If the image filename contains the true tensor values, the notebook can also compare prediction with ground truth.
---
Final Model Comparison
Model	Input	Method	Result
Model 1	CSV descriptors	Random Forest	Excellent, R² ≈ 0.996
Model 2	raw images	initial CNN	Poor, R² ≈ 0.01
Model 3	raw images	improved CNN	Strong, R² ≈ 0.991
---
Main Scientific Conclusions
Engineered microstructure descriptors can predict orientation tensors very accurately.
Raw CNN models can fail if the dataset is small or the labels are not properly normalized.
Improved CNN training with more data, label scaling, and early stopping greatly improves performance.
Physics-informed descriptors are more interpretable and data-efficient.
CNNs can learn directly from images, but they require more careful training and more data.
---
Materials Informatics Concepts Used
This project demonstrates several key concepts from MATSCI 459:
data representation
feature engineering
supervised regression
train/test split
Random Forest regression
convolutional neural networks
target normalization
early stopping
parity plots
model comparison
interpretability
materials structure-property relationships
---
Suggested Presentation Message
The central message of this project is:
```text
Machine learning can predict fiber orientation tensors from both engineered descriptors and raw microstructure images, but model performance strongly depends on data representation, dataset size, and training stability.
```
