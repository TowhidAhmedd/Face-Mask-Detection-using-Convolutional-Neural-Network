# Face Mask Detection using CNN

## Overview

This project implements a **Face Mask Detection system** using **Convolutional Neural Networks (CNNs)** to automatically identify whether a person is wearing a mask. It can be integrated into surveillance systems for real-time monitoring.

---

## Features

* Detects **masked** vs **unmasked** faces.
* Real-time image prediction capability.
* CNN-based architecture for accurate classification.
* Preprocessing: resizing, normalization, and RGB conversion.
* Train/Test split with validation implemented.

---

## Dataset

* Two classes: `with_mask` (1) and `without_mask` (0).
* Images resized to **128x128 pixels**.
* Stored in Google Drive folders: `/with_mask` and `/without_mask`.

---

## Technologies

* Python 3, TensorFlow/Keras, OpenCV, PIL, NumPy, Matplotlib, Scikit-learn.

---

## Methodology

1. **Preprocessing**: Read, resize, RGB conversion, normalize.
2. **CNN Architecture**:

   * Conv2D → MaxPooling2D → Conv2D → MaxPooling2D
   * Flatten → Dense → Dropout → Dense → Dropout → Output Dense (Sigmoid)
3. **Training**: Adam optimizer, Sparse Categorical Cross-Entropy loss, 5 epochs, 10% validation split.
4. **Evaluation**: Accuracy on test set; training/validation loss & accuracy plots.

---

## Usage

```python
input_image_path = 'path_to_image.jpg'
input_image = cv2.imread(input_image_path)
input_image_resized = cv2.resize(input_image, (128,128))/255
input_image_reshaped = np.reshape(input_image_resized, [1,128,128,3])
pred_label = np.argmax(model.predict(input_image_reshaped))
print("Mask" if pred_label == 1 else "No Mask")
```

---

## Results

* High accuracy in detecting masked vs unmasked faces.
* Can be extended to real-time webcam detection.

---

## Future Work

* Expand dataset for better generalization.
* Real-time webcam integration.
* Use pre-trained CNNs (MobileNet, ResNet) for transfer learning.
* Multi-class mask detection.

---

## Author

**Towhid Ahmed** – AI & Deep Learning Enthusiast
LinkedIn: [your-linkedin-profile]
GitHub: [your-github-profile]
