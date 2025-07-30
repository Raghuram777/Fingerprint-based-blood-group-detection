import tensorflow as tf
import numpy as np
import cv2
import os

# --------------------
# Step 1: Load Saved Model
# --------------------
model = tf.keras.models.load_model("model.h5")
print("✅ Model loaded successfully!")

# --------------------
# Step 2: Class Labels (must match training)
# --------------------
class_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
# You can update this if your folder names were different

# --------------------
# Step 3: Load and Preprocess Test Image
# --------------------

# 🔁 UPDATE this path to any image you want to test
test_image_path = "fingerprint_data\AB+\cluster_4_5971.BMP"

if not os.path.exists(test_image_path):
    raise FileNotFoundError("❌ Test image not found. Check the path.")

# Load image in grayscale
img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128)) / 255.0  # Normalize
img = img.reshape(1, 128, 128, 1)  # Model input shape

# --------------------
# Step 4: Predict
# --------------------
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_label = class_labels[predicted_index]
confidence = np.max(prediction) * 100

# --------------------
# Step 5: Output
# --------------------
print(f"\n🧠 Predicted Blood Group: {predicted_label}")
print(f"📊 Confidence: {confidence:.2f}%")
