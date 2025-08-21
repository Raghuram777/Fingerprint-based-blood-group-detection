import tensorflow as tf
import numpy as np
import cv2
import os
import json

# --------------------
# Step 1: Load Saved Model
# --------------------
model = tf.keras.models.load_model("model.h5")
print("✅ Model loaded successfully!")

# --------------------
# Step 2: Load Class Labels from class_indices_simple.json
# --------------------
try:
    with open("class_indices_simple.json") as f:
        class_indices = json.load(f)
    # Invert the mapping to get index -> label
    index_to_label = {v: k for k, v in class_indices.items()}
    # Ensure correct order
    class_labels = [index_to_label[i] for i in range(len(index_to_label))]
    print("✅ Loaded class labels from JSON:", class_labels)
except FileNotFoundError:
    # Fallback to manual order (alphabetical)
    class_labels = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
    print("⚠️ Using fallback class labels:", class_labels)

# --------------------
# Step 3: Load and Preprocess Test Image
# --------------------

# 🔁 UPDATE this path to any image you want to test
test_image_path = "fingerprint_data\A-\cluster_0_32.BMP"

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
