import tensorflow as tf
import numpy as np
import cv2
import os
import json

# --------------------
# Step 1: Load DenseNet Model
# --------------------
model = tf.keras.models.load_model("densenet_model.h5")
print("✅ DenseNet model loaded successfully!")

# --------------------
# Step 2: Load Class Labels from class_indices.json
# --------------------
with open("class_indices.json") as f:
    class_indices = json.load(f)
# Invert the mapping to get index -> label
index_to_label = {v: k for k, v in class_indices.items()}
# Ensure correct order
class_labels = [index_to_label[i] for i in range(len(index_to_label))]

# --------------------
# Step 3: Load and Preprocess Image
# --------------------
# 🔁 UPDATE this path to your test image
test_image_path = "fingerprint_data/A-/cluster_1_47.BMP"

if not os.path.exists(test_image_path):
    raise FileNotFoundError("❌ Test image not found. Check the path.")

# Read image in RGB
img = cv2.imread(test_image_path)
img = cv2.resize(img, (224, 224))  # Resize for DenseNet
img = img / 255.0  # Normalize
img = img.reshape(1, 224, 224, 3)  # Model input shape

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