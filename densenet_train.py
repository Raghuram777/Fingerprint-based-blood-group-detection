import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import json

# Path to full dataset
data_dir = "fingerprint_data"  # full version, not _small

img_size = (224, 224)  # Required for DenseNet
batch_size = 32

# Augmented Data Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.9, 1.1),
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='rgb',  # Required for pretrained DenseNet
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

num_classes = len(train.class_indices)

# Save class indices mapping
with open("class_indices.json", "w") as f:
    json.dump(train.class_indices, f)
print("✅ Saved class_indices.json:", train.class_indices)

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models

# Load pretrained DenseNet121 base
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base layers
base_model.trainable = False

# Build custom head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train with callbacks
callbacks = [
    ModelCheckpoint('densenet_best.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=7, monitor='val_accuracy', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_accuracy', min_lr=1e-7)
]

history = model.fit(
    train,
    epochs=30,  # Increased epochs
    validation_data=val,
    callbacks=callbacks
)

# Save model
model.save("densenet_model.h5")