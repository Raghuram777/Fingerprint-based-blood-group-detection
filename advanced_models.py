import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras import layers, models
import os
import json

def create_advanced_cnn(input_shape=(128, 128, 1), num_classes=8):
    """
    Advanced CNN with modern techniques for fingerprint classification
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Block - Feature extraction
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Block - Deeper features
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Block - Complex patterns
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Block - High-level features
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fifth Block - Very high-level features
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        # Global Average Pooling instead of Flatten
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with regularization
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=8):
    """
    EfficientNet-based model (state-of-the-art for image classification)
    """
    # Load pre-trained EfficientNetB3
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Unfreeze the top layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_resnet_model(input_shape=(224, 224, 3), num_classes=8):
    """
    ResNet50V2-based model with fine-tuning
    """
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Fine-tuning strategy
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_advanced_data_generator():
    """
    Advanced data augmentation for better generalization
    """
    return ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        channel_shift_range=0.1,
        fill_mode='nearest'
    )

def train_advanced_model(model_type='advanced_cnn'):
    """
    Train advanced model with best practices
    """
    # Dataset path
    data_dir = "fingerprint_data"
    
    if model_type == 'advanced_cnn':
        img_size = (128, 128)
        color_mode = 'grayscale'
        input_shape = (128, 128, 1)
        model = create_advanced_cnn(input_shape, 8)
    elif model_type == 'efficientnet':
        img_size = (224, 224)
        color_mode = 'rgb'
        input_shape = (224, 224, 3)
        model = create_efficientnet_model(input_shape, 8)
    elif model_type == 'resnet':
        img_size = (224, 224)
        color_mode = 'rgb'
        input_shape = (224, 224, 3)
        model = create_resnet_model(input_shape, 8)
    else:
        raise ValueError("model_type must be 'advanced_cnn', 'efficientnet', or 'resnet'")
    
    # Data generators
    datagen = get_advanced_data_generator()
    
    train = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=16,  # Smaller batch for stability
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Save class indices
    with open(f"class_indices_{model_type}.json", "w") as f:
        json.dump(train.class_indices, f)
    print(f"✅ Saved class_indices_{model_type}.json:", train.class_indices)
    
    # Compile model with different strategies for different models
    if model_type == 'advanced_cnn':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        # Lower learning rate for pre-trained models
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Advanced callbacks
    callbacks = [
        ModelCheckpoint(
            f'{model_type}_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            patience=10,
            monitor='val_accuracy',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.3,
            patience=5,
            monitor='val_accuracy',
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(f'{model_type}_training_log.csv')
    ]
    
    print(f"\n🚀 Training {model_type.upper()} model...")
    model.summary()
    
    # Train model
    epochs = 50 if model_type == 'advanced_cnn' else 30
    
    history = model.fit(
        train,
        epochs=epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(f"{model_type}_final.h5")
    print(f"✅ Saved {model_type}_final.h5")
    
    return model, history

if __name__ == "__main__":
    print("🎯 ADVANCED FINGERPRINT BLOOD GROUP CLASSIFICATION")
    print("=" * 60)
    
    # Train different models
    models_to_train = ['advanced_cnn', 'efficientnet', 'resnet']
    
    for model_type in models_to_train:
        try:
            print(f"\n🔄 Training {model_type.upper()}...")
            model, history = train_advanced_model(model_type)
            print(f"✅ {model_type.upper()} training completed!")
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            continue
    
    print("\n🎉 All models trained! Use ensemble_model.py to evaluate performance.")
