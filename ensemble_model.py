import tensorflow as tf
import numpy as np
import cv2
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleFingerprint:
    def __init__(self):
        self.models = {}
        self.class_labels = None
        
    def load_models(self):
        """Load all trained models"""
        try:
            # Load Simple CNN
            if os.path.exists("best_model.h5"):
                self.models['simple_cnn'] = tf.keras.models.load_model("best_model.h5")
                print("✅ Loaded Simple CNN")
            elif os.path.exists("model.h5"):
                self.models['simple_cnn'] = tf.keras.models.load_model("model.h5")
                print("✅ Loaded Simple CNN (fallback)")
                
            # Load DenseNet
            if os.path.exists("densenet_best.h5"):
                self.models['densenet'] = tf.keras.models.load_model("densenet_best.h5")
                print("✅ Loaded DenseNet (best)")
            elif os.path.exists("densenet_model.h5"):
                self.models['densenet'] = tf.keras.models.load_model("densenet_model.h5")
                print("✅ Loaded DenseNet (fallback)")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            
    def load_class_labels(self):
        """Load class labels with proper mapping"""
        try:
            # Try to load from JSON first
            if os.path.exists("class_indices.json"):
                with open("class_indices.json") as f:
                    class_indices = json.load(f)
                index_to_label = {v: k for k, v in class_indices.items()}
                self.class_labels = [index_to_label[i] for i in range(len(index_to_label))]
                print("✅ Loaded class labels from class_indices.json:", self.class_labels)
            elif os.path.exists("class_indices_simple.json"):
                with open("class_indices_simple.json") as f:
                    class_indices = json.load(f)
                index_to_label = {v: k for k, v in class_indices.items()}
                self.class_labels = [index_to_label[i] for i in range(len(index_to_label))]
                print("✅ Loaded class labels from class_indices_simple.json:", self.class_labels)
            else:
                # Fallback to alphabetical order
                self.class_labels = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
                print("⚠️ Using fallback alphabetical class labels:", self.class_labels)
        except Exception as e:
            print(f"❌ Error loading class labels: {e}")
            self.class_labels = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
            
    def preprocess_image_simple(self, image_path):
        """Preprocess image for Simple CNN (128x128 grayscale)"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128)) / 255.0
        img = img.reshape(1, 128, 128, 1)
        return img
        
    def preprocess_image_densenet(self, image_path):
        """Preprocess image for DenseNet (224x224 RGB)"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = img.reshape(1, 224, 224, 3)
        return img
        
    def predict_single(self, image_path, method='ensemble'):
        """
        Predict blood group for a single image
        method: 'ensemble', 'simple_cnn', 'densenet'
        """
        predictions = {}
        
        # Simple CNN prediction
        if 'simple_cnn' in self.models:
            img_simple = self.preprocess_image_simple(image_path)
            pred_simple = self.models['simple_cnn'].predict(img_simple, verbose=0)
            predictions['simple_cnn'] = pred_simple[0]
            
        # DenseNet prediction
        if 'densenet' in self.models:
            img_dense = self.preprocess_image_densenet(image_path)
            pred_dense = self.models['densenet'].predict(img_dense, verbose=0)
            predictions['densenet'] = pred_dense[0]
            
        if method == 'ensemble' and len(predictions) > 1:
            # Weighted ensemble (DenseNet usually performs better)
            weights = {'simple_cnn': 0.3, 'densenet': 0.7}
            ensemble_pred = np.zeros(8)
            
            for model_name, pred in predictions.items():
                ensemble_pred += weights.get(model_name, 1.0) * pred
                
            ensemble_pred = ensemble_pred / sum(weights.values())
            predicted_index = np.argmax(ensemble_pred)
            confidence = np.max(ensemble_pred) * 100
            
        elif method in predictions:
            pred = predictions[method]
            predicted_index = np.argmax(pred)
            confidence = np.max(pred) * 100
            
        else:
            # Use any available model
            pred = list(predictions.values())[0]
            predicted_index = np.argmax(pred)
            confidence = np.max(pred) * 100
            
        predicted_label = self.class_labels[predicted_index]
        
        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_predictions': predictions,
            'predicted_index': predicted_index
        }
        
    def evaluate_on_dataset(self, data_dir="fingerprint_data_small", method='ensemble'):
        """Evaluate ensemble on test dataset"""
        all_true_labels = []
        all_pred_labels = []
        
        for blood_group in self.class_labels:
            folder_path = os.path.join(data_dir, blood_group)
            if not os.path.exists(folder_path):
                continue
                
            for image_file in os.listdir(folder_path)[:20]:  # Test on 20 images per class
                if image_file.endswith(('.BMP', '.jpg', '.png')):
                    image_path = os.path.join(folder_path, image_file)
                    result = self.predict_single(image_path, method)
                    
                    all_true_labels.append(blood_group)
                    all_pred_labels.append(result['predicted_label'])
                    
        # Calculate accuracy
        accuracy = sum(1 for true, pred in zip(all_true_labels, all_pred_labels) if true == pred) / len(all_true_labels)
        
        print(f"\n🎯 {method.upper()} ACCURACY: {accuracy*100:.2f}%")
        print(f"📊 Total samples tested: {len(all_true_labels)}")
        
        # Classification report
        print("\n📋 CLASSIFICATION REPORT:")
        print(classification_report(all_true_labels, all_pred_labels))
        
        # Confusion Matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=self.class_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.title(f'{method.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{method}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy

# Example usage
if __name__ == "__main__":
    # Initialize ensemble
    ensemble = EnsembleFingerprint()
    ensemble.load_models()
    ensemble.load_class_labels()
    
    # Test single prediction
    test_image = "fingerprint_data/A-/cluster_1_47.BMP"
    if os.path.exists(test_image):
        result = ensemble.predict_single(test_image, method='ensemble')
        print(f"\n🧠 Ensemble Prediction: {result['predicted_label']}")
        print(f"📊 Confidence: {result['confidence']:.2f}%")
        
        # Compare individual models
        result_simple = ensemble.predict_single(test_image, method='simple_cnn')
        result_dense = ensemble.predict_single(test_image, method='densenet')
        
        print(f"\n📊 Model Comparison:")
        print(f"Simple CNN: {result_simple['predicted_label']} ({result_simple['confidence']:.2f}%)")
        print(f"DenseNet: {result_dense['predicted_label']} ({result_dense['confidence']:.2f}%)")
        print(f"Ensemble: {result['predicted_label']} ({result['confidence']:.2f}%)")
    
    # Evaluate on test dataset
    print("\n🔍 EVALUATING ENSEMBLE PERFORMANCE...")
    ensemble.evaluate_on_dataset(method='ensemble')
    ensemble.evaluate_on_dataset(method='simple_cnn')
    ensemble.evaluate_on_dataset(method='densenet')
