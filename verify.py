import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = load_model('leaf_disease_model.h5')

# Define class labels
CLASS_LABELS = ['Healthy', 'Disease_A', 'Disease_B', 'Disease_C']  # Update based on your dataset

# Load and preprocess an example image
image_path = 'path_to_an_example_leaf_image.jpg'
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
predictions = model.predict(img_array)
predicted_class = CLASS_LABELS[np.argmax(predictions)]

print(f"Predicted Class: {predicted_class}")