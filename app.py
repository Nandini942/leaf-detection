from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (Replace 'model_path.h5' with your trained model)
model = tf.keras.models.load_model('leaf_disease_model.h5')

# Define class labels (replace with your actual labels)
CLASS_LABELS = ['Healthy', 'Bacterial Spot Disease', 'Curl Virus', 'Early Blight Disease', 'Late Blight Disease', 'Leaf Mold Disease', 'Leaf Mold Disease', 'Leaf Spot Disease', 'Mosaic Virus', 'Target Spot Disease', 'Two Spotted Spider Motes Disease']

# Function to preprocess image and predict disease
def predict_disease(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Adjust size as per your model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image data

    # Predict the disease
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Return the predicted class label
    return CLASS_LABELS[predicted_class[0]]

# Route for the upload page
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Route to handle image upload and disease detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(url_for('upload_page'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_page'))

    # Save the uploaded image
    image_path = f"static/uploads/{file.filename}"
    image_pathOne = f"uploads/{file.filename}"
    file.save(image_path)

    # Detect disease in the image
    prediction = predict_disease(image_path)

    # Return the prediction to the user
    return render_template('result.html', prediction=prediction, image_path=image_pathOne)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
