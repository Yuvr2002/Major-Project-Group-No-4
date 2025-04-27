from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Paths
MODEL_PATH = "backend/model/model.pkl"
UPLOAD_FOLDER = "backend/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = joblib.load(MODEL_PATH)

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route: Home page (upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Route: Predict on uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Image preprocessing
    image = cv2.imread(filepath)
    image = cv2.resize(image, (64, 64)).flatten() / 255.0
    image = np.array([image])  # Shape for model

    # Prediction
    prediction = model.predict(image)[0]
    result = "Defective" if prediction == 1 else "Non-Defective"

    # Show result page with prediction and image
    return render_template('result.html', result=result, image_path=filename)

# Route: To serve uploaded image in result.html
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)