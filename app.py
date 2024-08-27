from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = Flask(__name__)

# Allow requests from your domain
CORS(app)

# Global variable to store the model
model = None

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/download-model')
def download_model():
    model_path = 'model.tflite'  # Make sure this path points to your actual .tflite file
    try:
        return send_file(model_path, as_attachment=True)  # Serve the file for download
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-model', methods=['POST'])
def upload_model():
    global model
    if 'model' not in request.files:
        return jsonify({'error': 'No model file part'}), 400

    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the model file
    model_path = 'model.tflite'
    file.save(model_path)

    # Load the TFLite model
    try:
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        return jsonify({'message': 'Model uploaded and loaded successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the image
    img = Image.open(file.stream).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    try:
        input_details = model.get_input_details()[0]
        output_details = model.get_output_details()[0]
        model.set_tensor(input_details['index'], img_array)
        model.invoke()
        predictions = model.get_tensor(output_details['index'])
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map prediction to flower category
        flower_name = list(categories.keys())[predicted_class]
        flower_info = categories.get(flower_name, {})

        return jsonify({
            'prediction': flower_name,
            'scientific_name': flower_info['scientific_name'],
            'origin': flower_info['origin'],
            'family': flower_info['family'],
            'symbolism': flower_info['symbolism'],
            'link': flower_info['link'],
            'image': flower_info['image']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
