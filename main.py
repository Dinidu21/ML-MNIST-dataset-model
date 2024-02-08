import re
from flask import Flask, render_template, request, jsonify
from mlengine import transform_image, get_prediction  # Fix the function name
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

ALLOWED_FILE_EXTENSION_TYPES = {'png', 'jpg', 'jpeg'}

def is_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE_EXTENSION_TYPES

@app.route('/predict', methods=['POST','GET'])
def predictfn():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not is_allowed(file.filename):
            return jsonify({'error': 'not a valid file'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return render_template('index2.html') + str(data)
        
        except Exception as e:  # Add the exception parameter
            return jsonify({'error': 'error at ml pipeline'})
    return render_template('index.html')


app.run(debug=True)