from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = 'extracted'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_FOLDER'] = EXTRACTED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(EXTRACTED_FOLDER):
    os.makedirs(EXTRACTED_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        extracted_paths = detect_and_extract_objects(file_path)
        extracted_files = [os.path.basename(path) for path in extracted_paths]
        return render_template('result.html', extracted_files=extracted_files)

def detect_and_extract_objects(file_path):
    # Read the image
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted_paths = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        extracted_image = image[y:y+h, x:x+w]
        extracted_path = os.path.join(app.config['EXTRACTED_FOLDER'], f'extracted_{i}.png')
        cv2.imwrite(extracted_path, extracted_image)
        extracted_paths.append(extracted_path)
    
    return extracted_paths

@app.route('/extracted/<filename>')
def get_extracted_file(filename):
    return send_from_directory(app.config['EXTRACTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
