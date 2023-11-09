from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from detecto import core, utils
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}, r"/uploads": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'

try:
    os.mkdir('./uploads')
except OSError as error:
    pass

# Load the Faster R-CNN model for object detection
object_detection_model = core.Model.load('model_weights.pth', ['biji kopi', 'kopi gelondong'])

# Load the MobileNet model for classification
classification_model = load_model('model_mobileNet_100_rlr_v10.h5')

class_dict = {
    0: 'Bentuk Tidak Wajar',
    1: 'Kelainan Lain',
    2: 'Normal',
    3: 'Warna Tidak Wajar'
}

def detect_objects_and_classify(img_path):
    # Detect objects using Faster R-CNN
    image = utils.read_image(img_path)
    labels, boxes, scores = object_detection_model.predict(image)

    detected_classes = []
    for label, box, score in zip(labels, boxes, scores):
        if label == 'biji kopi' or label == 'kopi gelondong':
            # Konversi nilai-nilai dalam box menjadi bilangan bulat
            x, y, x1, y1 = [int(val) for val in box]
            detected_classes.append((label, (x, y, x1, y1), score))

    # Classify the detected objects using MobileNet
    classifications = []
    for label, box, score in detected_classes:
        x, y, x1, y1 = box
        object_image = Image.open(img_path).crop((x, y, x1, y1))
        object_image = object_image.resize((224, 224))
        object_image_array = img_to_array(object_image) # Tidak membagi dengan 255
        object_image_array = np.expand_dims(object_image_array, 0)
        predicted_class = np.argmax(classification_model.predict(object_image_array))
        classification = class_dict[predicted_class]
        classifications.append(classification)

    return classifications

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No image found'})

        image = request.files['file']
        if image.filename == '':
            return jsonify({'error': 'No selected file'})

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        print("File path:", img_path)
        image.save(img_path)

        # Detect objects and classify
        classifications = detect_objects_and_classify(img_path)

        # Mendapatkan URL gambar hasil klasifikasi
        image_url = 'uploads/' + secure_filename(image.filename)

        result = {'uploaded_image': image.filename, 'classifications': classifications, 'image_url': image_url}
        return jsonify(result) # Pastikan respons sesuai dengan format JSON yang diharapkan oleh React

@app.route('/uploads/<filename>', methods=['POST'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)