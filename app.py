from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from detecto import core, utils
from flask_cors import CORS
from base64 import b64encode
import cv2
import matplotlib.pyplot as plt
import numpy as np
from detecto.visualize import show_labeled_image, plot_prediction_grid
import keras
import tensorflow as tf
from keras.models import Sequential, Model
import time
import base64


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

try:
    os.mkdir('./uploads')
except OSError as error:
    pass

#Import object detection Model
model_od = core.Model.load('model_weights.pth', ['Biji Kopi', 'Kopi Gelondong'])

#Import Classification Model
model_clf = keras.models.load_model('model_mobileNet_100_rlr_v10.h5')

class_dict = {
    0: 'Bentuk Tidak Wajar',
    1: 'Kelainan Lain',
    2: 'Normal',
    3: 'Warna Tidak Wajar'
}

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

        reverse_mapping_col = {
            0: (255, 255, 0),  # Bentuk Tidak Wajar (Kuning)
            1: (0, 0, 0),      # Kelainan Lain (Hitam)
            2: (0, 255, 0),    # Normal (Hijau)
            3: (255, 0, 0),    # Warna Tidak Wajar (Merah)
        }

        def mapper_col(value):
            return reverse_mapping_col[value]

        reverse_mapping = {
            0: 'Bentuk Tidak Wajar',
            1: 'Kelainan Lain',
            2: 'Normal',
            3: 'Warna Tidak Wajar'
        }

        def mapper(value):
            return reverse_mapping[value]

        thresh = 0.5  # Threshold untuk mengontrol sejauh mana hasil deteksi diterima

        # Membaca gambar dari file
        image_ = utils.read_image(img_path)

        # Melakukan prediksi objek pada gambar menggunakan model objek deteksi (model_od)
        predictions = model_od.predict(image_)
        labels, boxes, scores = predictions

        # Mencari indeks dari prediksi yang memiliki skor lebih tinggi dari threshold
        filtered_indices = np.where(scores > thresh)

        # Mengambil kotak deteksi yang lolos filter
        filtered_boxes = boxes[filtered_indices]

        # Memproses setiap kotak deteksi yang lolos filter
        area = filtered_boxes.numpy()
        result = []

        for a in area:
            # Mendapatkan koordinat kotak deteksi
            xmin = np.floor(a[0]).astype(int)
            ymin = np.floor(a[1]).astype(int)
            xmax = np.floor(a[2]).astype(int)
            ymax = np.floor(a[3]).astype(int)

            # Memotong bagian gambar sesuai dengan kotak deteksi
            crop_img = image_[ymin:ymax, xmin:xmax]

            # Menyimpan potongan gambar ke file sementara (ganti path sesuai kebutuhan)
            cv2.imwrite("./temp.png", crop_img)

            # Memuat gambar dari file sementara dan menyesuaikannya ke ukuran yang diinginkan
            img_ = load_img("./temp.png", target_size=(224, 224))
            image = img_to_array(img_)

            # Memprediksi jenis objek dalam potongan gambar menggunakan model klasifikasi (model_clf)
            prediction_image = np.array(crop_img)
            prediction_image = np.expand_dims(image, axis=0)
            prediction = model_clf.predict(prediction_image)

            # Mendapatkan nilai tertinggi dari prediksi dan indeks kelas yang sesuai
            score = round(np.max(prediction), 2)
            value = np.argmax(prediction)

            # Menambahkan hasil prediksi ke dalam daftar hasil
            result.append(reverse_mapping[value])

            # Membuat label untuk kotak deteksi dengan jenis objek dan skornya
            label = mapper(value) + "(" + str(score) + ")"

            # Menampilkan label dan kotak deteksi pada gambar
            #cv2.putText(image_, label, (xmin + 13, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, mapper_col(value), 2)
            #cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), mapper_col(value), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1  # Ukuran font yang lebih besar
            font_thickness = 3  # Ketebalan font yang lebih tebal
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = xmin + 13
            text_y = ymin + 30

            # Tambahkan label dengan font yang lebih besar dan tebal
            cv2.putText(image_, label, (text_x, text_y), font, font_scale, mapper_col(value), font_thickness)
            cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), mapper_col(value), 3)


        # Menampilkan gambar asli dengan kotak deteksi dan label
        timestamp = str(int(time.time()))  # Generate a unique timestamp
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted.png')
        cv2.imwrite(save_path, cv2.cvtColor(image_, cv2.COLOR_RGB2BGR))
        # Convert the saved image to base64
        with open(save_path, 'rb') as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        return jsonify({'success': 'Prediction successful', 'image_data': base64_image, 'classifications': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
