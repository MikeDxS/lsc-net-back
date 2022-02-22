import os
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from keras_video import VideoFrameGenerator
import numpy as np
import shutil

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '.\\video'
ALLOWED_EXTENSIONS = {'mp4'}
SIZE = (300, 300)
CHANNELS = 3
NBFRAME = 16
BS = 1

model = tf.keras.models.load_model('./model/conv3d.h5')

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return jsonify({"message": "hola"})

@app.route("/api/process", methods=['POST'])
def process():
    if 'video' not in request.files:
        return jsonify({ 'message': 'The request doesnt have a video file' }), 400
    jsonBody = request.form
    video = request.files['video']
    video_path = UPLOAD_FOLDER
    if video.filename == '':
        return jsonify({ 'message': 'The video file doesnt have a valid name' }), 400
    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        class_name = jsonBody['userName']
        os.mkdir(os.path.join(UPLOAD_FOLDER, class_name))
        video_path = os.path.join(UPLOAD_FOLDER, class_name, filename)
        video.save(video_path)
        glob_pattern = video_path.replace(f"{class_name}", '{classname}')
        train = VideoFrameGenerator(
            classes=[class_name], 
            glob_pattern=glob_pattern,
            nb_frames=NBFRAME,
            shuffle=True,
            batch_size=BS,
            target_shape=SIZE,
            nb_channel=CHANNELS,
            transformation=None)
        X, Y = train.next()
        out_np = model(X)
        classes = out_np[0].numpy()
        classes = np.around(classes, 4)
        y_hat = np.argmax(classes)
        shutil.rmtree(os.path.join(UPLOAD_FOLDER, class_name))
        return jsonify({"class": int(y_hat), "predictions": classes.tolist()})

if __name__ == '__main__':
    if not os.path.exists('./video'):
        os.mkdir('./video')
    app.run(host='127.0.0.1')