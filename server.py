import os
from pathlib import Path
import tensorflow as tf
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
from keras_video import VideoFrameGenerator
import numpy as np
import shutil
from typing import Optional
from pydantic import BaseModel


class Item(BaseModel):
    userName: str
    video: UploadFile
    description: Optional[str] = None

app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_FOLDER = './video'
ALLOWED_EXTENSIONS = {'mp4'}
SIZE = (300, 300)
CHANNELS = 3
NBFRAME = 16
BS = 1

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

model = tf.keras.models.load_model('./model/conv3d.h5')

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
async def home():
    return {"message": "hola"}

@app.post("/api/process")
async def process(userName: str = Form(...), video: UploadFile = File(...)):
    if video == None:
        return { 'message': 'The request doesnt have a video file' }, 400
    video_path = UPLOAD_FOLDER
    if video.filename == '':
        return { 'message': 'The video file doesnt have a valid name' }, 400
    if video and allowed_file(video.filename):
        print('lee nombre video')
        filename = secure_filename(video.filename)
        class_name = userName
        os.mkdir(os.path.join(UPLOAD_FOLDER, class_name))
        video_path = os.path.join(UPLOAD_FOLDER, class_name, filename)
        destination = Path(video_path)
        try:
            with destination.open("wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        finally:
            video.file.close()
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
        return {"class": int(y_hat), "predictions": classes.tolist()}