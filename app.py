import argparse
from cProfile import label
from distutils.command.config import config
import io
import os
from xmlrpc.client import ProtocolError
from PIL import Image
import cv2
import numpy as np
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import torch
from flask import Flask, render_template, request, redirect, Response, jsonify
from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

detected_letters = []

def detect_letters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(binary, config=custom_config)
    return text.strip()

# def process_frame(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model(img, size=224)
#     detected_objects = results.names[int(results.xyxy[0][:, -1])]
#     return detected_objects

def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return detect_letters(img)

# load model
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.pt", force_reload=True)

model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 

from io import BytesIO

def gen():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)

            # Proses frame untuk mendeteksi huruf
            detected_objects = process_frame(frame)

            # Masukkan hasil deteksi huruf ke dalam list
            detected_letters.append(detected_objects)

            # Hapus elemen terakhir dari list jika jumlah huruf terlalu banyak
            if len(detected_letters) > 10:
                detected_letters.pop(0)

            # Render frame dengan hasil deteksi huruf
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

        cap.release()

        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('webcam.html')

# @app.route('/webcam')
# def webcam():
#     return render_template('webcam.html')

@app.route('/reset')
def reset_letters():
    global detected_letters
    detected_letters = []
    return jsonify(success=True)

@app.route('/letters')
def get_detected_letters():
    global detected_letters
    return jsonify({'letters': detected_letters})

def gen():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            frame = cv2.flip(frame, 1)
            img = frame.copy()
            text = detect_letters(img)
            detected_letters.append(text)
            if len(detected_letters) > 10:
                detected_letters.pop(0)
            word = ''.join(detected_letters)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=224)
            img_BGR = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_RGB2BGR) 
        else:
            break
            cap.release()
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
'''
untuk deteksi video yang di upload                        
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
'''


'''
# ini untuk  ingin mendeteksi menggunakan upload gambar
# tapi harus membuat file html untuk upload gambar
# '''
# @app.route("/upload", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)
#         file = request.files["file"]
#         if not file:
#             return

#         img_bytes = file.read()
#         img = Image.open(io.BytesIO(img_bytes))
#         results = model(img, size=640)

#         # for debugging
#         # data = results.pandas().xyxy[0].to_json(orient="records")
#         # return data

#         results.render()  # updates results.imgs with boxes and labels
#         for img in results.ims:
#             img_base64 = Image.fromarray(img)
#             img_base64.save("static/image0.jpg", format="JPEG")
#         return redirect("static/image0.jpg")

#     # mengarahkan ke file upload.html, 
#     return render_template("upload.html")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Memindahkan model ke perangkat GPU
model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    '''
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    ).autoshape()  # force_reload = recache latest code
    model.eval()
    '''


    app.run(host="0.0.0.0", port=args.port)  