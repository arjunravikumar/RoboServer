import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")


def gen_frames():
    while True:
        print("here2")
        img = camera.Capture()
        detections = net.Detect(img)
        for detection in detections:
            print(detection)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    print("here0")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("here1")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run("0.0.0.0",port="8000",debug=True)
