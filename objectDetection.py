import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")


def gen_frames():
    while True:
        img = camera.Capture()
        detections = net.Detect(img)
        for detection in detections:
            print(detection)
        img_array = jetson.utils.cudaToNumpy(img)
        ret, buffer = cv2.imencode('.jpg', img_array)
        frame = buffer.tobytes()
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,600)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        cv2.putText(img,'FPS: '+str(net.GetNetworkFPS()), bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run("0.0.0.0",port="8000",debug=True)
