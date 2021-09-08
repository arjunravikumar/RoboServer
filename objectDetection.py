import jetson.inference
import jetson.utils
from flask import Flask, render_template, Response, send_from_directory
from flask_cors import *
import cv2

app = Flask(__name__)
CORS(app, supports_credentials=True)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("file://detectionImg.jpg") # 'my_video.mp4' for file

def gen():
	while True:
		img = camera.Capture()
		detections = net.Detect(img)
		for detection in detections:
			print(detection)
		display.Render(img)
		display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
		imgOut = jetson.utils.cudaToNumpy(img)
		yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + imgOut + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

app.run(debug=True)
