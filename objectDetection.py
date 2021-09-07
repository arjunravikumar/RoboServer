import jetson.inference
import jetson.utils
from flask import Flask, render_template, Response, send_from_directory
from flask_cors import *
import cv2

app = Flask(__name__)
CORS(app, supports_credentials=True)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("http://192.168.1.166:5000/video_feed")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("file://detectionImg.jpg",true) # 'my_video.mp4' for file

def gen():
	while True:
		img = camera.Capture()
		detections = net.Detect(img)
		for detection in detections:
			print(detection)
		display.Render(img)
		display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
		imgOut = jetson.utils.cudaToNumpy(img)
		yeild(imgOut)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)
