import jetson.inference
import jetson.utils
from flask import Flask, render_template, Response, send_from_directory
from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("http://192.168.1.166:5000/video_feed")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("file://detectionImg.jpg",true) # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	for detection in detections:
		print(detection)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
