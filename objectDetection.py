import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response
import numpy as np

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")
tracker = None

def createTracker(tracker_type):
    global tracker
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

def gen_frames():
    global tracker
    objectFound = False
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,650)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2
    while True:
        img = camera.Capture()
        img_array = jetson.utils.cudaToNumpy(img)
        if(objectFound == False):
            detections = net.Detect(img)
            detectionsForImageTracking = []
            img_array = jetson.utils.cudaToNumpy(img)
            for detection in detections:
                print(detection)
                if(detection.ClassID == 1):
                    detection = detections[0]
                    print(detection.Left+detection.Width,detection.Top+detection.Height)
                    bbox = [int(detection.Left),int(detection.Top),int(detection.Width),int(detection.Height)]
                    ok = tracker.init(img_array, bbox)
                    objectFound = True
        ok, bbox = tracker.update(img_array)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img_array, p1, p2, (255,0,0), 2, 1)
        else :
            cv2.putText(img_array, "Tracking failure detected", (100,80), font, 0.75,(0,0,255),2)
            objectFound = False
        cv2.putText(img_array, "Tracking "+ str(detection.ClassID) , (100,20), font, 0.75, (50,170,50),2)
        cv2.putText(img_array,'FPS: '+str(net.GetNetworkFPS()), bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
        ret, buffer = cv2.imencode('.jpg', img_array)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

createTracker('BOOSTING')
app.run("0.0.0.0",port="8000",debug=True)
