import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response
import numpy as np
import threading
import time

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")
tracker = None
labelClasses = {}
frame = None

def createTracker(trackerType):
    global tracker
    if trackerType == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if trackerType == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    if trackerType == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    if trackerType == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if trackerType == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if trackerType == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    if trackerType == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if trackerType == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

def gen_frames(toDetect):
    global tracker
    global frame
    global conditionObj
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
                if(labelClasses[detection.ClassID]["className"] == toDetect):
                    detection = detections[0]
                    print(detection.Left+detection.Width,detection.Top+detection.Height)
                    boundingBox = [int(detection.Left),int(detection.Top),int(detection.Width),int(detection.Height)]
                    ok = tracker.init(img_array, boundingBox)
                    objectFound = True
            cv2.putText(img_array,'FPS: '+str(net.GetNetworkFPS()), bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
        else:
            ok, boundingBox = tracker.update(img_array)
            if ok:
                p1 = (int(boundingBox[0]), int(boundingBox[1]))
                p2 = (int(boundingBox[0] + boundingBox[2]), int(boundingBox[1] + boundingBox[3]))
                cv2.rectangle(img_array, p1, p2, (255,0,0), 2, 1)
            else :
                cv2.putText(img_array, toDetect + "Not Visible in Vision", (20,20), font, 0.50,(0,0,255),2)
                objectFound = False
            cv2.putText(img_array, "Tracking "+ toDetect , (20,80), font, 0.50, (50,170,50),2)
        ret, buffer = cv2.imencode('.jpg', img_array)
        frame = buffer.tobytes()
        print(net.GetNetworkFPS())
        with conditionObj:
            conditionObj.notifyAll()

def getFrames():
    global frame
    global conditionObj
    while True:
        with conditionObj:
            conditionObj.wait()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(getFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def initalisePreProcessingProcedure():
    global labelClasses
    createTracker('BOOSTING')
    with open('label.txt','r') as f:
        lines = f.readlines()
        labelClasses = {}
        for line in lines:
            classVals = line.replace("\n","").split("\t")
            labelClasses[int(classVals[0])] = {"className": classVals[1],"classCategory": classVals[4]}

def startWebServer():
    app.run("0.0.0.0",port="8000",debug=True)

initalisePreProcessingProcedure()
conditionObj = threading.Condition()

generateFrames = threading.Thread(target=gen_frames, name='generateFrames',args=("person",))
generateFrames.start()

startWebServer()
generateFrames.join()