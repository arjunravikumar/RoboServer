import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response
import numpy as np
import threading
import time
from RoboControls import *
import sys

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")
tracker = None
labelClasses = {}
frame = None
robotControls = None
screenWidth = 1280
screenHeight = 720
GUIMode = True

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

def getDesiredObjectFromFrame(toDetect,img):
    global net
    detections = net.Detect(img)
    for detection in detections:
        if(labelClasses[detection.ClassID]["className"] == toDetect):
            bbox = [int(detection.Left),int(detection.Top),int(detection.Width),int(detection.Height)]
            return True, bbox, img
    return False, None, img

def trackObject(img_array,toDetect):
    global tracker
    ok, bBoxTrack = tracker.update(img_array)
    print(ok)
    objectInFrame = True
    if ok:
        objectInFrame = True
    else:
        objectInFrame = False
    if(GUIMode):
        if ok:
            p1 = (int(bBoxTrack[0]), int(bBoxTrack[1]))
            p2 = (int(bBoxTrack[0] + bBoxTrack[2]), int(bBoxTrack[1] + bBoxTrack[3]))
            cv2.rectangle(img_array, p1, p2, (255,0,0), 2, 1)
            cv2.putText(img_array, "Tracking "+ toDetect , (20,80), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50,170,50),2)
        else :
            cv2.putText(img_array, toDetect + "Not Visible in Vision", (20,20),\
             cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0,0,255),2)
    return objectInFrame,bBoxTrack,img_array

def printStatus(msg):
    print(msg)

def gen_frames(toDetect):
    global frame
    global conditionObj
    global GUIMode
    global camera
    global tracker
    objectFound            = False
    resetTracking          = True
    frameCount = 0
    while True:
        img = camera.Capture()
        if(frameCount%100 == 0 or objectFound == False):
            printStatus("Detecting Object")
            objectFound, bBoxDetect, img = getDesiredObjectFromFrame(toDetect,img)
            printStatus("Object Detected"+str(objectFound))
            resetTracking = True
            frameCount = 1
        frameCount += 1
        img_array = jetson.utils.cudaToNumpy(img)
        if(resetTracking and objectFound):
            printStatus("Reset Initiated"+str(bBoxDetect))
            resetTracking = False
            tracker.init(img_array, bBoxDetect)
            printStatus("Tracking Initialised")
        if(objectFound):
            printStatus("Tracking Object")
            objectFound, bBoxTrack,img_array = trackObject(img_array,toDetect)
            printStatus("Object Status"+str(objectFound))
        if(GUIMode):
            cv2.putText(img_array,'FPS: '+str(net.GetNetworkFPS()), (10,650), \
            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            ret, buffer = cv2.imencode('.jpg', img_array)
            frame = buffer.tobytes()
            with conditionObj:
                conditionObj.notifyAll()
        else:
            printStatus(net.GetNetworkFPS())

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
    global robotControls
    createTracker('MEDIANFLOW')
    with open('label.txt','r') as f:
        lines = f.readlines()
        labelClasses = {}
        for line in lines:
            classVals = line.replace("\n","").split("\t")
            labelClasses[int(classVals[0])] = {"className": classVals[1],"classCategory": classVals[4]}
    robotControls = RoboControls()

def startWebServer():
    app.run("0.0.0.0",port="8000",debug=True)

if(len(sys.argv) > 1 and sys.argv[1] == "False"):
    GUIMode = False
initalisePreProcessingProcedure()
conditionObj = threading.Condition()

generateFrames = threading.Thread(target=gen_frames, name='generateFrames',args=("person",))
generateFrames.start()

if(GUIMode):
    startWebServer()
generateFrames.join()