import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response
import numpy as np
import threading
import time
from sort import *
from RoboControls import *

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")
tracker = None
labelClasses = {}
frame = None
mot_tracker = Sort()
robotControls = None

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

def genFramesWithSort(toDetect):
    global tracker
    global frame
    global conditionObj
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,650)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2
    while True:
        img = camera.Capture()
        img_array = jetson.utils.cudaToNumpy(img)
        detections = net.Detect(img)
        detectionsForImageTracking = []
        img_array = jetson.utils.cudaToNumpy(img)
        if len(detections)> 0:
            print(detections)
            dets = []
            for detection in detections:
                dets.append(numpy.array([detection.Left,detection.Top,detection.Right,\
                detection.Bottom,detection.ClassID]))
            tracked_objects = mot_tracker.update(numpy.array(dets))
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img_array.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img_array.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img_array.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img_array.shape[1])
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h),
                             color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60,
                             y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)),
                            (x1, y1 - 10), font,
                            0.5, (255,255,255), 3)
        cv2.putText(img_array,'FPS: '+str(net.GetNetworkFPS()), bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
        ret, buffer = cv2.imencode('.jpg', img_array)
        frame = buffer.tobytes()
        print(net.GetNetworkFPS(), end='\r')
        with conditionObj:
            conditionObj.notifyAll()

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
        print(net.GetNetworkFPS(), end='\r')
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
    global robotControls
    createTracker('BOOSTING')
    with open('label.txt','r') as f:
        lines = f.readlines()
        labelClasses = {}
        for line in lines:
            classVals = line.replace("\n","").split("\t")
            labelClasses[int(classVals[0])] = {"className": classVals[1],"classCategory": classVals[4]}
    robotControls = RoboControls()

def startWebServer():
    app.run("0.0.0.0",port="8000",debug=True)

initalisePreProcessingProcedure()
conditionObj = threading.Condition()

generateFrames = threading.Thread(target=gen_frames, name='generateFrames',args=("person",))
generateFrames.start()

startWebServer()
generateFrames.join()