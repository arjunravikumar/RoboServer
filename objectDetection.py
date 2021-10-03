import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response
import numpy as np
import threading
import time
from RoboControls import *
import sys
import asyncio

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")
trackerType = 'MOSSE'
tracker = None
labelClasses = {}
frame = None
robotControls = None
screenWidth = 1280
screenHeight = 720
GUIMode = True
videoLatency = 0.1
currentDirection = "stop"
movementEndTime = 0
previousPos = []
pixelPerMilliseconds = 0.001
stopPos = []

def createNewTracker():
    global trackerType,tracker
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
    printStatus(ok)
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
    print(msg,end = "\n")

def prepareMessageToSend(bBoxTrack):
    global screenWidth, screenHeight, currentDirection, movementEndTime
    global previousPos, videoLatency, stopPos, pixelPerMilliseconds
    messageToSend = {}
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = "no"
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    messageToSend["turn"] = ""
    messageToSend["latency"] = videoLatency
    printStatus("bBoxTrack "+str(bBoxTrack))
    xMid,yMid = bBoxTrack[0]+(bBoxTrack[2]/2),bBoxTrack[1]+(bBoxTrack[3]/2)
    screenCenterX,screenCenterY = screenWidth/2,screenHeight/2
    if(len(previousPos) > 0 and abs(previousPos[0]-xMid) > 1):
        print("Diff ",abs(previousPos[0]-xMid),currentDirection)
    if(currentDirection == "stop" and len(previousPos) > 0):
        if(abs(previousPos[0]-xMid) < 5):
            videoLatency = (time.time() - movementEndTime)
            print("Latency ", round(videoLatency,2))
            if(len(stopPos) > 0):
                diffPixel = abs(stopPos[0]-xMid)
                print("pixel diff " , diffPixel)
                pixelPerMilliseconds = (pixelPerMilliseconds + (videoLatency/diffPixel))/2
                print("pixeltomillisecondcount" , pixelPerMilliseconds)
    previousPos = [xMid,yMid]
    if(abs(xMid - screenCenterX) > (screenWidth/20) and \
        (movementEndTime + videoLatency) < time.time() ):
        stopIn = (abs(xMid - screenCenterX)*pixelPerMilliseconds)
        if(xMid > screenCenterX and currentDirection != "right"):
            printStatus("right " + "cameraPos "+ str(xMid) +" "+ stopIn \
            + " " +str(screenCenterX))
            messageToSend["turn"] = "right"
            stopPos = []
            messageToSend["turn"] = currentDirection
            start_time = threading.Timer(stopIn,stopOnCenter)
            start_time.start()
            return True, messageToSend
        elif(xMid < screenCenterX and currentDirection != "left"):
            printStatus("left " + "cameraPos "+ str(xMid) +" "+ stopIn \
            + " " +str(screenCenterX))
            currentDirection = "left"
            stopPos = []
            messageToSend["turn"] = currentDirection
            start_time = threading.Timer(stopIn,stopOnCenter)
            start_time.start()
            return True, messageToSend
    return False,None

def stopOnCenter():
    global robotControls, videoLatency, currentDirection, stopPos, previousPos
    stopPos = previousPos[:]
    movementEndTime = time.time()
    messageToSend = {}
    messageToSend["reason"] = "Normal Stop"
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = "no"
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    printStatus("stop")
    messageToSend["direction"] = "stop"
    messageToSend["turn"] = ""
    messageToSend["requestTime"] = time.time() * 1000
    messageToSend["latency"] = videoLatency
    currentDirection = "stop"
    robotControls.send(messageToSend)

def emergencyStop():
    global robotControls, videoLatency, currentDirection, stopPos
    stopPos = []
    messageToSend = {}
    messageToSend["reason"] = "Emergency Stop - Object Not In Frame"
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = "no"
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    printStatus("stop")
    messageToSend["direction"] = "stop"
    messageToSend["turn"] = ""
    messageToSend["requestTime"] = time.time() * 1000
    messageToSend["latency"] = videoLatency
    currentDirection = "stop"
    movementEndTime = time.time()
    robotControls.send(messageToSend)

def trackSubjectUsingRobot(bBoxTrack):
    global robotControls
    toSend, data= prepareMessageToSend(bBoxTrack)
    if(toSend):
        data["requestTime"] = time.time() * 1000
        robotControls.send(data)

def gen_frames(toDetect):
    global frame, conditionObj, GUIMode, camera, tracker, currentDirection, previousPos
    objectFound             = False
    resetTracking           = True
    bBoxTrack               = None
    bBoxDetect              = None
    frameCount              = 0
    new_frame_time          = 0
    prev_frame_time         = 0
    while True:
        img = camera.Capture()
        if(frameCount%300 == 0 or objectFound == False):
            previousPos = []
            printStatus("Detecting Object")
            objectFound, bBoxDetect, img = getDesiredObjectFromFrame(toDetect,img)
            printStatus("Object Detected"+str(objectFound))
            resetTracking = True
            frameCount = 1
        img_array = jetson.utils.cudaToNumpy(img)
        if(resetTracking and objectFound):
            printStatus("Reset Initiated"+str(bBoxDetect))
            createNewTracker()
            tracker.init(img_array, bBoxDetect)
            resetTracking = False
            printStatus("Tracking Initialised")
        elif(objectFound):
            frameCount += 1
            printStatus("Tracking Object")
            objectFound, bBoxTrack,img_array = trackObject(img_array,toDetect)
            if(objectFound):
                trackSubjectUsingRobot(bBoxTrack)
            printStatus("Object Status"+str(objectFound))
        elif(currentDirection != "stop"):
            emergencyStop()
        if(GUIMode):
            cv2.putText(img_array,'FPS: '+str(net.GetNetworkFPS()), (10,650), \
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.putText(img_array,'BBOX Track: '+str(bBoxTrack), (10,600), \
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2)
            cv2.putText(img_array,'BBOX Detect: '+str(bBoxDetect), (10,550), \
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2)
            ret, buffer = cv2.imencode('.jpg', img_array)
            frame = buffer.tobytes()
            with conditionObj:
                conditionObj.notifyAll()
        else:
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            printStatus("FPS "+str(fps))

def getFrames():
    global frame, conditionObj
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

def startWebSocketClient():
    global robotControls
    robotControls.startWS()

def initalisePreProcessingProcedure():
    global labelClasses,robotControls
    robotControls = RoboControls()
    with open('label.txt','r') as f:
        lines = f.readlines()
        labelClasses = {}
        for line in lines:
            classVals = line.replace("\n","").split("\t")
            labelClasses[int(classVals[0])] = {"className": classVals[1],"classCategory": classVals[4]}

def startWebServer():
    app.run("0.0.0.0",port="8000",debug=True)

if(len(sys.argv) > 1 and sys.argv[1] == "False"):
    GUIMode = False

initalisePreProcessingProcedure()
conditionObj = threading.Condition()

startWebSocket = threading.Thread(target=startWebSocketClient, name='startWebSocket')
generateFrames = threading.Thread(target=gen_frames, name='generateFrames',args=("person",))

startWebSocket.start()
generateFrames.start()

if(GUIMode):
    startWebServer()

startWebSocket.join()
generateFrames.join()