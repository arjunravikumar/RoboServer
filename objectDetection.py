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
videoLatency = 0.09
currentDirection = "stop"
prevDirection = "stop"
movementEndTime = 0
previousPos = []
MSPerPixel_H = 0.0004
MSPerPixel_V = 0.075
stopPos = []
originalObjectDimension = []
pixelPerFrame_H = 60
pixelPerFrame_V = 10

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

def calibrateLatencyAndMovementValues(bBoxTrack):
    global screenWidth, screenHeight, currentDirection, movementEndTime
    global previousPos, videoLatency, stopPos, MSPerPixel_H
    xMid,yMid = bBoxTrack[0]+(bBoxTrack[2]/2),bBoxTrack[1]+(bBoxTrack[3]/2)
    xMidPrev,yMidPrev = 0,0
    xMidStop,yMidStop = 0,0
    if(len(previousPos)>0):
        xMidPrev,yMidPrev = previousPos[0]+(previousPos[2]/2),previousPos[1]+(previousPos[3]/2)
    if(len(stopPos) > 0):
         xMidStop,yMidStop = stopPos[0]+(stopPos[2]/2),stopPos[1]+(stopPos[3]/2)
    screenCenterX,screenCenterY = screenWidth/2,screenHeight/2
    if(len(previousPos) > 0 and abs(xMidPrev-xMid) > 1):
        print("Diff ",abs(xMidPrev-xMid),currentDirection)
    if(currentDirection == "stop" and len(previousPos) > 0):
        print("Current Pos after stop",xMid,yMid)
        if(abs(xMidPrev-xMid) < 5 and movementEndTime > 0):
#             videoLatency = (time.time() - movementEndTime)
            print("Latency ", round((time.time() - movementEndTime),2))
            if(len(stopPos) > 0 and (stopPos[0]-xMid) > 0):
                diffPixel = abs(stopPos[0]-xMid)
                print("pixel diff " , diffPixel)
#                 MSPerPixel_H = (MSPerPixel_H + (videoLatency/diffPixel))/2
                print("pixeltomillisecondcount" , MSPerPixel_H, (videoLatency/diffPixel))

def getRobotMovementDetails(bBoxTrack):
    global screenWidth, screenHeight, currentDirection, movementEndTime, prevDirection
    global previousPos, videoLatency, stopPos, MSPerPixel_H, originalObjectDimension
    global MSPerPixel_V, pixelPerFrame_H, pixelPerFrame_V
    xMid,yMid = bBoxTrack[0]+(bBoxTrack[2]/2),bBoxTrack[1]+(bBoxTrack[3]/2)
    objectHeight,objectWidth = bBoxTrack[2],bBoxTrack[3]
    originalObjectHeight,originalObjectWidth = originalObjectDimension[0],originalObjectDimension[0]
    screenCenterX,screenCenterY = screenWidth/2,screenHeight/2
    calibrateLatencyAndMovementValues(bBoxTrack)
    previousPos = bBoxTrack[:]
    xMidGroundTruth, yMidGroundTruth = xMid,yMid
    objectGroundTruthHeight, objectGroundTruthWidth = objectHeight,objectWidth
    stopIn= 0
    if(currentDirection == "right"):
        xMidGroundTruth -= pixelPerFrame_H
    elif(currentDirection == "left"):
        xMidGroundTruth += pixelPerFrame_H
    elif(currentDirection == "forward"):
        objectGroundTruthHeight += pixelPerFrame_V
    elif(currentDirection == "backward"):
        objectGroundTruthHeight += pixelPerFrame_V
    elif(currentDirection == "stop" and \
    (movementEndTime + videoLatency) > time.time()):
        if(prevDirection == "left"):
            xMidGroundTruth += ( (1/MSPerPixel_H) * (time.time() - movementEndTime) )
        elif(prevDirection == "right"):
            xMidGroundTruth -= ( (1/MSPerPixel_H) * (time.time() - movementEndTime ) )
        elif(prevDirection == "forward"):
            objectGroundTruthHeight += ( (1/MSPerPixel_V) * (time.time() - movementEndTime) )
        elif(prevDirection == "backward"):
            objectGroundTruthHeight -= ( (1/MSPerPixel_V) * (time.time() - movementEndTime ) )
    print("MovementEndTime :" , movementEndTime + videoLatency," CurrTime :",time.time()\
    ,currentDirection,prevDirection)
    print("Ground Truth H", xMidGroundTruth , "Camera Pos", xMid, currentDirection)
    startMovement = False
    print("Ground Truth V", objectGroundTruthHeight, "original height", originalObjectHeight , "ratio", (objectGroundTruthHeight/originalObjectHeight))
    if((movementEndTime + videoLatency) < time.time()):
        if(abs(xMidGroundTruth - screenCenterX) > (screenWidth/10) and\
        (movementEndTime + videoLatency) < time.time()):
            stopIn = (abs(xMidGroundTruth - screenCenterX)*MSPerPixel_H)
            if(xMidGroundTruth > screenCenterX and currentDirection != "right"):
                printStatus("right " + "cameraPos "+ str(xMidGroundTruth) +" "+ str(stopIn) \
                + " " +str(screenCenterX))
                currentDirection = "right"
                stopPos = []
                startMovement = True
            elif(xMidGroundTruth < screenCenterX and currentDirection != "left"):
                printStatus("left " + "cameraPos "+ str(xMidGroundTruth) +" "+ str(stopIn) \
                + " " +str(screenCenterX))
                currentDirection = "left"
                stopPos = []
                startMovement = True
        elif(abs(objectGroundTruthHeight - originalObjectHeight) > 10):
            stopIn = abs(objectGroundTruthHeight - originalObjectHeight)
            stopIn = stopIn * MSPerPixel_V
            if(objectGroundTruthHeight < originalObjectHeight and currentDirection != "forward"):
                currentDirection = "forward"
                stopPos = []
                startMovement = True
            elif(objectGroundTruthHeight > originalObjectHeight and currentDirection != "backward"):
                currentDirection = "backward"
                stopPos = []
                startMovement = True
    return startMovement, stopIn, xMidGroundTruth, xMid

def moveRobot(bBoxTrack):
    global currentDirection, videoLatency, MSPerPixel_H
    printStatus("bBoxTrack "+str(bBoxTrack))
    startMovement, stopIn, xMidGroundTruth, xMid = getRobotMovementDetails(bBoxTrack)
    messageToSend = {}
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = currentDirection
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    messageToSend["turn"] = ""
    messageToSend["latency"] = videoLatency
    messageToSend["xMid"] = xMid
    messageToSend["xMidGroundTruth"] = xMidGroundTruth
    messageToSend["stopIn"] = stopIn
    messageToSend["MSPerPixel_H"] = MSPerPixel_H
    messageToSend["turn"] = currentDirection
    if(startMovement == True):
        start_time = threading.Timer(stopIn,stopOnCenter)
        start_time.start()
    return startMovement,messageToSend

def stopOnCenter():
    global robotControls, videoLatency, currentDirection, stopPos, previousPos, prevDirection
    global movementEndTime
    if(currentDirection == "stop"):
        return
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
    messageToSend["requestTime"] = time.time()
    messageToSend["latency"] = videoLatency
    prevDirection = currentDirection
    currentDirection = "stop"
    robotControls.send(messageToSend)

def emergencyStop():
    global robotControls, videoLatency, currentDirection, stopPos, prevDirection
    global movementEndTime
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
    messageToSend["requestTime"] = time.time()
    messageToSend["latency"] = videoLatency
    prevDirection = currentDirection
    currentDirection = "stop"
    movementEndTime = 0
    robotControls.send(messageToSend)

def trackSubjectUsingRobot(bBoxTrack):
    global robotControls
    toSend, messageToSend= moveRobot(bBoxTrack)
    if(toSend):
        messageToSend["requestTime"] = time.time()
        robotControls.send(messageToSend)

def gen_frames(toDetect):
    global frame, conditionObj, GUIMode, camera, tracker, currentDirection, previousPos
    global originalObjectDimension
    objectFound             = False
    resetTracking           = True
    bBoxTrack               = None
    bBoxDetect              = None
    frameCount              = 0
    new_frame_time          = 0
    prev_frame_time         = 0
    while True:
        img = camera.Capture()
        objectFound, bBoxDetect, img = getDesiredObjectFromFrame(toDetect,img)
        bBoxTrack = bBoxDetect
        print("Object Found ",objectFound)
        print("Object Location ",bBoxDetect)
        if(objectFound):
            if(len(originalObjectDimension) == 0):
                originalObjectDimension = [bBoxDetect[2],bBoxDetect[3]]
            trackSubjectUsingRobot(bBoxDetect)
        elif(currentDirection != "stop"):
            emergencyStop()
        if(GUIMode):
            img_array = jetson.utils.cudaToNumpy(img)
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