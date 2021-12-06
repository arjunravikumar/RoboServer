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
MSPerPixel_H = 0.0002
MSPerPixel_V = 0.01
stopPos = []
pixelPerFrame_H = 60
pixelPerFrame_V = 10
calibrationMode = True
calibrationStartTime = 0
calibrationVariables = {"stopTime":0, "movementEndTime": 0, "stopPos" : [], "previousPos" :[],\
                        "currentDirection" :"", "previousDirection" : "start"}
objectFound = False
objectLostTime = 0
toDetect = ""
searchMode = 1

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

def searchMovement():
    global robotControls, videoLatency, MSPerPixel_H
    messageToSend = {}
    messageToSend["reason"] = "Search Mode"
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = "left"
    messageToSend["speed"] = 60
    messageToSend["rads"] = 0.5
    messageToSend["turn"] = "left"
    messageToSend["latency"] = videoLatency
    messageToSend["stopIn"] = 0
    messageToSend["MSPerPixel_H"] = MSPerPixel_H
    messageToSend["MSPerPixel_V"] = MSPerPixel_V
    messageToSend["requestTime"] = time.time()
    robotControls.send(messageToSend)
    start_time = threading.Timer(5,stopSearchMovement)
    start_time.start()

def stopSearchMovement():
    global robotControls, videoLatency, MSPerPixel_H
    global calibrationVariables,toDetect,objectFound, searchMode
    if(objectFound == True or searchMode != 0):
        return
    messageToSend = {}
    messageToSend["reason"] = "Searching Stop"
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = "no"
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    printStatus("stop")
    messageToSend["direction"] = "stop"
    messageToSend["turn"] = ""
    messageToSend["requestTime"] = time.time()
    messageToSend["latency"] = videoLatency
    messageToSend["MSPerPixel_H"] = MSPerPixel_H
    messageToSend["MSPerPixel_V"] = MSPerPixel_V
    robotControls.send(messageToSend)
    toDetect = ""
    searchMode = -1

def calibrateMovement(direction,turn,stopIn):
    global robotControls, videoLatency, MSPerPixel_H
    messageToSend = {}
    messageToSend["reason"] = "Calibration "+direction+" "+ turn
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = direction
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    messageToSend["turn"] = turn
    messageToSend["latency"] = videoLatency
    messageToSend["stopIn"] = 0.1
    messageToSend["MSPerPixel_H"] = MSPerPixel_H
    messageToSend["MSPerPixel_V"] = MSPerPixel_V
    messageToSend["requestTime"] = time.time()
    robotControls.send(messageToSend)
    start_time = threading.Timer(stopIn,stopCalibrationMovement)
    start_time.start()

def stopCalibrationMovement(initialCall = False):
    global robotControls, videoLatency, MSPerPixel_H
    global calibrationVariables
    messageToSend = {}
    messageToSend["reason"] = "Calibration Stop"
    messageToSend["type"] = "mobility"
    messageToSend["direction"] = "no"
    messageToSend["speed"] = 100
    messageToSend["rads"] = 0.5
    printStatus("stop")
    messageToSend["direction"] = "stop"
    messageToSend["turn"] = ""
    messageToSend["requestTime"] = time.time()
    messageToSend["latency"] = videoLatency
    messageToSend["MSPerPixel_H"] = MSPerPixel_H
    messageToSend["MSPerPixel_V"] = MSPerPixel_V
    if(initialCall == False):
        calibrationVariables["currentDirection"] = "stop"
        calibrationVariables["stopPos"] = calibrationVariables["previousPos"]
        calibrationVariables["movementEndTime"] = time.time()
    robotControls.send(messageToSend)

def calibration(bBoxTrack,currTime):
    global calibrationMode, calibrationStartTime, calibrationVariables
    diffTime = 1
    calibrateLatencyAndMovementValues(bBoxTrack)
    calibrationVariables["previousPos"] = bBoxTrack[:]
    if(calibrationStartTime == 0):
        calibrationStartTime = currTime
        stopCalibrationMovement(True)
    elif(currTime > (calibrationStartTime + (diffTime * 1 )) \
            and calibrationVariables["previousDirection"] == "start"):
        calibrationVariables["previousDirection"] = "left1"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("left", "left" , 0.05)
    elif(currTime > (calibrationStartTime + (diffTime * 2 )) \
            and calibrationVariables["previousDirection"] == "left1"):
        calibrationVariables["previousDirection"] = "right1"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("right", "right", 0.05)
    elif(currTime > (calibrationStartTime + (diffTime * 3 ))\
            and calibrationVariables["previousDirection"] == "right1"):
        calibrationVariables["previousDirection"] = "right2"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("right", "right", 0.05)
    elif(currTime > (calibrationStartTime + (diffTime * 4 ))\
            and calibrationVariables["previousDirection"] == "right2"):
        calibrationVariables["previousDirection"] = "left2"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("left", "left", 0.05)
    elif(currTime > (calibrationStartTime + (diffTime * 5 ))\
            and calibrationVariables["previousDirection"] == "left2"):
        calibrationVariables["previousDirection"] = "forward1"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("forward", "noturn", 0.1)
    elif(currTime > (calibrationStartTime + (diffTime * 6 ))\
            and calibrationVariables["previousDirection"] == "forward1"):
        calibrationVariables["previousDirection"] = "backward1"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("backward", "noturn", 0.1)
    elif(currTime > (calibrationStartTime + (diffTime * 7 ))\
            and calibrationVariables["previousDirection"] == "backward1"):
        calibrationVariables["previousDirection"] = "forward2"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("forward", "noturn", 0.1)
    elif(currTime > (calibrationStartTime + (diffTime * 8 ))\
            and calibrationVariables["previousDirection"] == "forward2"):
        calibrationVariables["previousDirection"] = "backward2"
        calibrationVariables["currentDirection"] = calibrationVariables["previousDirection"][:-1]
        calibrateMovement("backward", "noturn", 0.1)
    elif (currTime > (calibrationStartTime + (diffTime * 9 ))\
            and calibrationVariables["previousDirection"] == "backward2"):
        calibrationMode = False

def calibrateLatencyAndMovementValues(bBoxTrack):
    global screenWidth, screenHeight, calibrationVariables
    global videoLatency, MSPerPixel_H, MSPerPixel_V
    xMid,yMid = bBoxTrack[0]+(bBoxTrack[2]/2),bBoxTrack[1]+(bBoxTrack[3]/2)
    xMidPrev,yMidPrev = 0,0
    xMidStop,yMidStop = 0,0
    currHeight,currWidth = bBoxTrack[2],bBoxTrack[3]
    heightPrev = 0
    heightStop = 0
    print(calibrationVariables)
    if("ward" in calibrationVariables["previousDirection"]):
        if(len(calibrationVariables["previousPos"])>0):
            xHeightPrev = calibrationVariables["previousPos"][2]
        if(len(calibrationVariables["stopPos"]) > 0):
            heightStop = calibrationVariables["stopPos"][2]
        if(len(calibrationVariables["previousPos"]) > 0 and abs(xMidPrev-currHeight) > 1):
            print("Diff ",abs(heightPrev-currHeight),currentDirection)
        if(calibrationVariables["currentDirection"] == "stop" and len(calibrationVariables["previousPos"]) > 0):
            print("Current Pos after stop",currHeight,currWidth)
            if(abs(heightPrev-currHeight) < 5 and calibrationVariables["movementEndTime"] > 0):
                # videoLatency = (videoLatency + (time.time() - calibrationVariables["movementEndTime"]))/2
                print("Latency ", round((time.time() - calibrationVariables["movementEndTime"]),2))
                calibrationVariables["movementEndTime"] = 0
                if(len(calibrationVariables["stopPos"]) > 0 and (heightStop-currHeight) > 0):
                    diffPixel = abs(heightStop-currHeight)
                    print("pixel diff " , diffPixel)
                    # MSPerPixel_V = (MSPerPixel_V + (videoLatency/diffPixel))/2
                    print("pixeltomillisecondcount" , MSPerPixel_V, (videoLatency/diffPixel))
    else:
        if(len(calibrationVariables["previousPos"])>0):
            xMidPrev,yMidPrev = calibrationVariables["previousPos"][0]+(calibrationVariables["previousPos"][2]/2),\
                                calibrationVariables["previousPos"][1]+(calibrationVariables["previousPos"][3]/2)
        if(len(calibrationVariables["stopPos"]) > 0):
            xMidStop = calibrationVariables["stopPos"][0]+(calibrationVariables["stopPos"][2]/2)
            yMidStop = calibrationVariables["stopPos"][1]+(calibrationVariables["stopPos"][3]/2)
        screenCenterX,screenCenterY = screenWidth/2,screenHeight/2
        if(len(calibrationVariables["previousPos"]) > 0 and abs(xMidPrev-xMid) > 1):
            print("Diff ",abs(xMidPrev-xMid),currentDirection)
        if(calibrationVariables["currentDirection"] == "stop" and len(calibrationVariables["previousPos"]) > 0):
            print("Current Pos after stop",xMid,yMid)
            if(abs(xMidPrev-xMid) < 10 and calibrationVariables["movementEndTime"] > 0):
                # videoLatency = (videoLatency + (time.time() - calibrationVariables["movementEndTime"]))/2
                print("Latency ", round((time.time() - calibrationVariables["movementEndTime"]),2))
                calibrationVariables["movementEndTime"] = 0
                if(len(calibrationVariables["stopPos"]) > 0 and (xMidStop-xMid) > 0):
                    diffPixel = abs(xMidStop-xMid)
                    print("pixel diff " , diffPixel)
                    # MSPerPixel_H = (MSPerPixel_H + (videoLatency/diffPixel))/2
                    print("pixeltomillisecondcount" , MSPerPixel_H, (videoLatency/diffPixel))

def getRobotMovementDetails(bBoxTrack):
    global screenWidth, screenHeight, currentDirection, movementEndTime, prevDirection
    global previousPos, videoLatency, stopPos, MSPerPixel_H
    global MSPerPixel_V, pixelPerFrame_H, pixelPerFrame_V
    xMid,yMid = bBoxTrack[0]+(bBoxTrack[2]/2),bBoxTrack[1]+(bBoxTrack[3]/2)
    objectHeight,objectWidth = bBoxTrack[2],bBoxTrack[3]
    screenCenterX,screenCenterY = screenWidth/2,screenHeight/2
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
    elif(currentDirection == "stop" and (movementEndTime + videoLatency) > time.time()):
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
    if((movementEndTime + videoLatency) < time.time() and currentDirection == "stop"):
        if(abs(xMidGroundTruth - screenCenterX) > (screenWidth/10) and\
        (movementEndTime + videoLatency) < time.time()):
            stopIn = (abs(xMidGroundTruth - screenCenterX)*MSPerPixel_H)
            if(xMidGroundTruth > screenCenterX):
                printStatus("right " + "cameraPos "+ str(xMidGroundTruth) +" "+ str(stopIn) \
                + " " +str(screenCenterX))
                currentDirection = "right"
                stopPos = []
                startMovement = True
            elif(xMidGroundTruth < screenCenterX):
                printStatus("left " + "cameraPos "+ str(xMidGroundTruth) +" "+ str(stopIn) \
                + " " +str(screenCenterX))
                currentDirection = "left"
                stopPos = []
                startMovement = True
        elif(abs(objectGroundTruthHeight - (screenHeight/2)) > screenHeight/50):
            stopIn = abs(objectGroundTruthHeight - (screenHeight/2))
            stopIn = stopIn * MSPerPixel_V
            if(objectGroundTruthHeight < (screenHeight/2)):
                currentDirection = "forward"
                stopPos = []
                startMovement = True
            elif(objectGroundTruthHeight > (screenHeight/2)):
                currentDirection = "backward"
                stopPos = []
                startMovement = True
    return startMovement, stopIn, xMidGroundTruth, xMid

def moveRobot(bBoxTrack):
    global currentDirection, videoLatency, MSPerPixel_H, MSPerPixel_V
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
    messageToSend["MSPerPixel_V"] = MSPerPixel_V
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
    messageToSend["MSPerPixel_H"] = MSPerPixel_H
    messageToSend["MSPerPixel_V"] = MSPerPixel_V
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

def gen_frames():
    global frame, conditionObj, GUIMode, camera, tracker, currentDirection, previousPos
    global calibrationMode, calibrationStartTime, searchMode, toDetect, objectLostTime
    global objectFound
    resetTracking           = True
    bBoxTrack               = None
    bBoxDetect              = None
    frameCount              = 0
    new_frame_time          = 0
    prev_frame_time         = 0
    while True:
        print(toDetect,objectLostTime)
        if(toDetect == ""):
            toDetect = input("Enter the object to find")
        img = camera.Capture()
        objectFound, bBoxDetect, img = getDesiredObjectFromFrame(toDetect,img)
        bBoxTrack = bBoxDetect
        print("Object Found ",objectFound)
        print("Object Location ",bBoxDetect)
        if(objectFound == False and searchMode == -1):
            if(objectLostTime == 0):
                objectLostTime = time.time()
            elif(time.time() - objectLostTime > 5):
                toDetect = ""
                searchMode = 1
                objectLostTime = 0
        if(objectFound == True):
            if (searchMode == 1):
                calibrationMode = False
                searchMode = -1
                objectLostTime = 0
                emergencyStop()
        if(searchMode == 1 and toDetect != ""):
            searchMovement()
            searchMode = 0
        if(calibrationMode):
            if(objectFound):
                calibration(bBoxDetect,time.time())
            # else:
            #     calibrationStartTime = 0
        else:
            if(objectFound):
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
generateFrames = threading.Thread(target=gen_frames, name='generateFrames')

startWebSocket.start()
generateFrames.start()

if(GUIMode):
    startWebServer()

startWebSocket.join()
generateFrames.join()