import jetson.inference
import jetson.utils
import cv2
from flask import Flask, render_template, Response
from sort import *

app = Flask(__name__)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")

mot_tracker = Sort()

def gen_frames():
    while True:
        print("here")
        img = camera.Capture()
        detections = net.Detect(img)
        detectionsForImageTracking = []
        for detection in detections:
            detectedObjInImg = np.array([detection.Left,detection.Bottom,detection.Right,detection.Top,detection.ClassID])
            detectionsForImageTracking.append(detectionsForImageTracking)
        if(len(detectionsForImageTracking) > 0):
            detectionsForImageTracking = np.array(detectionsForImageTracking)
            tracked_objects = mot_tracker.update(detectionsForImageTracking)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        img_array = jetson.utils.cudaToNumpy(img)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,600)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
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

app.run("0.0.0.0",port="8000",debug=True)
