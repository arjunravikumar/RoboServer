import jetson.inference
import jetson.utils
import cv2
import time
import threading
from flask import Response, Flask

global video_frame
video_frame = None

global thread_lock
thread_lock = threading.Lock()

app = Flask(__name__)

def captureFrames():
    global video_frame, thread_lock
    camera = jetson.utils.videoSource("rtsp://192.168.1.166:8554/unicast")

    while True and video_capture.isOpened():
        frame = camera.Capture()
        with thread_lock:
            video_frame = frame.copy()

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    video_capture.release()

def encodeFrame():
    global thread_lock
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
    while True:
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            detections = net.Detect(video_frame)
            for detection in detections:
                print(detection)
            return_key, buffer = cv2.imencode(".jpg", video_frame)
            encoded_image = buffer.tobytes()
            if not return_key:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_image) + b'\r\n')

@app.route("/")
def streamFrames():
    return Response(encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    process_thread = threading.Thread(target=captureFrames)
    process_thread.daemon = True
    process_thread.start()
    app.run("0.0.0.0", port="8000")

